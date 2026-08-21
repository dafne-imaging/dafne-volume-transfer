import numpy as np
import torch

from dafne_sam2.matching import (
    build_multiclass_bags, multiclass_score_maps, multiclass_masks,
    _positional_channels,
)
from dafne_sam2.preprocessing import body_mask_2d
from dafne_sam2.semi_automatic.slice_matching import (
    align_region, body_geometry, build_body_geometry, build_roi_pool_weights,
    build_roi_region_masks, build_slice_bags, fit_z_map, normalized_geometry,
    query_support_slice_similarity, query_window_from_z_map, roi_reference_centroid,
)
from dafne_sam2.utils import _to_grid

from .types import AcceptedMatch, MatchCandidate

class SliceMatchSession:
    """
    Semi-automatic anchor collection, one query slice at a time. Loop the caller (GUI)
    drives, per roi: suggest_support(roi, q_idx) -> user confirms one ->
    accept_candidate(candidate) -> anchor mask -> next_query_slice(roi) proposes where to go next ->
    repeat; anchors()/z_bounds() then feed api.propagate.

    Each added pair refines the affine z map (support_idx = a*query_idx + b), which
    narrows the support search range and predicts the roi's query extent.

    Two pooling modes, measured against each other: enhanced=False (base) pools every roi
    over the binary union of its support masks, applied to the query at the same pixel
    coordinates. enhanced=True pools over the roi's exclusive core (build_roi_pool_weights)
    and moves that region into the query slice's own body frame first (align_region).

    Two matching modality is developed: 
        1. suggest_support(): compared regions to find the most similar support slice 
        2. accept_candidate(): more accurately matched
    """

    def __init__(self, seg, support_slices: dict, support_masks: dict, query_slices: dict,
                 thr_hi: float = 0.7, thr_lo: float = 0.3,
                 body_thresh: float = 10.0, body_min_px: int = 50,
                 score_thresh: float = 0.0, score_mode: str = 'sum_margin',
                 search_radius: int = 8, min_gap: int = 3,
                 enhanced: bool = False, align: bool = None, soft_pool: bool = None,
                 pool_gamma: float = 1.0):
        """
        search_radius: support slices searched either side of the z-map prediction (<=0
        searches all). min_gap: query slices kept between anchors. align/soft_pool: the
        two halves of enhanced, each defaulting to it when None.
        """
        self.seg = seg
        self.support_slices = support_slices
        self.support_masks = support_masks
        self.query_slices = query_slices
        self.thr_hi, self.thr_lo = thr_hi, thr_lo
        self.body_thresh, self.body_min_px = body_thresh, body_min_px
        self.score_thresh, self.score_mode = score_thresh, score_mode
        self.search_radius, self.min_gap = search_radius, min_gap
        self.enhanced = bool(enhanced)
        self.align = self.enhanced if align is None else bool(align)
        self.soft_pool = self.enhanced if soft_pool is None else bool(soft_pool)

        self.q_bounds = (min(query_slices), max(query_slices))
        self.regions = (build_roi_pool_weights(support_masks, gamma=pool_gamma)
                        if self.soft_pool else build_roi_region_masks(support_masks))
        self._slice_bags = None          # built on first use, all rois at once
        self._bag_cache = {}             # support slice_idx -> multiclass bags
        self._supp_geo = None            # support slice_idx -> body (cy, cx, scale)
        self._query_geo = {}             # query slice_idx -> body (cy, cx, scale) or None
        self._roi_centroid = {}          # roi_name -> the frame its region is drawn in
        self._scale = None               # query body size / support body size, no pair yet
        self._pairs = {}                 # roi_name -> list[(query_idx, support_idx)], confirmed query-support pairs
        self._prompts = {}               # roi_name -> dict[query_idx -> anchor mask], anchors on query mask
        self._scores = {}                # roi_name -> dict[query_idx -> anchor score], anchor score on query mask

    def mode_label(self) -> str:
        """Which of align/soft_pool is on, for a status line."""
        on = [n for n, f in (("align", self.align), ("core-pool", self.soft_pool)) if f]
        return "+".join(on) if on else "base"

    # ALIGNEMENT
    def _support_geometry(self) -> dict:
        """dict[slice_idx -> (cy,cx,scale)] for support slices carrying a mask.
            Return: dict with slice idx as index and (cy, cx) centroid coordinates 
            and scale (area in pixel) for selected ROI
        """
        if self._supp_geo is None:
            idxs = {i for masks in self.support_masks.values() for i in masks}
            self._supp_geo = build_body_geometry(self.support_slices, self.body_thresh,
                                                 self.body_min_px, idxs=idxs)
        return self._supp_geo

    def _query_geometry(self, q_idx: int) -> tuple:
        """(cy, cx, scale) of the body on query slice q_idx, or None."""
        if q_idx not in self._query_geo:
            self._query_geo[q_idx] = body_geometry(
                body_mask_2d(self.query_slices[q_idx], self.body_thresh, self.body_min_px))
        return self._query_geo[q_idx]

    def scale_ratio(self, roi_name: str = None) -> float:
        """
        Query body size / support body size (relative units, one number for the whole
        volume pair on purpose): the body tapers along z, and that taper is the cue the
        matching lives on, so rescaling each slice by its own body size would erase the
        signal being measured.
        """
        supp_shape = self._support_shape()
        supp = self._support_geometry()

        def q_size(i):
            g = self._query_geometry(i)
            return None if g is None else normalized_geometry(g, self.query_slices[i].shape)[2]

        pairs = self._pairs.get(roi_name) if roi_name else None
        if pairs:
            ratios = [q_size(q) / normalized_geometry(supp[s], supp_shape)[2]
                      for q, s in pairs if s in supp and q_size(q) is not None]
            if ratios:
                return float(np.median(ratios))
        if self._scale is None:
            q_sizes = [v for v in (q_size(i) for i in self.query_slices) if v is not None]
            s_sizes = [normalized_geometry(g, supp_shape)[2] for g in supp.values()]
            self._scale = (float(np.median(q_sizes) / np.median(s_sizes))
                           if q_sizes and s_sizes else 1.0)
        return self._scale

    def _src_centroid(self, roi_name: str) -> tuple:
        """(cy, cx), the frame the roi's pooling region is drawn in, or None."""
        if roi_name not in self._roi_centroid:
            self._roi_centroid[roi_name] = roi_reference_centroid(
                self._support_geometry(), self.support_masks.get(roi_name, {}))
        return self._roi_centroid[roi_name]

    def _support_shape(self) -> tuple:
        """(H, W) of the support slices, the frame every pooling region is drawn in."""
        return next(iter(self.support_slices.values())).shape

    def region_for(self, roi_name: str, q_idx: int) -> np.ndarray:
        """
        The roi's pooling region corrected for where the body sits on query slice q_idx
        (raw region when alignment is off or no body was found). Both slices are already
        resized to the encoder's input, so most of a pixel offset between the two volumes
        is field of view, already gone -- only the residual (body sitting higher/lower in
        its own frame) needs correcting. With no residual this returns exactly the base
        region, so alignment can only correct, never displace.
        """
        region = self.regions.get(roi_name)
        if region is None or not self.align:
            return region
        src = self._src_centroid(roi_name) #compute ROI median centroid
        dst = self._query_geometry(q_idx)
        if src is None or dst is None:
            return region
        H, W = self._support_shape()
        dy_n, dx_n, _ = normalized_geometry(dst, self.query_slices[q_idx].shape)
        shifted = (src[0] + (dy_n - src[0] / H) * H, src[1] + (dx_n - src[1] / W) * W)
        warped = align_region(region, src, shifted, scale=self.scale_ratio(roi_name),
                              out_shape=(H, W))
        return warped if warped is not None and warped.any() else region

    # SIMILARITY
    def _bags(self) -> dict:
        if self._slice_bags is None:
            self._slice_bags = build_slice_bags(self.seg, self.support_slices,
                                                self.support_masks, pool_region=self.regions,
                                                soft=self.soft_pool)
        return self._slice_bags

    def support_range(self, roi_name: str) -> tuple:
        """(lo, hi) support slices the roi's mask spans."""
        idxs = sorted(self.support_masks.get(roi_name, {}))
        return (idxs[0], idxs[-1]) if idxs else None

    def search_range(self, roi_name: str, q_idx: int) -> tuple:
        """(lo, hi) support slices to search for q_idx, centred on the z map's prediction;
        None (search all) until a pair exists or radius <= 0."""
        if self.search_radius <= 0 or not self._pairs.get(roi_name):
            return None
        a, b = fit_z_map(self._pairs[roi_name]) #estimate slopes and intercept
        centre = int(round(a * q_idx + b)) #estimate centre using a and b (linear regression)
        return centre - self.search_radius, centre + self.search_radius

    def suggest_support(self, roi_name: str, q_idx: int, top_k: int = 3,
                        constrain: bool = True) -> list[MatchCandidate]:
        """Best-first support candidates for one ROI and query slice.

        At most ``top_k`` candidates are returned. When ``constrain`` is true, an existing
        z map limits the support range; an empty constrained result falls back to all slices.
        """
        idx_range = self.search_range(roi_name, q_idx) if constrain else None #lower and upper end
        region = self.region_for(roi_name, q_idx)
        ranked = query_support_slice_similarity(
            self.seg, self.query_slices[q_idx], self._bags(), roi_name,
            region_mask=region, idx_range=idx_range, soft=self.soft_pool)
        if not ranked and idx_range is not None:   # prediction fell outside the roi
            ranked = query_support_slice_similarity(
                self.seg, self.query_slices[q_idx], self._bags(), roi_name,
                region_mask=region, soft=self.soft_pool)
        return [
            MatchCandidate(
                roi_name=roi_name,
                query_index=q_idx,
                support_index=s_idx,
                similarity=float(similarity),
            )
            for s_idx, similarity in ranked[:top_k]
        ]

    # ANCHORS
    def _bags_for_support_slice(self, s_idx: int) -> dict:
        """Multiclass bags built from support slice s_idx alone, every roi present on it
        kept as a rival class (plus background), so scoring stays winner-take-all."""
        if s_idx not in self._bag_cache:
            sub = {roi: {s_idx: masks[s_idx]}
                   for roi, masks in self.support_masks.items() if s_idx in masks}
            self._bag_cache[s_idx] = build_multiclass_bags(
                self.seg, self.support_slices, sub, thr_hi=self.thr_hi, thr_lo=self.thr_lo,
                body_thresh=self.body_thresh, body_min_px=self.body_min_px)
        return self._bag_cache[s_idx]

    def anchor_from_pair(self, roi_name: str, q_idx: int, s_idx: int) -> tuple:
        """(score, mask) for roi_name on the query slice, or None. Matching, not a copy of
        the support mask: the support slice's bag scores the query cell by cell, same
        machinery as api.find_prompts, one support slice wide."""
        bags = self._bags_for_support_slice(s_idx)
        if roi_name not in bags:
            return None
        frame_u8 = self.query_slices[q_idx]
        body = body_mask_2d(frame_u8, self.body_thresh, self.body_min_px)
        feat = self.seg.encoder_f_extractor(frame_u8)
        C, h, w = feat.shape
        pos = _positional_channels(_to_grid(body, h, w, feat.device))
        feat = torch.cat([feat, pos.to(feat.dtype)], dim=0)
        masks = multiclass_masks(multiclass_score_maps(feat, bags), body, frame_u8.shape,
                                 self.score_thresh, score_mode=self.score_mode)
        del feat
        return masks.get(roi_name)

    def _store_anchor(self, roi_name: str, q_idx: int, s_idx: int,
                      score: float, mask: np.ndarray) -> None:
        """Commit a successfully computed pair and anchor to the session state."""
        self._pairs.setdefault(roi_name, []).append((q_idx, s_idx))
        self._prompts.setdefault(roi_name, {})[q_idx] = mask
        self._scores.setdefault(roi_name, {})[q_idx] = score

    def accept_candidate(self, candidate: MatchCandidate) -> AcceptedMatch | None:
        found = self.anchor_from_pair(
            candidate.roi_name,
            candidate.query_index,
            candidate.support_index,
        )
        if found is None:
            return None

        score, mask = found
        self._store_anchor(
            candidate.roi_name,
            candidate.query_index,
            candidate.support_index,
            score,
            mask,
        )
        return AcceptedMatch(candidate=candidate, score=score, mask=mask)

    def add_pair(self, roi_name: str, q_idx: int, s_idx: int) -> np.ndarray | None:
        """Compute and atomically record a pair, returning its anchor mask or None.

        Prefer ``accept_candidate`` when the pair comes from ``suggest_support``.
        """
        found = self.anchor_from_pair(roi_name, q_idx, s_idx)
        if found is None:
            return None
        score, mask = found
        self._store_anchor(roi_name, q_idx, s_idx, score, mask)
        return mask

    def drop_pair(self, roi_name: str, q_idx: int) -> None:
        """Undo one step: forget the pair anchored at query slice q_idx."""
        self._pairs[roi_name] = [p for p in self._pairs.get(roi_name, []) if p[0] != q_idx]
        self._prompts.get(roi_name, {}).pop(q_idx, None)
        self._scores.get(roi_name, {}).pop(q_idx, None)

    def matches_for(self, roi_name: str) -> tuple[tuple[int, int], ...]:
        """Confirmed ``(query_index, support_index)`` pairs for one ROI."""
        return tuple(self._pairs.get(roi_name, ()))

    def anchor_indices(self, roi_name: str) -> tuple[int, ...]:
        """Sorted query indices carrying an anchor for one ROI."""
        return tuple(sorted(self._prompts.get(roi_name, ())))

    def has_anchors(self, roi_name: str | None = None) -> bool:
        """Whether the session, or one ROI, contains at least one anchor."""
        if roi_name is not None:
            return bool(self._prompts.get(roi_name))
        return any(self._prompts.values())

    # WHERE TO GO NEXT
    def z_map(self, roi_name: str) -> tuple:
        """(a, b) with support_idx = a*query_idx + b, or None with no pair."""
        pairs = self._pairs.get(roi_name)
        return fit_z_map(pairs) if pairs else None

    def query_window(self, roi_name: str) -> tuple:
        """(lo, hi) query slices the roi is predicted to span (its known support extent
        pushed through the inverse z map), or None."""
        zmap, s_range = self.z_map(roi_name), self.support_range(roi_name)
        if zmap is None or s_range is None:
            return None
        return query_window_from_z_map(*zmap, *s_range, self.q_bounds)

    def next_query_slice(self, roi_name: str) -> int:
        """Query slice to review next, or None when the window is used up. Farthest-point
        pick inside the predicted window: max distance to the slices already anchored,
        min_gap apart, so anchors spread over the roi instead of clustering."""
        window = self.query_window(roi_name)
        if window is None:
            return None
        lo, hi = window
        used = sorted(self._prompts.get(roi_name, {}))
        free = [i for i in sorted(self.query_slices) if lo <= i <= hi
                and all(abs(i - u) >= self.min_gap for u in used)]
        if not free:
            return None
        if not used:
            return min(free, key=lambda i: abs(i - (lo + hi) / 2.0))
        return max(free, key=lambda i: min(abs(i - u) for u in used))

    # HAND OVER TO PROPAGATE
    def anchors(self) -> dict:
        """dict[roi_name -> dict[slice_idx -> mask]], api.propagate's prompts."""
        return {roi: dict(p) for roi, p in self._prompts.items() if p}

    def z_bounds(self) -> dict:
        """dict[roi_name -> (lo, hi)], api.propagate's z_bounds: predicted window, widened
        to cover every anchor."""
        out = {}
        for roi, prompts in self._prompts.items():
            if not prompts:
                continue
            window = self.query_window(roi) or (min(prompts), max(prompts))
            out[roi] = (min(window[0], min(prompts)), max(window[1], max(prompts)))
        return out


def collect_anchors(seg, support_slices: dict, support_masks: dict, query_slices: dict,
                    roi_name: str, start_query_idx: int, n_steps: int = 3,
                    **session_kwargs) -> SliceMatchSession:
    """Unattended version of the loop: always takes the top-1 support match. For scripted
    runs and testing; the GUI drives SliceMatchSession itself so a person confirms each
    match. Returns the session (anchors()/z_bounds() ready for api.propagate)."""
    ses = SliceMatchSession(seg, support_slices, support_masks, query_slices, **session_kwargs)
    q_idx = start_query_idx
    for _ in range(n_steps):
        if q_idx is None:
            break
        ranked = ses.suggest_support(roi_name, q_idx, top_k=1)
        if not ranked:
            break
        candidate = ranked[0]
        if ses.accept_candidate(candidate) is None:
            break
        q_idx = ses.next_query_slice(roi_name)
    return ses
