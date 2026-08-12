import gc
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2_video_predictor_npz
from sam2_support_match.automatic.device_utils import pick_device, empty_cache
from sam2_support_match.preprocessing import resize_grayscale_to_rgb_and_resize, IMG_SIZE

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

FUSE_TARGET_LEVEL = 0  # stride4, 128x128: max resolution


def mask_to_box(mask: np.ndarray, margin: int = 0) -> tuple | None:
    """[H,W] bool -> tight (x0,y0,x1,y1) + margin px, or None if mask is empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    H, W = mask.shape
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + 1 + margin
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + 1 + margin
    return (max(0, x0), max(0, y0), min(W, x1), min(H, y1))


class MedSAM2Segmenter:
    def __init__(self, checkpoint: str, model_cfg: str, device: str='auto'):
        """Load the SAM2 video predictor from checkpoint/config. device: see pick_device."""
        self.checkpoint = checkpoint
        self.device = pick_device(device)
        # 'cuda:1' is a valid place to put tensors but not a valid autocast argument,
        # which takes a device TYPE; keep both rather than dropping the index
        self.device_type = self.device.split(':')[0]
        self.predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint, device=self.device)
        self.model_cfg = model_cfg

    def free_cache(self) -> None:
        """Return this segmenter's unused allocator blocks to the driver (empty_cache).
        Results are unaffected; the loaded model stays resident. See release()."""
        empty_cache(self.device_type)

    def release(self) -> None:
        """Drops the loaded model and frees its GPU memory. Segmenter is unusable after
        (build a new one) -- for shutdown, not between runs (reload costs a full
        checkpoint read)."""
        self.predictor = None
        gc.collect()
        self.free_cache()

    def _release_state(self, state) -> None:
        """Drop one propagation session: its resized frames and its per-frame memory bank
        are the bulk of a run's GPU use, and holding them into the next roi's init_state
        doubles the peak for no reason. Cleared in place, so a caller's reference to the
        same dict does not keep the tensors alive either."""
        if state is None:
            return
        reset = getattr(self.predictor, "reset_state", None)
        if reset is not None:
            try:
                reset(state)
            except Exception:  # a partially-built state must still be droppable
                pass
        if isinstance(state, dict):
            state.clear()

    def _autocast(self):
        """bfloat16 on CUDA, disabled elsewhere (cpu bfloat16 is slower than fp32 without
        AMX; mps autocast unvalidated here)."""
        if self.device_type == 'cuda':
            return torch.autocast('cuda', dtype=torch.bfloat16)
        return torch.autocast(self.device_type, enabled=False)

    def _preprocess(self, vol_u8: np.ndarray) -> torch.Tensor:
        '''[Z,H,W] uint8 -> [Z,3,IMG_SIZE,IMG_SIZE] tensor, ImageNet-normalized.'''
        img_processed = resize_grayscale_to_rgb_and_resize(vol_u8, IMG_SIZE)
        t = torch.from_numpy(img_processed).to(self.device).float() / 255.0  # ImageNet stats are on [0,1]
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device)[:, None, None]
        std = torch.tensor(_IMAGENET_STD, device=self.device)[:, None, None]
        return (t - mean) / std

    @staticmethod
    def _fuse_fpn_levels(backbone_fpn: list, target_level: int = 0) -> torch.Tensor:
        """
        Resamples SAM2's FPN levels ([0]=stride4 finest .. [-1]=stride16 most semantic)
        onto target_level's grid and concatenates them: [B, C*n_levels, h_t, w_t]. One
        stride16 cell is ~8-9 original px here, too coarse to keep a thin muscle or an
        inter-organ boundary from falling inside a single cell. Per-level L2-normalize
        stops one level's raw activation magnitude from dominating the cosine similarity
        once concatenated (channels already match, d_model=256 uniform).
        """
        target_h, target_w = backbone_fpn[target_level].shape[-2:]
        levels = []
        for lvl in backbone_fpn:
            lvl = lvl.float()
            if lvl.shape[-2:] != (target_h, target_w):
                lvl = F.interpolate(lvl, size=(target_h, target_w), mode='bilinear', align_corners=False)
            levels.append(F.normalize(lvl, dim=1))
        return torch.cat(levels, dim=1)

    @torch.inference_mode()
    def encoder_f_extractor(self, frame_u8: np.ndarray) -> torch.Tensor:
        '''[H,W] uint8 -> [C,h,w] fused-FPN feature (see _fuse_fpn_levels).'''
        frame_u8_processed = self._preprocess(frame_u8[None])
        with self._autocast():
            out = self.predictor.forward_image(frame_u8_processed)
        return self._fuse_fpn_levels(out["backbone_fpn"], target_level=FUSE_TARGET_LEVEL)[0]

    @torch.inference_mode()
    def encoder_frames_iter(self, frames_u8: list, chunk_size: int = 8):
        '''Yields one [C,h,w] feature at a time, chunk_size frames per forward. Generator,
        not a list: one fused feature is ~50 MB, a 60-slice window would cost 3 GB held at
        once when the consumer only needs one at a time.'''
        for start in range(0, len(frames_u8), chunk_size):
            chunk = np.stack(frames_u8[start:start+chunk_size], axis=0)
            img = self._preprocess(chunk)
            with self._autocast():
                out = self.predictor.forward_image(img)
            fpn = self._fuse_fpn_levels(out['backbone_fpn'], target_level=FUSE_TARGET_LEVEL)
            del out, img
            for i in range(fpn.shape[0]):
                yield fpn[i]
            del fpn

    def encoder_frames_batched(self, frames_u8: list, chunk_size: int = 8) -> list:
        '''Materializes what encoder_frames_iter streams -- GBs for a long list, prefer
        the iterator unless every feature really is needed at once.'''
        return list(self.encoder_frames_iter(frames_u8, chunk_size=chunk_size))

    @torch.inference_mode()
    def segment_volume_mask(self, vol_u8: np.ndarray,
                            masks: dict[int, np.ndarray],
                            return_logits: bool = False):
        """Mask-prompt propagation for ONE object. return_logits also returns the raw
        per-pixel logit volume, so a caller can arbitrate contested pixels by SAM2's own
        confidence (api._resolve_overlaps)."""
        Z, H, W = vol_u8.shape
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        logit_vol = np.full((Z, H, W), -1e4, dtype=np.float32) if return_logits else None
        if not masks:
            return (seg, logit_vol) if return_logits else seg

        img_resized = self._preprocess(vol_u8)
        state = None
        try:
            with self._autocast():
                state = self.predictor.init_state(img_resized, H, W)
                for fidx, mask in sorted(masks.items()):
                    self.predictor.add_new_mask(
                        inference_state=state, frame_idx=int(fidx), obj_id=1, mask=mask)

                for reverse in (False, True):
                    for fidx, _oids, logits in self.predictor.propagate_in_video(
                            state, reverse=reverse):
                        l = logits[0][0].float().cpu().numpy()
                        seg[fidx][l > 0.0] = 1
                        if return_logits:
                            logit_vol[fidx] = l
        finally:
            # session outlives this call otherwise, and api.propagate opens one per roi
            self._release_state(state)
            del img_resized
            self.free_cache()
        return (seg, logit_vol) if return_logits else seg

    @torch.inference_mode()
    def segment_volume_joint(self, vol_u8: np.ndarray,
                             prompts: dict,
                             kind_of: dict,
                             return_logits: bool = False):
        """
        Multi-object variant: every roi shares ONE session (own obj_id, same init_state),
        so SAM2's memory attention sees them all per frame and a tracker is discouraged
        from drifting onto territory another roi holds -- independent sessions
        (segment_volume_mask/box) have no such awareness.
        """
        Z, H, W = vol_u8.shape
        roi_names = list(prompts)
        seg = {r: np.zeros((Z, H, W), dtype=np.uint8) for r in roi_names}
        logit_vol = ({r: np.full((Z, H, W), -1e4, dtype=np.float32) for r in roi_names}
                    if return_logits else None)
        if not roi_names:
            return (seg, logit_vol) if return_logits else seg

        img_resized = self._preprocess(vol_u8)
        state = None
        try:
            with self._autocast():
                state = self.predictor.init_state(img_resized, H, W)
                # every object must be registered before propagation starts (SAM2 forbids
                # adding new objects once tracking has begun)
                for obj_id, roi_name in enumerate(roi_names, start=1):
                    for fidx, prompt in sorted(prompts[roi_name].items()):
                        if kind_of[roi_name] == 'box':
                            self.predictor.add_new_points_or_box(
                                inference_state=state, frame_idx=int(fidx), obj_id=obj_id,
                                box=np.asarray(prompt, dtype=np.float32))
                        else:
                            self.predictor.add_new_mask(
                                inference_state=state, frame_idx=int(fidx), obj_id=obj_id,
                                mask=prompt)

                for reverse in (False, True):
                    for fidx, oids, logits in self.predictor.propagate_in_video(
                            state, reverse=reverse):
                        for i, oid in enumerate(oids):
                            roi_name = roi_names[oid - 1]
                            l = logits[i][0].float().cpu().numpy()
                            seg[roi_name][fidx][l > 0.0] = 1
                            if return_logits:
                                logit_vol[roi_name][fidx] = l
        finally:
            self._release_state(state)
            del img_resized
            self.free_cache()
        return (seg, logit_vol) if return_logits else seg

    @torch.inference_mode()
    def segment_volume_box(self, vol_u8: np.ndarray,
                           boxes: dict[int, tuple],
                           return_logits: bool = False):
        """Box-prompt variant of segment_volume_mask: when the pseudo-label anchor mask is
        unreliable (small/paired structures, e.g. kidneys) a plain box is a more honest
        prompt than a pixel-precise but wrong-shaped mask."""
        Z, H, W = vol_u8.shape
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        logit_vol = np.full((Z, H, W), -1e4, dtype=np.float32) if return_logits else None
        if not boxes:
            return (seg, logit_vol) if return_logits else seg

        img_resized = self._preprocess(vol_u8)
        state = None
        try:
            with self._autocast():
                state = self.predictor.init_state(img_resized, H, W)
                for fidx, box in sorted(boxes.items()):
                    self.predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=int(fidx), obj_id=1,
                        box=np.asarray(box, dtype=np.float32))

                for reverse in (False, True):
                    for fidx, _oids, logits in self.predictor.propagate_in_video(
                            state, reverse=reverse):
                        l = logits[0][0].float().cpu().numpy()  # [H,W] raw logit
                        seg[fidx][l > 0.0] = 1
                        if return_logits:
                            logit_vol[fidx] = l
        finally:
            self._release_state(state)
            del img_resized
            self.free_cache()
        return (seg, logit_vol) if return_logits else seg
