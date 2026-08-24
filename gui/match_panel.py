from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QMessageBox

from gui.worker import run_in_thread
from dafne_sam2 import metrics
from dafne_sam2.automatic.api import propagate
from dafne_sam2.preprocessing import volume_to_slices, volume_to_uint8
from dafne_sam2.semi_automatic.api import (
    SliceMatchConfig,
    SliceMatchSession,
    create_session_from_volumes,
)


class MatchPanelMixin:
    """The 'Slice match' route: SliceMatchSession-driven anchor collection + its propagate."""

    def _get_session(self) -> SliceMatchSession:
        """The live SliceMatchSession, built on first use. Its segmenter is re-attached
        every call: a Segment/Propagate run releases the model, while the session's cached
        descriptors stay valid (same checkpoint, same device) and are worth keeping."""
        if self.session is None:
            self.session = create_session_from_volumes(
                self._get_seg(),
                self.support_vol,
                self.support_lbl,
                self.roi_names,
                self.query_vol,
                config=SliceMatchConfig(enhanced=True),
            )
        else:
            self.session.seg = self._get_seg()
        return self.session

    def _match_ready(self, roi: str) -> bool:
        if self.support_vol is None or self.query_vol is None or not roi:
            QMessageBox.warning(self, "Missing data", "Load support and query, then pick an ROI.")
            return False
        return True

    def _refresh_match_label(self):
        roi = self.roi_combo.currentText()
        pairs = self.session.matches_for(roi) if self.session else ()
        # the mode the live session was built with, not the checkbox: they differ until the
        # next Find rebuilds the descriptors
        mode = self.session.mode_label() if self.session else "no session"
        if not pairs:
            self.match_label.setText(f"pairs: none   [{mode}]")
        else:
            window = self.session.query_window(roi)
            shown = "  ".join(f"q{q}→s{s}" for q, s in sorted(pairs))
            a, b = self.session.z_map(roi)
            self.match_label.setText(
                f"{shown}\nz map: s = {a:.2f}·q {b:+.1f}   window: {window}   [{mode}]")
        self._update_enabled()

    def _on_find_match(self):
        roi = self.roi_combo.currentText()
        if not self._match_ready(roi):
            return
        q_idx = self.query_pane.z
        self._set_busy(True)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status.setText(f"Matching query slice {q_idx}…")

        def work(report):
            return self._get_session().suggest_support(roi, q_idx, top_k=3)

        def on_finished(ranked):
            QApplication.restoreOverrideCursor()
            self.match_combo.blockSignals(True)
            self.match_combo.clear()
            for candidate in ranked:
                self.match_combo.addItem(
                    f"support {candidate.support_index}   sim={candidate.similarity:.3f}",
                    candidate,
                )
            self.match_combo.blockSignals(False)
            if not ranked:
                self.status.setText(f"{roi}: no support slice carries this ROI.")
            else:
                best = ranked[0]
                self.support_pane.set_z(best.support_index)
                self.status.setText(
                    f"{best.roi_name}: query {best.query_index} → support {best.support_index} "
                    f"(sim {best.similarity:.3f}). Review, then accept."
                )
            self._set_busy(False)

        def on_error(msg):
            QApplication.restoreOverrideCursor()
            self._set_busy(False)
            QMessageBox.critical(self, "Matching failed", msg)

        self._bg_thread, self._bg_worker = run_in_thread(self, work, None, on_finished, on_error)

    def _on_match_pick(self, index: int):
        candidate = self.match_combo.itemData(index)
        if candidate is not None:
            self.support_pane.set_z(candidate.support_index)

    def _on_accept_match(self):
        candidate = self.match_combo.currentData()
        if candidate is None:
            QMessageBox.warning(self, "No candidate", "Run 'Find support match' first.")
            return
        roi = candidate.roi_name
        if not self._match_ready(roi):
            return
        q_idx = candidate.query_index
        s_idx = candidate.support_index
        ses = self._get_session()
        self._set_busy(True)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def work(report):
            return ses.accept_candidate(candidate)

        def on_finished(accepted):
            QApplication.restoreOverrideCursor()
            if accepted is None:
                self.status.setText(f"{roi}: nothing matched on query slice {q_idx}.")
                self._set_busy(False)
                self._refresh_match_label()
                return

            self.anchors = ses.anchors()
            self.view_combo.setCurrentIndex(1)  # anchors view, so the new mask is visible
            self._refresh_query_labels()
            self.query_pane.set_marks(set(ses.anchor_indices(roi)), "anchor")
            # the pane stays where it is: jumping on accept moves the slice out from under
            # the person still looking at what they just confirmed. The next slice is
            # suggested in the status bar instead, and they scroll there when ready.
            nxt = ses.next_query_slice(roi)
            tail = (f" Next suggested query slice: {nxt}." if nxt is not None
                    else " Window covered -- propagate when done.")
            self.status.setText(f"{roi}: anchored query {q_idx} from support {s_idx}.{tail}")
            self._set_busy(False)
            self._refresh_match_label()

        def on_error(msg):
            QApplication.restoreOverrideCursor()
            self._set_busy(False)
            QMessageBox.critical(self, "Anchor failed", msg)

        self._bg_thread, self._bg_worker = run_in_thread(self, work, None, on_finished, on_error)

    def _on_undo_match(self):
        roi = self.roi_combo.currentText()
        pairs = self.session.matches_for(roi) if self.session else ()
        if not pairs:
            return
        q_idx = pairs[-1][0]
        self.session.drop_pair(roi, q_idx)
        self.anchors = self.session.anchors()
        self._refresh_query_labels()
        self.query_pane.set_marks(set(self.session.anchor_indices(roi)), "anchor")
        self.status.setText(f"{roi}: dropped the anchor on query slice {q_idx}.")
        self._refresh_match_label()

    def _on_reset_session(self):
        """Forget every pair and anchor. Encoded support descriptors go with them, so the
        next Find pays for them again."""
        self.session = None
        self.anchors.clear()
        self.match_combo.clear()
        self.query_pane.set_marks(set(), "anchor")
        self._refresh_query_labels()
        self.status.setText("Matching session reset.")
        self._refresh_match_label()

    def _on_propagate_anchors(self):
        """SAM2 from the anchors the person collected, bounded by the z map's predicted
        query window (no confirmed extent involved -- see api.propagate's z_bounds)."""
        if not (self.session and self.session.anchors()):
            QMessageBox.warning(self, "No anchors", "Accept at least one match first.")
            return
        anchors, bounds = self.session.anchors(), self.session.z_bounds()
        self.anchors = anchors
        self._set_busy(True)
        self.status.setText("Propagating from anchors…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # read on the GUI thread, not inside work(): a background thread must never touch
        # a Qt widget, even just to read it
        resolve_overlaps = self.overlap_check.isChecked()
        joint_propagate = self.joint_check.isChecked()
        fill_gaps = self.fillgaps_check.isChecked()

        def work(report):
            query_slices = volume_to_slices(volume_to_uint8(self.query_vol))
            result = propagate(query_slices, anchors, z_bounds=bounds,
                               seg=self._get_seg(), prompt_kind=self.prompt_kind,
                               resolve_overlaps=resolve_overlaps,
                               joint_propagate=joint_propagate,
                               fill_gaps=fill_gaps,
                               refine_mask_prompt=False,
                               progress_callback=lambda done, total:
                                   report(f"Propagating from anchors… ROI {done}/{total}"))
            scores = None
            query_gt = self._query_gt_masks()
            if query_gt is not None:
                scores = metrics.evaluate(result, query_gt, windows=bounds)
                print("[eval]\n" + metrics.format_report(scores), flush=True)
            return result, scores

        def on_finished(payload):
            self._release_seg()
            QApplication.restoreOverrideCursor()
            self.result, scores = payload
            gt = f"  dice={scores['_mean_dice']:.3f}" if scores else ""
            picked = ", ".join(f"{r}: {sorted(a)}" for r, a in sorted(anchors.items()))
            self.status.setText(f"Done from anchors.{gt}  {picked}")
            self.view_combo.setCurrentIndex(0)
            self._refresh_query_labels()
            self._set_busy(False)

        def on_error(msg):
            self._release_seg()
            QApplication.restoreOverrideCursor()
            self.status.setText("Propagation failed.")
            self._set_busy(False)
            QMessageBox.critical(self, "Propagation failed", msg)

        self._bg_thread, self._bg_worker = run_in_thread(
            self, work, self.status.setText, on_finished, on_error)
