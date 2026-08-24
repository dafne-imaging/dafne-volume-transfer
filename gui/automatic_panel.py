import os

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QMessageBox

from gui.config import _CKPT_NAME
from gui.worker import run_in_thread
from dafne_sam2 import metrics
from dafne_sam2.automatic import debugging
from dafne_sam2.automatic.api import find_prompts, propagate
from dafne_sam2.automatic.backbone import SAM2Segmenter
from dafne_sam2.automatic.checkpoints import CHECKPOINT_MODELS, download_checkpoint, resolve_checkpoint
from dafne_sam2.preprocessing import labels_to_masks, volume_to_slices, volume_to_uint8


class AutomaticPanelMixin:
    """The 'Extent' route's run: confirmed window per roi -> find_prompts + propagate."""

    def _get_seg(self) -> SAM2Segmenter:
        if self.seg is None:
            ckpt_path = os.environ["DAFNE_SAM2_CHECKPOINT"]
            if not os.path.isfile(ckpt_path):
                # download_checkpoint saves under CHECKPOINT_MODELS[name]['file_name'], so
                # pick the entry whose filename matches what DAFNE_SAM2_CHECKPOINT
                # points at, in case a custom checkpoint path is configured
                ckpt_name = next((n for n, d in CHECKPOINT_MODELS.items()
                                  if d["file_name"] == os.path.basename(ckpt_path)), _CKPT_NAME)
                download_checkpoint(ckpt_name, os.path.dirname(ckpt_path))
            checkpoint, model_cfg = resolve_checkpoint(None, None)
            # 'auto': CUDA when the machine has a usable one, cpu otherwise. Overridable
            # with DAFNE_SAM2_DEVICE (e.g. 'cpu' to force, 'mps' to try Metal,
            # 'cuda:1' to pick a card) -- see backbone.pick_device.
            device = os.environ.get("DAFNE_SAM2_DEVICE", "auto")
            self.seg = SAM2Segmenter(checkpoint, model_cfg, device=device)
            print(f"[dafne-sam2] running on device: {self.seg.device}", flush=True)
        return self.seg

    def _release_seg(self):
        """Unload the model and free the GPU memory it holds, weights included. Called at
        the end of every Segment run and on close: nothing outside a run needs the card,
        and the next _get_seg() rebuilds the segmenter from the checkpoint."""
        if self.seg is None:
            return
        self.seg.release()
        self.seg = None

    def _on_segment(self):
        if self.support_vol is None or self.query_vol is None:
            QMessageBox.warning(self, "Missing data", "Load support and query first.")
            return
        if not self.windows:
            QMessageBox.warning(self, "No extent", "Confirm at least one ROI's extent first.")
            return

        support_masks = labels_to_masks(self.support_lbl, self.roi_names)
        windows = {r: w for r, w in self.windows.items() if r in support_masks}
        if not windows:
            QMessageBox.warning(self, "No extent", "The confirmed ROI(s) are empty on the support labels.")
            return

        self.segment_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        debug_dir = os.environ.get("DAFNE_SAM2_DEBUG_DIR")
        debug_sink = {} if debug_dir else None
        # printed every run: an env var that silently did not reach the process (set on
        # its own shell line instead of the python line, so never exported) otherwise
        # looks exactly like "the dump feature is broken"
        print(f"[debug] dir={os.path.abspath(debug_dir) if debug_dir else 'off'}", flush=True)
        self.status.setText("Matching support and query slices…")

        def work(report):
            support_slices = volume_to_slices(volume_to_uint8(self.support_vol))
            query_slices = volume_to_slices(volume_to_uint8(self.query_vol))
            # unwindowed rois are not segmented, but still act as rival classes when scoring
            query_gt = self._query_gt_masks()
            anchors, bounds = find_prompts(support_slices, query_slices, support_masks,
                                           windows, return_windows=True, seg=self._get_seg(),
                                           debug_sink=debug_sink, query_masks=query_gt)
            # bounds stop the tracker once the organ ends, see api.propagate
            report("Propagating anchors…")
            result = propagate(query_slices, anchors, z_bounds=bounds,
                               seg=self._get_seg(), prompt_kind=self.prompt_kind,
                               resolve_overlaps=self.overlap_check.isChecked(),
                               joint_propagate=self.joint_check.isChecked(),
                               fill_gaps=self.fillgaps_check.isChecked(),
                               progress_callback=lambda done, total:
                                   report(f"Propagating anchors… ROI {done}/{total}"))
            scores = None
            if query_gt is not None:
                scores = metrics.evaluate(result, query_gt, windows=bounds)
                print("[eval]\n" + metrics.format_report(scores), flush=True)
            dump_path = None
            if debug_dir:
                dump_path = debugging.dump_run(debug_dir, self.support_path or "support",
                                               self.query_path or "query",
                                               debug_sink, scores=scores)
                print(f"[debug] wrote {dump_path}", flush=True)
            return anchors, result, scores, dump_path

        def on_finished(payload):
            self._release_seg()
            QApplication.restoreOverrideCursor()
            self.anchors, self.result, scores, dump_path = payload
            picked = ", ".join(f"{r}: {sorted(a)}" for r, a in sorted(self.anchors.items()))
            dumps = "  dump written" if dump_path else ""
            gt = f"  dice={scores['_mean_dice']:.3f}" if scores else ""
            self.status.setText(f"Done.{dumps}{gt}  Anchors — {picked}")
            self._refresh_query_labels()
            self._update_enabled()

        def on_error(msg):
            self._release_seg()
            QApplication.restoreOverrideCursor()
            self.status.setText("Segmentation failed.")
            self._update_enabled()
            QMessageBox.critical(self, "Segmentation failed", msg)

        self._bg_thread, self._bg_worker = run_in_thread(
            self, work, self.status.setText, on_finished, on_error)
