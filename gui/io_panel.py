import os

import numpy as np
from qtpy.QtWidgets import QFileDialog, QMessageBox

from gui.config import DEMO_QUERY, DEMO_SUPPORT, REPO
from gui.slice_pane import load_case
from dafne_sam2.preprocessing import labels_to_masks, masks_to_labels


class IOPanelMixin:
    """Loading support/query cases, GT lookup, and export."""

    def _apply_support(self, path: str):
        self.support_path = path
        self.support_vol, self.support_lbl, self.roi_real = load_case(path)
        self.roi_names = [f"roi_{i}" for i in range(1, len(self.roi_real) + 1)]
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems(self.roi_names)
        self.roi_combo.blockSignals(False)
        self.support_pane.set_volume(self.support_vol, self.support_lbl, len(self.roi_names))
        self._refresh_legend()
        self.windows = {r: w for r, w in self.windows.items() if r in self.roi_names}
        self.prompt_kind = {r: k for r, k in self.prompt_kind.items() if r in self.roi_names}
        self.suggested.clear()  # support changed: any earlier suggestion is stale
        self.session = None     # its bags describe the old support volume
        self.match_combo.clear()
        self._maybe_estimate_windows()
        self._refresh_extent_label()

    def _apply_query(self, path: str):
        self.query_path = path
        # GT is kept but never shown or fed to matching: it is read only after a run,
        # to score it (see metrics.evaluate / _on_segment). query_roi is the file's own
        # roi name order, needed to tell whether its GT lines up with the support's rois.
        self.query_vol, self.query_lbl, self.query_roi = load_case(path)
        self.windows.clear()
        self.suggested.clear()
        self.result.clear()
        self.anchors.clear()
        self.session = None
        self.match_combo.clear()
        self.query_pane.set_volume(self.query_vol, None, len(self.roi_names))
        for w in (self.lo_spin, self.hi_spin):
            w.blockSignals(True)
            w.setEnabled(True)
            w.setMinimum(0)
            w.setMaximum(self.query_vol.shape[0] - 1)
            w.blockSignals(False)
        self._maybe_estimate_windows()
        self._refresh_extent_label()

    def _query_gt_masks(self):
        """Return: the query file's own GT as dict[roi_N -> dict[z -> mask]], or None.
        None unless the query's roi names match the support's exactly -- roi_N is
        positional, so mismatched files would score roi_3 against a different roi_3."""
        if self.query_lbl is None or self.query_roi != self.roi_real:
            return None
        return labels_to_masks(self.query_lbl, self.roi_names)

    def _load_demo(self):
        missing = [p for p in (DEMO_SUPPORT, DEMO_QUERY) if not os.path.isfile(p)]
        if missing:
            QMessageBox.warning(self, "Demo data missing", "Not found:\n" + "\n".join(missing))
            return
        self._apply_support(DEMO_SUPPORT)
        self._apply_query(DEMO_QUERY)
        self.status.setText("Demo loaded. Pick an ROI, review the suggested extent, confirm.")
        self._update_enabled()

    def _load_support(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load support .npz", REPO, "NumPy (*.npz)")
        if not path:
            return
        self._apply_support(path)
        self.status.setText(f"Support: {os.path.basename(path)}  ·  "
                            f"{len(self.roi_names)} ROI(s)")
        self._update_enabled()

    def _load_query(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load query .npz", REPO, "NumPy (*.npz)")
        if not path:
            return
        self._apply_query(path)
        self.status.setText(f"Query: {os.path.basename(path)}  ·  "
                            f"{self.query_vol.shape[0]} slices")
        self._update_enabled()

    def _on_export(self):
        if not self.result:
            QMessageBox.warning(self, "Nothing to export", "Run 'Segment confirmed ROIs' first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export segmentation", "segmentation.npy",
                                              "NumPy (*.npy)")
        if not path:
            return
        np.save(path, masks_to_labels(self.result, self.query_vol.shape, self.roi_names))
        self.status.setText(f"Saved {os.path.basename(path)}")
