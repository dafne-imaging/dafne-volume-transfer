from qtpy.QtWidgets import QMessageBox

from dafne_sam2.automatic.api import estimate_windows
from dafne_sam2.preprocessing import labels_to_masks, volume_to_slices, volume_to_uint8


class WindowPanelMixin:
    """The 'Extent' route: GT-free suggested query range per roi, confirm, clear."""

    def _maybe_estimate_windows(self):
        """Fill self.suggested with each roi's query slice range once both volumes are
        loaded. GT-free (support GT + both body silhouettes); a starting point to
        review, not a constraint. See api.estimate_windows."""
        if self.support_vol is None or self.query_vol is None:
            return
        support_masks = labels_to_masks(self.support_lbl, self.roi_names)
        support_slices = volume_to_slices(volume_to_uint8(self.support_vol))
        query_slices = volume_to_slices(volume_to_uint8(self.query_vol))
        try:
            self.suggested = estimate_windows(support_slices, support_masks, query_slices)
        except ValueError:
            self.suggested = {}  # e.g. an all-black volume: no body mask to anchor on
        self._on_roi_changed()

    def _on_kind_changed(self, index: int):
        roi = self.roi_combo.currentText()
        if not roi:
            return
        self.prompt_kind[roi] = "box" if index == 1 else "mask"

    def _on_suggest(self):
        roi = self.roi_combo.currentText()
        if not roi:
            return
        if roi not in self.suggested:
            QMessageBox.information(self, "No suggestion",
                                    "Load both support and query first (and make sure "
                                    f"{roi} has a mask on the support).")
            return
        lo, hi = self.suggested[roi]
        for w, v in ((self.lo_spin, lo), (self.hi_spin, hi)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.status.setText(f"{roi}: suggested [{lo}, {hi}] -- review, then confirm.")

    def _on_confirm_extent(self):
        roi = self.roi_combo.currentText()
        if self.query_vol is None or not roi:
            QMessageBox.warning(self, "Missing selection", "Load a query volume and pick an ROI first.")
            return
        lo, hi = self.lo_spin.value(), self.hi_spin.value()
        if lo > hi:
            lo, hi = hi, lo
        self.windows[roi] = (lo, hi)
        self._refresh_extent_label()
        self.status.setText(f"{roi} confirmed at [{lo}, {hi}].")
        self._update_enabled()

    def _on_clear_windows(self):
        self.windows.clear()
        self.result.clear()
        self.anchors.clear()
        self.query_pane.set_labels(None)
        self._refresh_extent_label()
        self._update_enabled()

    def _refresh_extent_label(self):
        if not self.windows:
            self.extent_label.setText("extents: none")
        else:
            self.extent_label.setText(
                "  ".join(f"{r}=[{lo},{hi}]" for r, (lo, hi) in sorted(self.windows.items())))
        self._on_roi_changed()
