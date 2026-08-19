import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from gui.automatic_panel import AutomaticPanelMixin
from gui.io_panel import IOPanelMixin
from gui.match_panel import MatchPanelMixin
from gui.slice_pane import SlicePane
from gui.style import ROI_COLORS
from gui.window_panel import WindowPanelMixin
from dafne_sam2.automatic.backbone import SAM2Segmenter, mask_to_box
from dafne_sam2.preprocessing import masks_to_labels
from dafne_sam2.semi_automatic.slice_api import SliceMatchSession


class MainWindow(QMainWindow, IOPanelMixin, WindowPanelMixin, AutomaticPanelMixin, MatchPanelMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dafne-sam2")

        self.support_vol = self.support_lbl = None
        self.query_vol = self.query_lbl = None
        self.support_path = self.query_path = None
        self.roi_names: list[str] = []
        self.roi_real: list[str] = []          # names as stored in the npz, for the legend
        self.query_roi: list[str] = []         # the query file's own roi names, for GT scoring
        self.suggested: dict[str, tuple[int, int]] = {}  # GT-free auto-suggestion, per roi
        self.windows: dict[str, tuple[int, int]] = {}     # user-confirmed (lo, hi), per roi
        self.prompt_kind: dict[str, str] = {}             # 'mask' (default) or 'box', per roi
        self.result: dict[str, dict[int, np.ndarray]] = {}
        self.anchors: dict[str, dict[int, np.ndarray]] = {}
        self.session: SliceMatchSession | None = None  # semi-auto slice matching, see slice_api
        self.seg: SAM2Segmenter | None = None

        self.support_pane = SlicePane("Support", "reference masks")
        self.query_pane = SlicePane("Query", "review/confirm the suggested extent")

        # splitter, not a fixed 50/50: on a narrow window one pane can be given the room
        # instead of both shrinking below what the image needs
        panes = QSplitter(Qt.Horizontal)
        panes.addWidget(self.support_pane)
        panes.addWidget(self.query_pane)
        panes.setSizes([600, 600])
        panes.setChildrenCollapsible(False)

        main = QHBoxLayout()
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(10)
        main.addWidget(self._build_sidebar())
        main.addWidget(panes, stretch=1)

        central = QWidget()
        central.setLayout(main)
        self.setCentralWidget(central)
        self.setMinimumSize(880, 560)  # below this the sidebar scrolls, panes stay usable
        self.resize(1360, 820)
        self._update_enabled()

    # -- sidebar -------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        demo_btn = QPushButton("Load demo case  (CHAOS → AMOS)")
        demo_btn.clicked.connect(self._load_demo)
        support_btn = QPushButton("Load support .npz…")
        support_btn.clicked.connect(self._load_support)
        query_btn = QPushButton("Load query .npz…")
        query_btn.clicked.connect(self._load_query)

        data_box = QGroupBox("Data")
        data_lay = QVBoxLayout()
        data_lay.setSpacing(6)
        for w in (demo_btn, support_btn, query_btn):
            data_lay.addWidget(w)
        data_box.setLayout(data_lay)

        # the roi selector drives both routes, so it sits above the tabs, not inside one
        self.roi_combo = QComboBox()
        self.roi_combo.currentTextChanged.connect(lambda _t: self._on_roi_changed())
        roi_box = QGroupBox("ROI")
        roi_lay = QVBoxLayout()
        roi_lay.setSpacing(6)
        roi_lay.addWidget(self.roi_combo)
        roi_box.setLayout(roi_lay)

        self.lo_spin = QSpinBox()
        self.lo_spin.setEnabled(False)
        self.lo_spin.setKeyboardTracking(False)
        self.hi_spin = QSpinBox()
        self.hi_spin.setEnabled(False)
        self.hi_spin.setKeyboardTracking(False)
        range_row = QHBoxLayout()
        range_row.setSpacing(6)
        range_row.addWidget(QLabel("from"))
        range_row.addWidget(self.lo_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.hi_spin)

        self.suggest_btn = QPushButton("Suggest extent (GT-free)")
        self.suggest_btn.setToolTip(
            "Body-extent position transfer from the support -- a starting guess, "
            "review before confirming (unreliable on small/paired structures).")
        self.suggest_btn.clicked.connect(self._on_suggest)
        self.confirm_btn = QPushButton("Confirm extent for ROI")
        self.confirm_btn.clicked.connect(self._on_confirm_extent)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["Anchor: pseudolabel mask", "Anchor: bbox"])
        self.kind_combo.setToolTip(
            "Per-ROI anchor prompt fed to SAM2. bbox trades pixel precision for a more "
            "honest prompt when the pseudolabel mask itself is unreliable (small/paired "
            "structures, e.g. kidneys).")
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        self.clear_btn = QPushButton("Clear extents")
        self.clear_btn.clicked.connect(self._on_clear_windows)
        self.extent_label = QLabel("extents: none")
        self.extent_label.setObjectName("seeds")
        self.extent_label.setWordWrap(True)

        self.segment_btn = QPushButton("Segment confirmed ROIs")
        self.segment_btn.setObjectName("primary")
        self.segment_btn.clicked.connect(self._on_segment)

        seed_tab = QWidget()
        seed_lay = QVBoxLayout()
        seed_lay.setContentsMargins(8, 10, 8, 8)
        seed_lay.setSpacing(6)
        seed_lay.addLayout(range_row)
        for w in (self.suggest_btn, self.confirm_btn, self.kind_combo, self.clear_btn,
                  self.segment_btn, self.extent_label):
            seed_lay.addWidget(w)
        seed_lay.addStretch()
        seed_tab.setLayout(seed_lay)

        self.find_btn = QPushButton("Find support match for this slice")
        self.find_btn.setToolTip(
            "Stand on a query slice, get the support slices whose ROI region looks most "
            "like it (whole-slice descriptors, slice_api.SliceMatchSession). Once a pair "
            "exists the search is restricted around the z map's prediction.")
        self.find_btn.clicked.connect(self._on_find_match)
        self.match_combo = QComboBox()
        self.match_combo.setToolTip("Candidate support slices, best first. Picking one jumps the support pane.")
        self.match_combo.currentIndexChanged.connect(self._on_match_pick)
        self.accept_btn = QPushButton("Accept match → anchor")
        self.accept_btn.setToolTip(
            "Confirm the pair: its anchor mask is matched on the query slice, the z map is "
            "refitted, and the query pane jumps to the next slice worth reviewing.")
        self.accept_btn.clicked.connect(self._on_accept_match)
        self.undo_btn = QPushButton("Undo last anchor")
        self.undo_btn.clicked.connect(self._on_undo_match)
        self.reset_match_btn = QPushButton("Reset matching session")
        self.reset_match_btn.clicked.connect(self._on_reset_session)
        self.propagate_btn = QPushButton("Propagate from anchors")
        self.propagate_btn.setObjectName("primary")
        self.propagate_btn.setToolTip(
            "Run SAM2 from the anchors collected above, bounded by the z map's predicted "
            "query window. No confirmed extent needed.")
        self.propagate_btn.clicked.connect(self._on_propagate_anchors)
        self.match_label = QLabel("pairs: none")
        self.match_label.setObjectName("seeds")
        self.match_label.setWordWrap(True)

        match_tab = QWidget()
        match_lay = QVBoxLayout()
        match_lay.setContentsMargins(8, 10, 8, 8)
        match_lay.setSpacing(6)
        for w in (self.find_btn, self.match_combo, self.accept_btn,
                  self.undo_btn, self.reset_match_btn, self.propagate_btn, self.match_label):
            match_lay.addWidget(w)
        match_lay.addStretch()
        match_tab.setLayout(match_lay)

        # one tab per route: only one is used at a time, and side by side they no longer
        # fit a sidebar
        self.route_tabs = QTabWidget()
        self.route_tabs.addTab(seed_tab, "Extent")
        self.route_tabs.addTab(match_tab, "Slice match")
        self.route_tabs.setToolTip(
            "Extent: confirm the roi's z range, then segment. Slice match: confirm one "
            "support match at a time, collect anchors, then propagate.")

        self.overlap_check = QCheckBox("Resolve overlaps by confidence")
        self.overlap_check.setToolTip(
            "Each roi's SAM2 tracker runs independently and can bleed onto an "
            "adjacent, similar-looking roi inside its own confirmed z range. When "
            "checked, any pixel claimed by more than one roi goes to whichever has "
            "the higher raw SAM2 confidence there. Off by default (previous behavior); "
            "toggle and re-run Segment to compare.")
        self.joint_check = QCheckBox("Joint propagation (shared tracker)")
        self.joint_check.setToolTip(
            "Track every confirmed roi in ONE shared SAM2 session instead of one "
            "independent session per roi. SAM2 then sees every roi at every frame, "
            "which discourages one tracker from drifting onto territory another "
            "already holds -- fixes cases 'Resolve overlaps by confidence' cannot, "
            "where only one roi's tracker ever reaches a pixel at all (nothing left "
            "to arbitrate there). Off by default (previous behavior); toggle and "
            "re-run Segment to compare. Combinable with the checkbox above.")
        self.fillgaps_check = QCheckBox("Fill single-slice gaps")
        self.fillgaps_check.setToolTip(
            "SAM2 can occasionally lose a roi for one isolated slice (confidence dips "
            "below 0 there, then recovers) -- most visible with joint propagation, "
            "where co-tracked rois compete for attention on a weak frame. When checked, "
            "an interior slice left fully empty for a roi is filled with the union of "
            "its immediate neighbours, IF both of those have a mask. A genuine multi-"
            "slice absence is left alone. Off by default; toggle and re-run Segment.")
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Show: segmentation", "Show: anchors"])
        self.view_combo.currentIndexChanged.connect(lambda _i: self._refresh_query_labels())
        self.export_btn = QPushButton("Export segmentation…")
        self.export_btn.clicked.connect(self._on_export)

        run_box = QGroupBox("Options & output")
        run_lay = QVBoxLayout()
        run_lay.setSpacing(6)
        run_lay.addWidget(self.overlap_check)
        run_lay.addWidget(self.joint_check)
        run_lay.addWidget(self.fillgaps_check)
        run_lay.addWidget(self.view_combo)
        run_lay.addWidget(self.export_btn)
        run_box.setLayout(run_lay)

        self.legend_box = QGroupBox("ROIs")
        self.legend_lay = QVBoxLayout()
        self.legend_lay.setSpacing(4)
        self.legend_box.setLayout(self.legend_lay)
        self.legend_box.setVisible(False)

        self.status = QLabel("Load a case to start.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)

        hint = QLabel("Slice: wheel, ◀ ▶, the number box, or arrows / PageUp / PageDown "
                      "after clicking a pane.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 8, 0)  # right margin leaves room for the scrollbar
        side.setSpacing(10)
        for w in (data_box, roi_box, self.route_tabs, run_box, self.legend_box):
            side.addWidget(w)
        side.addStretch()
        side.addWidget(hint)

        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel.setLayout(side)

        # the sidebar is taller than a short screen, so it scrolls instead of squeezing
        # its widgets into each other; status stays pinned below it, always visible
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        wrap = QVBoxLayout()
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.setSpacing(6)
        wrap.addWidget(scroll, stretch=1)
        wrap.addWidget(self.status)

        holder = QWidget()
        holder.setLayout(wrap)
        holder.setFixedWidth(288)
        return holder

    def _refresh_legend(self):
        while self.legend_lay.count():
            item = self.legend_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, name in enumerate(self.roi_names, start=1):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: {ROI_COLORS[(i - 1) % len(ROI_COLORS)]}; border-radius: 3px;")
            real = self.roi_real[i - 1] if i - 1 < len(self.roi_real) else ""
            row.addWidget(swatch)
            row.addWidget(QLabel(f"{name}  ·  {real}"))
            row.addStretch()
            holder = QWidget()
            holder.setObjectName("legendRow")
            holder.setLayout(row)
            self.legend_lay.addWidget(holder)
        self.legend_box.setVisible(bool(self.roi_names))

    def _update_enabled(self):
        has_query = self.query_vol is not None
        has_both = has_query and self.support_vol is not None
        self.suggest_btn.setEnabled(has_both and bool(self.roi_names))
        self.confirm_btn.setEnabled(has_query and bool(self.roi_names))
        self.kind_combo.setEnabled(has_query and bool(self.roi_names))
        self.clear_btn.setEnabled(bool(self.windows))
        self.segment_btn.setEnabled(has_both and bool(self.windows))

        has_roi = has_both and bool(self.roi_names)
        pairs = self.session.pairs.get(self.roi_combo.currentText(), []) if self.session else []
        self.find_btn.setEnabled(has_roi)
        self.match_combo.setEnabled(self.match_combo.count() > 0)
        self.accept_btn.setEnabled(has_roi and self.match_combo.count() > 0)
        self.undo_btn.setEnabled(bool(pairs))
        self.reset_match_btn.setEnabled(self.session is not None)
        self.propagate_btn.setEnabled(bool(self.session and self.session.anchors()))
        self.view_combo.setEnabled(bool(self.result))
        self.export_btn.setEnabled(bool(self.result))

    # -- extent ----------------------------------------------------------------
    def _on_roi_changed(self):
        """Pre-fill from/to for the selected roi: its confirmed window, else its
        suggestion, else leave the spin boxes alone. Also flags the range on the query
        pane and the roi's true extent on the support pane."""
        roi = self.roi_combo.currentText()
        rng = self.windows.get(roi) or self.suggested.get(roi)
        if rng is not None and self.lo_spin.isEnabled():
            lo, hi = rng
            for w, v in ((self.lo_spin, lo), (self.hi_spin, hi)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

        self.kind_combo.blockSignals(True)
        self.kind_combo.setCurrentIndex(1 if self.prompt_kind.get(roi) == "box" else 0)
        self.kind_combo.blockSignals(False)

        anchored = set(self.anchors.get(roi, {})) if self.session else set()
        if anchored:  # a live matching session: its anchors are the useful marks
            self.query_pane.set_marks(anchored, "anchor")
        else:
            confirmed = self.windows.get(roi)
            self.query_pane.set_marks(
                set(range(confirmed[0], confirmed[1] + 1)) if confirmed else set(), "extent")
        self._refresh_match_label()
        if self.support_lbl is not None and roi in self.roi_names:
            i = self.roi_names.index(roi) + 1
            present = {int(z) for z in np.flatnonzero((self.support_lbl == i).any(axis=(1, 2)))}
            self.support_pane.set_marks(present, roi)

    # -- lifecycle -------------------------------------------------------------
    def closeEvent(self, event):
        """Free the GPU before the window goes away: a Qt app can outlive its main window
        (or exit slowly), and the weights stay resident the whole time otherwise."""
        self._release_seg()
        super().closeEvent(event)

    # -- results -----------------------------------------------------------------
    def _anchors_as_shown(self) -> dict:
        """Return: self.anchors with each 'box' roi's mask replaced by its mask_to_box
        rectangle, so the anchors view shows what propagate() actually fed SAM2."""
        out = {}
        for roi, roi_prompts in self.anchors.items():
            if self.prompt_kind.get(roi) != "box":
                out[roi] = roi_prompts
                continue
            boxed = {}
            for idx, mask in roi_prompts.items():
                box = mask_to_box(mask)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                rect = np.zeros_like(mask)
                rect[y0:y1, x0:x1] = True
                boxed[idx] = rect
            out[roi] = boxed
        return out

    def _refresh_query_labels(self):
        """Rebuild the query pane's label volume from whichever result the user selected."""
        if self.query_vol is None:
            return
        masks = self._anchors_as_shown() if self.view_combo.currentIndex() == 1 else self.result
        if not masks:
            self.query_pane.set_labels(None)
            return
        self.query_pane.set_labels(masks_to_labels(masks, self.query_vol.shape, self.roi_names))
