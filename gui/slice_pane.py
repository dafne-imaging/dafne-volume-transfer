import time

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QSpinBox, QVBoxLayout,
)

from gui.style import OVERLAY_ALPHA, PANEL, ROI_COLORS, SCROLL_MIN_INTERVAL


def load_case(path: str):
    """.npz with 'data' [H,W,Z] and one 'mask_<name>' [H,W,Z] bool per roi -> (vol, labels, roi_names)."""
    d = np.load(path)
    vol = d["data"].transpose(2, 0, 1)  # [H,W,Z] -> [Z,H,W]
    rois = sorted(k[5:] for k in d.files if k.startswith("mask_"))
    lbl = np.zeros(vol.shape, dtype=np.int32)
    for i, name in enumerate(rois, start=1):
        lbl[d[f"mask_{name}"].transpose(2, 0, 1)] = i
    return vol, lbl, rois


class SlicePane(QFrame):
    """One volume + its label volume, drawn together on a single axes, with a z slider."""

    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.setObjectName("pane")
        self.vol: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.n_rois = 0
        self.marks: set[int] = set()  # slices to flag in the header (seeds / anchors)
        self.mark_word = "seed"
        self._last_scroll = 0.0

        self._title = QLabel(title)
        self._title.setObjectName("paneTitle")
        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("hint")
        # the subtitle is a caption: it must never be what decides the pane's min width
        self._subtitle.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._subtitle.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.fig = Figure(facecolor=PANEL)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(160, 160)  # keeps the image from collapsing to a line
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor("black")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", lambda _e: self.setFocus())
        self.setFocusPolicy(Qt.StrongFocus)  # so the arrow keys below reach this pane

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.valueChanged.connect(self._on_slider)

        self.spin = QSpinBox()
        self.spin.setEnabled(False)
        self.spin.setKeyboardTracking(False)  # jump on Enter/arrows, not mid-typing
        self.spin.valueChanged.connect(self.set_z)

        prev_btn = QPushButton("◀")
        prev_btn.setObjectName("stepBtn")
        prev_btn.setToolTip("Previous slice")
        prev_btn.clicked.connect(lambda: self.set_z(self.z - 1))
        next_btn = QPushButton("▶")
        next_btn.setObjectName("stepBtn")
        next_btn.setToolTip("Next slice")
        next_btn.clicked.connect(lambda: self.set_z(self.z + 1))
        self._step_btns = (prev_btn, next_btn)
        for b in self._step_btns:
            b.setEnabled(False)

        self.z_label = QLabel("/ -")
        self.z_label.setObjectName("zLabel")
        self.z_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._title)
        head.addStretch()
        head.addWidget(self._subtitle)

        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.setSpacing(6)
        foot.addWidget(prev_btn)
        foot.addWidget(self.slider, stretch=1)
        foot.addWidget(next_btn)
        foot.addWidget(self.spin)
        foot.addWidget(self.z_label)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(head)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(foot)
        self.setLayout(layout)
        # the z row (buttons + slider + spin + label) is the pane's real floor
        self.setMinimumWidth(230)

    #  state 
    def set_volume(self, vol: np.ndarray, labels: np.ndarray | None, n_rois: int):
        self.vol, self.labels, self.n_rois = vol, labels, n_rois
        last = vol.shape[0] - 1
        for w, enable in ((self.slider, True), (self.spin, True)):
            w.blockSignals(True)
            w.setEnabled(enable)
            w.setMinimum(0)
            w.setMaximum(last)
            w.setValue(vol.shape[0] // 2)
            w.blockSignals(False)
        for b in self._step_btns:
            b.setEnabled(True)
        self.redraw()

    def set_labels(self, labels: np.ndarray | None):
        self.labels = labels
        self.redraw()

    def set_marks(self, marks: set, word: str = "seed"):
        self.marks, self.mark_word = set(marks), word
        self.redraw()

    @property
    def z(self) -> int:
        return self.slider.value()

    def set_z(self, z: int):
        """Single entry point for every navigation control, clamped to the volume."""
        if not self.slider.isEnabled():
            return
        z = max(self.slider.minimum(), min(self.slider.maximum(), int(z)))
        if z != self.slider.value():
            self.slider.setValue(z)  # _on_slider syncs the spin box and redraws
        else:
            self._sync_spin(z)

    def _on_slider(self, value: int):
        self._sync_spin(value)
        self.redraw()

    def _sync_spin(self, value: int):
        if self.spin.value() != value:
            self.spin.blockSignals(True)
            self.spin.setValue(value)
            self.spin.blockSignals(False)

    def _on_scroll(self, event):
        if not self.slider.isEnabled():
            return
        now = time.monotonic()
        if now - self._last_scroll < SCROLL_MIN_INTERVAL:
            return  # inside the throttle window: drop this event entirely
        self._last_scroll = now
        step = event.step or 0
        up = step > 0 if step else event.button == "up"
        self.set_z(self.z + (1 if up else -1))  # direction only, never a multi-slice jump

    def keyPressEvent(self, event):
        deltas = {Qt.Key_Left: -1, Qt.Key_Down: -1, Qt.Key_Right: 1, Qt.Key_Up: 1,
                  Qt.Key_PageDown: -5, Qt.Key_PageUp: 5}
        key = event.key()
        if key in deltas:
            self.set_z(self.z + deltas[key])
        elif key == Qt.Key_Home:
            self.set_z(self.slider.minimum())
        elif key == Qt.Key_End:
            self.set_z(self.slider.maximum())
        else:
            super().keyPressEvent(event)

    #  drawing
    def redraw(self):
        if self.vol is None:
            return
        z = self.z
        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.imshow(self.vol[z], cmap="gray", interpolation="nearest")

        if self.labels is not None:
            lbl = self.labels[z]
            overlay = np.zeros(lbl.shape + (4,), dtype=np.float32)
            for i in range(1, self.n_rois + 1):
                m = lbl == i
                if not m.any():
                    continue
                rgb = to_rgb(ROI_COLORS[(i - 1) % len(ROI_COLORS)])
                overlay[m, :3] = rgb
                overlay[m, 3] = OVERLAY_ALPHA
                self.ax.contour(m, levels=[0.5], colors=[rgb], linewidths=1.3)
            if overlay[..., 3].any():
                self.ax.imshow(overlay, interpolation="nearest")

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for s in self.ax.spines.values():
            s.set_visible(False)
        self.canvas.draw_idle()

        tag = f"  ({self.mark_word})" if z in self.marks else ""
        self.z_label.setText(f"/ {self.vol.shape[0] - 1}{tag}")
