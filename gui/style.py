ROI_COLORS = ["#ff5252", "#40c4ff", "#ffd740", "#69f0ae", "#e040fb", "#ff9100"]
OVERLAY_ALPHA = 0.35

SCROLL_MIN_INTERVAL = 0.09

BG = "#1e1f26"
PANEL = "#282a36"
FG = "#e6e6ea"
ACCENT = "#5b9dd9"

STYLESHEET = f"""
QMainWindow, QWidget {{ background: {BG}; color: {FG};
                        font-family: "Helvetica Neue", Arial; font-size: 12px; }}
QWidget#legendRow {{ background: transparent; }}
QGroupBox {{ background: {PANEL}; border: 1px solid #3a3d4d; border-radius: 6px;
             margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: #9aa0b5; font-size: 11px; text-transform: uppercase;
                    letter-spacing: 1px; }}
QPushButton {{ background: #363948; border: 1px solid #454857; border-radius: 5px;
               padding: 7px 10px; color: {FG}; text-align: left; }}
QPushButton:hover {{ background: #414459; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #2c2e3b; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #10131a;
                       font-weight: 700; text-align: center; }}
QPushButton#primary:hover {{ background: #6fb0e8; }}
QPushButton#primary:disabled {{ background: #3a3d4d; border-color: #3a3d4d; color: #6b6f80; }}
QPushButton#stepBtn {{ padding: 2px 0; text-align: center; font-weight: 700;
                       min-width: 24px; max-width: 24px; }}
QSpinBox {{ background: #363948; border: 1px solid #454857; border-radius: 5px;
            padding: 3px 4px; color: {FG}; font-family: Menlo, Consolas, monospace;
            font-size: 11px; min-width: 46px; }}
QSpinBox:hover {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 13px; background: #454857;
                                              border: none; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {ACCENT}; }}
QComboBox {{ background: #363948; border: 1px solid #454857; border-radius: 5px;
             padding: 5px 8px; color: {FG}; }}
QComboBox:hover {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; color: {FG};
                               selection-background-color: {ACCENT}; selection-color: #10131a; }}
QSlider::groove:horizontal {{ height: 4px; background: #3a3d4d; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 13px; margin: -5px 0;
                              border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: #7fbcf0; }}
QSlider::sub-page:horizontal {{ background: #46688c; border-radius: 2px; }}
QLabel#hint {{ color: #8b90a3; font-size: 11px; }}
QLabel#status {{ color: #9aa0b5; padding: 4px 2px; }}
QLabel#seeds {{ color: #ffd740; font-family: Menlo, Consolas, monospace; font-size: 11px; }}
QLabel#paneTitle {{ color: #9aa0b5; font-size: 11px; font-weight: 700;
                    letter-spacing: 1px; text-transform: uppercase; }}
QLabel#zLabel {{ color: #9aa0b5; font-family: Menlo, Consolas, monospace;
                 font-size: 11px; min-width: 74px; }}
QFrame#pane {{ background: {PANEL}; border: 1px solid #3a3d4d; border-radius: 6px; }}
QWidget#sidePanel {{ background: transparent; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #454857; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QTabWidget::pane {{ background: {PANEL}; border: 1px solid #3a3d4d; border-radius: 6px;
                    top: -1px; }}
QTabBar::tab {{ background: #2c2e3b; color: #9aa0b5; border: 1px solid #3a3d4d;
                border-bottom: none; border-top-left-radius: 5px;
                border-top-right-radius: 5px; padding: 5px 12px; margin-right: 2px;
                font-size: 11px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {FG}; }}
QTabBar::tab:hover {{ color: {FG}; }}
QSplitter::handle {{ background: transparent; width: 10px; }}
QSplitter::handle:hover {{ background: #3a3d4d; }}
"""
