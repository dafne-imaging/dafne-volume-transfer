import sys

from qtpy.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.style import STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
