import os
import sys

if sys.platform.startswith("linux") and os.environ.get("QT_QPA_PLATFORM") == "x11":
    # "x11" is not a valid Qt platform plugin name (the X11 backend is "xcb") --
    # some shells/venvs persist this wrong value and Qt fails to start.
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from gui.main import main

if __name__ == "__main__":
    main()
