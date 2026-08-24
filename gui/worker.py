from qtpy.QtCore import QObject, QThread, Signal


class _Worker(QObject):
    """Runs one callable on a background thread so the Qt event loop keeps pumping
    (repaints, resize, close) while SAM2 matches/propagates. The callable receives a
    `report(str)` function for progress -- it must never touch a widget directly from
    here, since Qt widgets are only safe to touch from the GUI thread. `report` is
    just the `progress` signal's emit, so it's already thread-safe: Qt queues the
    connected slot back onto the receiver's (GUI) thread."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn(self.progress.emit)
        except Exception as e:
            self.error.emit(str(e))
        else:
            self.finished.emit(result)


def run_in_thread(owner, fn, on_progress, on_finished, on_error):
    """Start fn(report) on a background QThread owned by `owner`. report(msg: str) is
    safe to call from `fn` for a status update. on_progress/on_finished/on_error run
    back on the GUI thread once the corresponding signal fires.

    The caller MUST keep the returned (thread, worker) pair alive on an attribute
    (e.g. self._bg_thread, self._bg_worker) until on_finished/on_error runs -- nothing
    else references them, so Python (or Qt) can garbage-collect a QThread that's still
    running otherwise."""
    thread = QThread(owner)
    worker = _Worker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress is not None:
        worker.progress.connect(on_progress)

    def _cleanup():
        thread.quit()
        thread.wait()

    def _on_finished(result):
        _cleanup()
        on_finished(result)

    def _on_error(msg):
        _cleanup()
        on_error(msg)

    worker.finished.connect(_on_finished)
    worker.error.connect(_on_error)
    thread.start()
    return thread, worker
