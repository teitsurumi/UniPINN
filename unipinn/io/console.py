"""Console I/O utilities: stream tee and timed input."""

import threading
import queue


class StreamTee:
    """Wraps multiple streams, writing to all simultaneously.

    Used to capture console output while still printing to terminal.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for s in self.streams:
            try:
                s.write(message)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def input_with_timeout(prompt: str, timeout: float = 10.0, default: str = "y") -> str:
    """Prompt for terminal input with a timeout.

    Returns ``default`` if the user does not respond within ``timeout`` seconds.
    """
    q = queue.Queue()

    def _worker():
        try:
            q.put(input(prompt).strip().lower())
        except Exception:
            q.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        res = q.get(timeout=timeout)
        return res if res is not None else default
    except queue.Empty:
        return default
