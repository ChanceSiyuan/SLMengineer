"""Full-screen SLM display via wxPython.

Displays uint8 phase patterns on a secondary monitor connected to the
LCOS-SLM. Runs wxPython in a background thread so the caller is not
blocked by the GUI main loop.

Requires: wxPython (``pip install wxPython``).
Only needed on the Windows lab machine with a physical SLM.

Ported from ~/slm-code/slmpy.py (Sebastien Popoff, 2015).
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import wx
except ImportError as exc:
    raise ImportError(
        "wxPython is required for SLM display. "
        "Install with: pip install wxPython"
    ) from exc


EVT_NEW_IMAGE = wx.PyEventBinder(wx.NewEventType(), 0)


class _ImageEvent(wx.PyCommandEvent):
    """Custom event carrying a wx.Image to display."""

    def __init__(self, event_type: int = EVT_NEW_IMAGE.evtType[0], id: int = 0):
        super().__init__(event_type, id)
        self.img: wx.Image | None = None
        self.event_lock: threading.Lock | None = None


class _SLMFrame(wx.Frame):
    """Full-screen frame on the target monitor."""

    def __init__(self, monitor: int, is_image_lock: bool = True):
        self.is_image_lock = is_image_lock
        x0, y0, res_x, res_y = wx.Display(monitor).GetGeometry()
        self._x0, self._y0, self._res_x, self._res_y = x0, y0, res_x, res_y
        super().__init__(
            None, -1, "SLM window", pos=(x0, y0), size=(res_x, res_y)
        )
        self.img = wx.Image(2, 2)
        self.bmp = self.img.ConvertToBitmap()
        self.client_size = self.GetClientSize()
        self.Bind(EVT_NEW_IMAGE, self._on_update_image)
        self.ShowFullScreen(True, wx.FULLSCREEN_ALL)
        self.SetFocus()

    def _init_buffer(self) -> None:
        self.client_size = self.GetClientSize()
        self.bmp = self.img.Scale(
            self.client_size[0], self.client_size[1]
        ).ConvertToBitmap()
        dc = wx.ClientDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def _on_update_image(self, event: _ImageEvent) -> None:
        self._event_lock = event.event_lock
        self.img = event.img
        self._init_buffer()
        self._release_event_lock()

    def _release_event_lock(self) -> None:
        if hasattr(self, "_event_lock") and self._event_lock:
            if self._event_lock.locked():
                self._event_lock.release()

    def set_monitor(self, monitor: int) -> None:
        if monitor < 0 or monitor > wx.Display.GetCount() - 1:
            raise ValueError(f"Invalid monitor index: {monitor}")
        self._x0, self._y0, self._res_x, self._res_y = (
            wx.Display(monitor).GetGeometry()
        )


class _VideoThread(threading.Thread):
    """Runs the wxPython main loop in a daemon thread."""

    def __init__(self, parent: SLMDisplay):
        super().__init__(daemon=True)
        self.parent = parent
        self.frame: _SLMFrame | None = None
        self._lock = threading.Lock()
        self._lock.acquire()
        self.start()
        # Block until the frame is ready
        self._lock.acquire()
        self._lock.release()

    def run(self) -> None:
        app = wx.App()
        self.frame = _SLMFrame(
            monitor=self.parent._monitor,
            is_image_lock=self.parent._is_image_lock,
        )
        self.frame.Show(True)
        self._lock.release()
        app.MainLoop()


class SLMDisplay:
    """Interface for displaying phase patterns on a physical SLM.

    Parameters
    ----------
    monitor : display index (0-based) for the SLM output.
    is_image_lock : if True, ``update_array`` blocks until the previous
        frame has been rendered before accepting a new one.

    Example
    -------
    ::

        with SLMDisplay(monitor=0, is_image_lock=True) as slm:
            slm.update_array(phase_screen_uint8)
    """

    def __init__(self, monitor: int = 1, is_image_lock: bool = False) -> None:
        self._monitor = monitor
        self._is_image_lock = is_image_lock
        self._event_lock = threading.Lock()
        self._vt = _VideoThread(self)

    def get_size(self) -> tuple[int, int]:
        """Return (width, height) of the SLM display."""
        f = self._vt.frame
        return f._res_x, f._res_y

    def update_array(self, array: np.ndarray) -> None:
        """Send a uint8 array to the SLM display.

        The array is scaled to fill the screen. For greyscale (2-D) arrays,
        the image is broadcast to RGB automatically.
        """
        h, w = array.shape[:2]
        if array.ndim == 2:
            rgb = np.stack([array] * 3, axis=-1)
            data = rgb.tobytes()
        else:
            data = array.tobytes()

        img = wx.ImageFromBuffer(width=w, height=h, dataBuffer=data)
        event = _ImageEvent()
        event.img = img
        event.event_lock = self._event_lock

        if self._is_image_lock:
            event.event_lock.acquire()

        self._vt.frame.AddPendingEvent(event)

    def close(self) -> None:
        """Close the SLM display window."""
        self._vt.frame.Close()
        wx.DisableAsserts()

    def __enter__(self) -> SLMDisplay:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
