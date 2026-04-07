"""Allied Vision camera capture via the Vimba SDK.

Single-frame acquisition from USB3 or GigE Allied Vision cameras.

Requires: vmbpy (``pip install vmbpy``).
Only needed on the lab machine with the physical camera.

Ported from ~/slm-code/avt.py.
"""

from __future__ import annotations

from contextlib import ExitStack
from threading import Event

import numpy as np

try:
    from vmbpy import FrameStatus, VmbSystem
except ImportError as exc:
    raise ImportError(
        "vmbpy is required for camera capture. "
        "Install with: pip install vmbpy"
    ) from exc


class VimbaCamera:
    """Allied Vision camera interface for focal-plane intensity capture.

    Parameters
    ----------
    cam_index : index of the camera to open (0-based).

    Example
    -------
    ::

        with VimbaCamera(cam_index=0) as camera:
            image = camera.capture(exposure_time_us=30)
    """

    def __init__(self, cam_index: int = 0) -> None:
        self._stack = ExitStack()
        self._vmb = self._stack.enter_context(VmbSystem.get_instance())
        cameras = self._vmb.get_all_cameras()
        if not cameras:
            self._stack.close()
            raise RuntimeError("No Allied Vision camera found")
        if cam_index >= len(cameras):
            self._stack.close()
            raise IndexError(
                f"cam_index {cam_index} out of range "
                f"(found {len(cameras)} camera(s))"
            )
        self._cam = self._stack.enter_context(cameras[cam_index])

    def capture(
        self, exposure_time_us: float, timeout: float = 2.0
    ) -> np.ndarray:
        """Acquire a single frame and return it as a 2-D numpy array.

        Parameters
        ----------
        exposure_time_us : exposure time in microseconds.
        timeout : maximum seconds to wait for a frame.

        Returns
        -------
        Grayscale image as a 2-D uint8 or uint16 array (H, W).
        """
        grab_event = Event()
        grabbed: dict = {}

        def handler(cam, stream, frame):
            if frame.get_status() == FrameStatus.Complete:
                grabbed["frame"] = frame
                grab_event.set()
            stream.queue_frame(frame)

        self._cam.get_feature_by_name("ExposureTime").set(exposure_time_us)
        self._cam.start_streaming(handler=handler, buffer_count=1)

        if not grab_event.wait(timeout=timeout):
            self._cam.stop_streaming()
            raise TimeoutError("Frame acquisition timed out")

        self._cam.stop_streaming()
        return grabbed["frame"].as_numpy_ndarray()[:, :, 0]

    def close(self) -> None:
        """Release camera and SDK resources."""
        self._stack.close()

    def __enter__(self) -> VimbaCamera:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
