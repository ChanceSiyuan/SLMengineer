"""SLM iterative optimization algorithms for holographic beam shaping.

Scripts under ``scripts/`` import submodules directly
(``from slm.cgm import ...``, ``from slm.generation import SLM_class``,
``from slm import imgpy``, etc.).  Nothing is re-exported at package
level so the package loads lazily and broken peripheral modules do not
block the core path.
"""

__version__ = "0.1.0"

from .dataio import (  # noqa: E402,F401
    load_bmp,
    colorize,
    compute_reweight,
    detect_sheet_bbox,
)
from .gs import gs_phase_correction  # noqa: E402,F401
from .feedback import (  # noqa: E402,F401
    load_camera_roi,
    embed_camera_into_focal,
    align_camera_to_focal,
)
