"""SLM iterative optimization algorithms for holographic beam shaping.

Scripts under ``scripts/`` import submodules directly
(``from slm.cgm import ...``, ``from slm.generation import SLM_class``,
``from slm import imgpy``, etc.).  Nothing is re-exported at package
level so the package loads lazily and broken peripheral modules do not
block the core path.
"""

__version__ = "0.1.0"
