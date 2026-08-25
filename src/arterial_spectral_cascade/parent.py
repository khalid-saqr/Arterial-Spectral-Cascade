"""Parent-reference and paired integration compatibility surface."""
from .parent_base import *
from . import paired_runtime as _paired_runtime
from .paired_runtime import run_paired_case, verify_paired_restart_equivalence

# Preserve the historical monkeypatch/test surface on arterial_spectral_cascade.parent.
def run_paired_case(*args, **kwargs):
    if "etdrk4_step" in globals():
        _paired_runtime.etdrk4_step = globals()["etdrk4_step"]
    return _paired_runtime.run_paired_case(*args, **kwargs)

__all__ = [name for name in globals() if not name.startswith("_")]
