"""Parent-reference, matched-mean, paired integration, and Solver Design verification."""
from .parent_base import run_parent_detailed_case
from . import paired_runtime as _paired_runtime
from .paired_runtime import *
from . import solver_verification as _solver_verification
from .solver_verification import *

# Preserve the historical monkeypatch/test surfaces on arterial_spectral_cascade.parent
# while keeping focused implementations in their own modules.
def run_paired_case(*args, **kwargs):
    if "etdrk4_step" in globals():
        _paired_runtime.etdrk4_step = globals()["etdrk4_step"]
    return _paired_runtime.run_paired_case(*args, **kwargs)


def run_parent_reference_audit(*args, **kwargs):
    if "run_case" in globals():
        _solver_verification.run_case = globals()["run_case"]
    return _solver_verification.run_parent_reference_audit(*args, **kwargs)


__all__ = [name for name in globals() if not name.startswith("_")]
