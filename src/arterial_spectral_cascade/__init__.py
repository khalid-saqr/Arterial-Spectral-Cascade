"""Coefficient-space arterial spectral-cascade solver and study orchestration."""

from .core import (
    MODEL_SCHEMA,
    SOLVER_SCHEMA,
    RESULT_SCHEMA,
    PARENT_REFERENCE_SCHEMA,
    ModelConstants,
    CaseSpec,
    PreparedCase,
    Lesion,
    DistributedMode,
    localized_morphology,
    multiple_morphology,
    distributed_morphology,
    morphology_field,
    prepare_case,
    make_matched_mean,
)
from .storage import ProjectPaths, init_project_paths, run_case
from .parent import run_parent_reference_audit, run_paired_case, full_verification_suite
from .study import (
    STUDY_CONFIG,
    MORPHOLOGY_CLASSES,
    morphology_class_table,
    configured_case_table,
    case_record_to_spec,
    geometry_derived_spec,
    run_full_study,
    run_study_mode,
)
from . import performance as _performance
from .performance import benchmark_performance_backend, verify_optimized_backend_equivalence

PERFORMANCE_BACKEND_STATUS = _performance.install_performance_backend()
_performance.PERFORMANCE_BACKEND_STATUS = PERFORMANCE_BACKEND_STATUS
# Rebind after installation so the exported verification suite includes the
# optimized/reference equivalence check when the optimized backend is active.
from .parent import full_verification_suite as full_verification_suite

__version__ = "0.3.0"

__all__ = [
    "MODEL_SCHEMA","SOLVER_SCHEMA","RESULT_SCHEMA","PARENT_REFERENCE_SCHEMA",
    "ModelConstants","CaseSpec","PreparedCase","Lesion","DistributedMode",
    "localized_morphology","multiple_morphology","distributed_morphology","morphology_field",
    "prepare_case","make_matched_mean","ProjectPaths","init_project_paths","run_case",
    "run_parent_reference_audit","run_paired_case","full_verification_suite",
    "STUDY_CONFIG","MORPHOLOGY_CLASSES","morphology_class_table","configured_case_table",
    "case_record_to_spec","geometry_derived_spec","run_full_study","run_study_mode",
    "PERFORMANCE_BACKEND_STATUS","benchmark_performance_backend","verify_optimized_backend_equivalence",
]
