"""Arterial disease-resolved spectral-cascade solver and study orchestration."""

from .core import MODEL_SCHEMA, SOLVER_SCHEMA, RESULT_SCHEMA, PARENT_REFERENCE_SCHEMA, ModelConstants, CaseSpec, Lesion, prepare_case
from .storage import ProjectPaths, init_project_paths, run_case
from .parent import run_parent_reference_audit, run_paired_case, full_verification_suite
from .study import STUDY_CONFIG, EVIDENCE_PROFILES, EVIDENCE_REFERENCES, evidence_profile_table, run_full_study, run_study_mode

__version__ = "0.1.0"

__all__ = [
    "MODEL_SCHEMA","SOLVER_SCHEMA","RESULT_SCHEMA","PARENT_REFERENCE_SCHEMA",
    "ModelConstants","CaseSpec","Lesion","prepare_case","ProjectPaths","init_project_paths","run_case",
    "run_parent_reference_audit","run_paired_case","full_verification_suite",
    "STUDY_CONFIG","EVIDENCE_PROFILES","EVIDENCE_REFERENCES","evidence_profile_table","run_full_study","run_study_mode"
]
