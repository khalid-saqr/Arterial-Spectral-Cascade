from copy import deepcopy
from arterial_spectral_cascade.study import STUDY_CONFIG, ALLOWED_MODES, evidence_profile_table, resolve_primary_width


def test_default_mode_and_public_modes():
    assert STUDY_CONFIG["RUN_MODE"] == "FULL_STUDY"
    assert ALLOWED_MODES == {"QUICK_CHECK", "VERIFICATION", "PARAMETER_SELECTION", "FIGURES", "FULL_STUDY"}


def test_evidence_representations_and_common_width():
    df = evidence_profile_table()
    assert set(df.profile_id) == {"S10", "S20", "S30", "D20", "D50", "D60"}
    cfg = deepcopy(STUDY_CONFIG)
    resolved = resolve_primary_width(cfg, N=128)
    assert resolved["width"] > 0
    assert resolved["design_limit"] <= 0.3
