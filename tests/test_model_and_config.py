from copy import deepcopy
import numpy as np

from arterial_spectral_cascade.core import (
    CONSTS, CaseSpec, Lesion, DistributedMode, prepare_case, make_matched_mean,
)
from arterial_spectral_cascade.study import (
    STUDY_CONFIG, ALLOWED_MODES, MORPHOLOGY_CLASSES, morphology_class_table,
    case_record_to_spec, geometry_derived_spec,
)


def test_default_mode_and_public_modes():
    assert STUDY_CONFIG["RUN_MODE"] == "FULL_STUDY"
    assert ALLOWED_MODES == {"QUICK_CHECK", "VERIFICATION", "PARAMETER_SELECTION", "FIGURES", "FULL_STUDY"}
    assert set(MORPHOLOGY_CLASSES) == {"DL", "DM", "DR"}
    assert set(morphology_class_table().case_class) == {"DL", "DM", "DR"}


def test_localized_coefficient_closure_matches_mathematical_model():
    spec = CaseSpec("DL", Wo0=10.0, N=128, dt=.002, T_final=.02,
                    chi_b=.12, chi_g=-.08, w=3.0, p=1,
                    R0_over_L0=.01, slow_variation_limit=.1)
    prep = prepare_case(spec)
    b0 = CONSTS.b_ref * spec.Wo0**-2
    g0 = CONSTS.g_ref * (1 + CONSTS.Cg/spec.Wo0)
    np.testing.assert_allclose(prep.b, b0*(1 + spec.chi_b*prep.psi_D), rtol=0, atol=2e-14)
    np.testing.assert_allclose(prep.g, g0*(1 + spec.chi_g*prep.psi_D), rtol=0, atol=2e-14)
    assert prep.admissibility["status"] == "ADMISSIBLE"


def test_parent_pointwise_recovery_has_zero_disease_field():
    spec = CaseSpec("P1", Wo0=5.0, N=96, dt=.002, T_final=.02, eps_b=.2, eps_g=.1, q=1.0)
    prep = prepare_case(spec)
    assert np.max(np.abs(prep.psi_D)) == 0.0
    b0 = CONSTS.b_ref * spec.Wo0**-2
    g0 = CONSTS.g_ref * (1 + CONSTS.Cg/spec.Wo0)
    np.testing.assert_allclose(prep.b, b0*(1 + spec.eps_b*np.cos(spec.q*prep.grid.xi)), atol=2e-14)
    np.testing.assert_allclose(prep.g, g0*(1 + spec.eps_g*np.cos(spec.q*prep.grid.xi)), atol=2e-14)


def test_multiple_and_distributed_morphologies_are_normalized():
    dm = CaseSpec("DM", Wo0=10, N=128, dt=.002, T_final=.02, chi_b=.05, chi_g=.05,
                  lesions=(Lesion(1.0, 2.0, 2.5, 1), Lesion(.7, 7.0, 1.8, 2)),
                  R0_over_L0=.01, slow_variation_limit=.1)
    p_dm = prepare_case(dm)
    assert -1e-14 <= p_dm.psi_D_raw.min() <= 1
    assert abs(p_dm.psi_D_raw.max()-1.0) < 1e-14

    dr = CaseSpec("DR", Wo0=10, N=128, dt=.002, T_final=.02, chi_b=.05, chi_g=-.04,
                  distributed_modes=(DistributedMode(1.0,1.0,0.0), DistributedMode(.3,2.0,.2)),
                  morphology_scale=2.0, R0_over_L0=.01, slow_variation_limit=.1)
    p_dr = prepare_case(dr)
    assert abs(p_dr.psi_D_raw.min()) < 1e-14
    assert abs(p_dr.psi_D_raw.max()-1.0) < 1e-14


def test_matched_mean_preserves_exact_parent_means():
    spec = CaseSpec("DL", Wo0=10, N=96, dt=.002, T_final=.02, chi_b=.1, chi_g=.06, w=3.0,
                    R0_over_L0=.01, slow_variation_limit=.1)
    parent = prepare_case(spec)
    mm = make_matched_mean(parent)
    assert mm.spec.case_class == "MM"
    np.testing.assert_allclose(mm.b, parent.b_bar)
    np.testing.assert_allclose(mm.g, parent.g_bar)
    assert np.max(np.abs(mm.b_tilde)) < 1e-14
    assert np.max(np.abs(mm.g_tilde)) < 1e-14
    assert mm.parent_case_id == parent.case_id


def test_sampled_geometry_is_input_morphology_not_local_womersley_closure():
    cfg=deepcopy(STUDY_CONFIG); cfg["R0_OVER_L0"]=.01; cfg["SLOW_VARIATION_LIMIT"]=.1
    N=64; xi=np.linspace(0,4*np.pi,N,endpoint=False); psi=.5*(1+np.cos(xi))
    spec=geometry_derived_spec(psi,"synthetic geometry-derived morphology for test",Wo=10,chi_b=.05,chi_g=-.03,
                               morphology_scale=2.0,N=N,dt=.002,T_final=.02,cfg=cfg)
    prep=prepare_case(spec)
    np.testing.assert_allclose(prep.psi_D_raw,psi)
    assert prep.morphology_provenance.startswith("synthetic geometry-derived")
    assert not hasattr(prep,"Wo_R")
    assert not hasattr(prep,"r")


def test_case_record_requires_explicit_sensitivities_and_disease_only_background():
    cfg=deepcopy(STUDY_CONFIG); cfg["R0_OVER_L0"]=.01; cfg["SLOW_VARIATION_LIMIT"]=.1
    rec={"case_id":"L1","case_class":"DL","w":3.0,"p":1,"chi_b":.1,"chi_g":.05}
    spec=case_record_to_spec(rec,10,N=64,dt=.002,T_final=.02,cfg=cfg)
    assert spec.eps_b==0 and spec.eps_g==0
