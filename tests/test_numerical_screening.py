import json
import numpy as np

import arterial_spectral_cascade.planning as planning
import arterial_spectral_cascade.storage as storage
from arterial_spectral_cascade.core import CaseSpec, prepare_case
from arterial_spectral_cascade.study import convergence_acceptance


def test_run_case_stops_on_nonfinite_spectral_state(monkeypatch):
    spec=CaseSpec("H0",Wo0=10.0,N=32,dt=0.01,T_final=0.03,output_every_steps=1,checkpoint_every_steps=1)
    prep=prepare_case(spec)
    def bad_step(ahat,prep,etd,include_nonlinearity=True,include_heterogeneity=True):
        out=np.array(ahat,dtype=np.complex128,copy=True); out[0]=np.nan+0j; return out
    monkeypatch.setattr(storage,"etdrk4_step",bad_step)
    result=storage.run_case(prep,paths=None,resume=False,progress=False); summary=result["summary"]
    assert summary["completed"] is False
    assert summary["runtime_valid"] is False
    assert summary["numerical_status"] == "UNSTABLE"
    assert summary["termination_reason"] == "NONFINITE_SPECTRAL_STATE"
    assert summary["termination_step"] == 1
    assert summary["R_max"] is None and summary["I2_final"] is None
    json.dumps(summary,allow_nan=False)


def test_temporal_convergence_excludes_unstable_trial_and_is_json_safe(monkeypatch):
    case=CaseSpec("H0",Wo0=10.0,N=32,dt=0.02,T_final=0.16,output_every_steps=1,checkpoint_every_steps=1)
    def fake_run_case(prep,paths=None,resume=False,progress=False):
        dt=prep.spec.dt
        if np.isclose(dt,0.08):
            summary={"runtime_valid":False,"numerical_status":"UNSTABLE","termination_reason":"NONFINITE_SPECTRAL_STATE",
                     "termination_step":2,"termination_s":0.16,"R_max":np.inf,"s_R_max":np.nan,"I2_final":np.nan}
            return {"summary":summary}
        ss=np.arange(0.0,prep.spec.T_final+1e-12,0.04)
        perturb=dt*1e-7
        I2=1.0+0.01*ss+perturb; R=0.2+0.02*ss+perturb
        hist={"s":ss,"I2":I2,"R":R,"eta_tail":np.full_like(ss,1e-10),
              "RI1_integrated":np.full_like(ss,dt*1e-12),"RE_integrated":np.full_like(ss,dt*1e-12)}
        summary={"runtime_valid":True,"numerical_status":"VALID","termination_reason":None,
                 "termination_step":int(round(prep.spec.T_final/dt)),"termination_s":prep.spec.T_final,
                 "R_max":float(R.max()),"s_R_max":float(ss[-1]),"I2_final":float(I2[-1]),
                 "eta_tail_max":1e-10,"chi_h_max":0.1,"max_abs_RI1_integrated":dt*1e-12,"max_abs_RE_integrated":dt*1e-12}
        return {"summary":summary,"history":hist}
    monkeypatch.setattr(planning,"run_case",fake_run_case)
    report=planning.temporal_convergence(case,[0.08,0.04,0.02],progress=False); rows=report["rows"]
    assert rows[0]["model_status"]=="ADMISSIBLE" and rows[0]["numerical_status"]=="UNSTABLE"
    assert rows[0]["status"]=="NUMERICALLY_UNSTABLE" and rows[0]["R_max"] is None and rows[0]["I2_final"] is None
    assert rows[2]["prev_dt"]==0.04
    accepted=convergence_acceptance(rows,"dt",1e-5,1e-3); assert accepted==0.04
    json.dumps(report,allow_nan=False)


def test_spatial_convergence_excludes_invalid_trial(monkeypatch):
    case=CaseSpec("H0",Wo0=10.0,N=64,dt=0.01,T_final=0.02,output_every_steps=1,checkpoint_every_steps=1)
    def fake_run_case(prep,paths=None,resume=False,progress=False):
        if prep.spec.N==32:
            summary={"runtime_valid":False,"numerical_status":"INVALID","termination_reason":"RUNTIME_STATE_CHECK_FAILED",
                     "termination_step":1,"termination_s":prep.spec.dt,"R_max":np.inf,"s_R_max":np.nan,"I2_final":np.nan,"eta_tail_max":np.nan}
            return {"summary":summary}
        ss=np.array([0.0,0.01,0.02]); perturb=1e-8/prep.spec.N
        I2=1.0+0.01*ss+perturb; R=0.2+0.01*ss+perturb
        tail=np.full_like(ss,1e-10+perturb*1e-3)
        hist={"s":ss,"I2":I2,"R":R,"eta_tail":tail,"RI1_integrated":np.full_like(ss,1e-12),"RE_integrated":np.full_like(ss,1e-12)}
        summary={"runtime_valid":True,"numerical_status":"VALID","termination_reason":None,"termination_step":2,
                 "termination_s":prep.spec.T_final,"R_max":float(R.max()),"s_R_max":0.02,"I2_final":float(I2[-1]),
                 "eta_tail_max":float(tail.max()),"chi_h_max":0.1,"max_abs_RI1_integrated":1e-12,"max_abs_RE_integrated":1e-12}
        return {"summary":summary,"history":hist}
    monkeypatch.setattr(planning,"run_case",fake_run_case)
    report=planning.spatial_convergence(case,[32,64,128],progress=False); rows=report["rows"]
    assert rows[0]["status"]=="NUMERICALLY_INVALID" and rows[0]["R_max"] is None
    assert rows[2]["prev_N"]==64
    accepted=convergence_acceptance(rows,"N",1e-5,1e-3); assert accepted==64
    json.dumps(report,allow_nan=False)


def test_paired_case_stops_early_and_writes_failure_marker(monkeypatch,tmp_path):
    import arterial_spectral_cascade.parent as parent_mod
    spec=CaseSpec("DL",Wo0=10.0,N=32,dt=0.01,T_final=0.03,chi_b=0.02,chi_g=0.02,
                  w=3.0,p=1,R0_over_L0=0.01,slow_variation_limit=0.1,
                  output_every_steps=1,checkpoint_every_steps=1)
    prep=prepare_case(spec)
    original=parent_mod.etdrk4_step
    calls={"n":0}
    def bad_step(ahat,prep,etd,include_nonlinearity=True,include_heterogeneity=True):
        calls["n"]+=1
        if include_heterogeneity:
            out=np.array(ahat,dtype=np.complex128,copy=True); out[0]=np.nan+0j; return out
        return original(ahat,prep,etd,include_nonlinearity,include_heterogeneity)
    monkeypatch.setattr(parent_mod,"etdrk4_step",bad_step)
    paths=storage.init_project_paths(tmp_path)
    result=parent_mod.run_paired_case(prep,paths=paths,resume=False,progress=False)
    summary=result["summary"]
    assert summary["completed"] is False
    assert summary["runtime_valid"] is False
    assert summary["numerical_status"] == "UNSTABLE"
    assert summary["termination_reason"] == "NONFINITE_SPECTRAL_STATE"
    assert summary["termination_step"] == 1
    outdir=storage.case_result_subdir(prep,paths)
    assert (outdir/"PAIRED_FAILED.json").exists()
    assert not (outdir/"PAIRED_COMPLETED.json").exists()
    assert not (outdir/"paired_result.npz").exists()
    json.dumps(summary,allow_nan=False)


def test_explicit_stiffness_screen_matches_solver_design_formula():
    from arterial_spectral_cascade.core import explicit_stiffness_screen, initial_condition
    spec=CaseSpec("P1",Wo0=10.0,N=64,dt=0.002,T_final=0.01,eps_b=0.1,eps_g=0.08,q=1.0,k0=0.5)
    prep=prepare_case(spec)
    ah=np.fft.fft(initial_condition(spec,prep.grid))
    a=np.fft.ifft(ah).real
    expected=spec.dt*(np.max(np.abs(a))*prep.grid.k_ret
                      +np.max(np.abs(prep.b_tilde))*prep.grid.k_ret**3
                      +np.max(np.abs(prep.g_tilde))*prep.grid.k_ret)
    assert abs(explicit_stiffness_screen(ah,prep)-expected) < 1e-14


def test_paired_spatial_convergence_contains_full_solver_design_histories():
    spec=CaseSpec("DL",Wo0=10.0,N=32,dt=0.002,T_final=0.02,k0=0.5,chi_b=0.005,chi_g=0.005,
                  w=3.0,p=1,R0_over_L0=0.01,slow_variation_limit=0.1,
                  output_every_steps=2,checkpoint_every_steps=10)
    report=planning.spatial_convergence(spec,[32,64],progress=False)
    fine=report["rows"][1]
    assert report["paired"] is True
    assert set(("I2","R","DeltaR","D2","eta_tail","RI1_integrated","RE_integrated")).issubset(fine["history"])
    assert fine["epsilon_I2_history"] < 1e-10
    assert fine["integrated_balance_refinement_pass"] is True
    assert fine["chi_h_max"] > 0.0
    assert planning.convergence_acceptance(report["rows"],"N",1e-5,1e-3)==32


def test_strict_json_persistence_rejects_nonfinite_scalars(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        storage.atomic_write_json(tmp_path/"bad.json",{"value":np.nan})


def test_parameter_selection_uses_converged_space_before_time(tmp_path):
    from copy import deepcopy
    from arterial_spectral_cascade.study import STUDY_CONFIG, run_parameter_selection
    cfg=deepcopy(STUDY_CONFIG)
    cfg.update({
        "DISEASE_CASES":({"case_id":"DL-test","case_class":"DL","chi_b":0.005,"chi_g":0.005,"w":3.0,"p":1},),
        "COARSE_WO":(10.0,),"STUDY_N":64,"STUDY_DT":0.002,"STUDY_T_FINAL":0.02,
        "OUTPUT_INTERVAL":0.004,"CHECKPOINT_INTERVAL":0.02,
        "SELECTION_N_VALUES":(64,128),"SELECTION_DT_VALUES":(0.004,0.002),"SELECTION_T_FINAL":0.02,
    })
    report=run_parameter_selection(storage.init_project_paths(tmp_path),cfg,progress=False)
    assert report["pass"] is True
    assert report["recommended_N"] == 64
    assert report["recommended_dt"] == 0.004
    temporal=report["convergence"]["DL"]["temporal"]
    assert temporal["paired"] is True
    assert all(row.get("model_status")=="ADMISSIBLE" for row in temporal["rows"])
    json.dumps(report,allow_nan=False)
