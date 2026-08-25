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
        else:
            summary={"runtime_valid":True,"numerical_status":"VALID","termination_reason":None,
                     "termination_step":int(round(prep.spec.T_final/dt)),"termination_s":prep.spec.T_final,
                     "R_max":2.0+1e-6*dt,"s_R_max":0.1,"I2_final":1.0+1e-8*dt}
        return {"summary":summary}
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
        else:
            summary={"runtime_valid":True,"numerical_status":"VALID","termination_reason":None,"termination_step":2,
                     "termination_s":prep.spec.T_final,"R_max":3.0+1e-9/prep.spec.N,"s_R_max":0.01,
                     "I2_final":1.0+1e-10/prep.spec.N,"eta_tail_max":1e-8}
        return {"summary":summary}
    monkeypatch.setattr(planning,"run_case",fake_run_case)
    report=planning.spatial_convergence(case,[32,64,128],progress=False); rows=report["rows"]
    assert rows[0]["status"]=="NUMERICALLY_INVALID" and rows[0]["R_max"] is None
    assert rows[2]["prev_N"]==64
    accepted=convergence_acceptance(rows,"N",1e-5,1e-3); assert accepted==64
    json.dumps(report,allow_nan=False)
