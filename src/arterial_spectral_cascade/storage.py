from __future__ import annotations

from .core import *
from .core import _jsonable
from .storage_base import *
from .storage_base import (_finite_float_or_none, _runtime_state_check, _record_is_finite,
                           _history_append, finalize_history, load_checkpoint, _integrity_payload)

def save_checkpoint(prep: PreparedCase, paths: ProjectPaths, step: int, ahat: np.ndarray,
                    history: Dict[str,List[float]], peak: Dict[str,Any]) -> Path:
    if not np.all(np.isfinite(ahat)):
        raise ValueError(f"Refusing to checkpoint non-finite spectral state for {prep.case_id} at step {step}.")
    peak_ahat=np.asarray(peak.get("ahat",ahat))
    if not np.all(np.isfinite(peak_ahat)):
        raise ValueError(f"Refusing to checkpoint non-finite peak state for {prep.case_id} at step {step}.")
    statecheck=_runtime_state_check(ahat,prep)
    if not statecheck["pass"]:
        raise ValueError(f"Refusing to checkpoint numerically invalid state for {prep.case_id} at step {step}.")
    cdir=paths.checkpoints/prep.case_id; cdir.mkdir(parents=True,exist_ok=True)
    path=cdir/"latest.npz"
    integrity=stable_hash(_integrity_payload(prep),32)
    payload={"step":np.array(step,dtype=np.int64),"ahat":ahat,
             "case_metadata_hash":np.array(stable_hash({"case_id":prep.case_id,"spec":_jsonable(prep.spec)})),
             "integrity_hash":np.array(integrity),"peak_R":np.array(float(peak.get("R",-np.inf))),
             "peak_step":np.array(int(peak.get("step",0)),dtype=np.int64),"peak_ahat":peak_ahat}
    for k,v in history.items(): payload[f"hist_{k}"]=np.asarray(v)
    atomic_save_npz(path,**payload); return path


def _record(prep: PreparedCase, ahat: np.ndarray, step: int) -> Dict[str,float]:
    rhs=full_rhs_hat(ahat,prep)
    bal=integrals_and_balance(ahat,prep,rhs); sp=spectral_broadening(ahat,prep)
    chi_h=explicit_stiffness_screen(ahat,prep)
    return {"step":int(step),"s":float(step*prep.spec.dt),**bal,**sp,"chi_h":float(chi_h)}


def run_case(prep: PreparedCase, paths: Optional[ProjectPaths]=None, resume: bool=True, progress: bool=False,
             require_admissible: bool=True, include_nonlinearity: bool=True, include_heterogeneity: bool=True) -> Dict[str,Any]:
    if require_admissible and prep.admissibility["status"]!="ADMISSIBLE":
        raise RuntimeError(f"Case {prep.case_id} is not admissible: {prep.admissibility['status']}")
    spec=prep.spec; steps=int(round(spec.T_final/spec.dt))
    if abs(steps*spec.dt-spec.T_final)>1e-10*max(1.0,spec.T_final):
        raise ValueError("T_final/dt must be an integer for deterministic fixed-step integration.")
    if paths is not None: save_case_metadata(prep,paths)
    cp=load_checkpoint(prep,paths) if (paths is not None and resume) else None
    if cp:
        start=cp["step"]; ah=cp["ahat"]; history=cp["history"]; peak=cp["peak"]
    else:
        start=0; ah=project_hat(np.fft.fft(initial_condition(spec,prep.grid)),prep.grid); history={}; rec=_record(prep,ah,0)
        if not _record_is_finite(rec): raise RuntimeError(f"Initial diagnostic record is non-finite for {prep.case_id}.")
        _history_append(history,rec); peak={"R":rec["R"],"step":0,"ahat":ah.copy()}
    initial_ahat=project_hat(np.fft.fft(initial_condition(spec,prep.grid)),prep.grid)
    etd=etd_coefficients(prep)
    iterator=range(start+1,steps+1)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator=tqdm(iterator,desc=prep.case_id,leave=False)
        except Exception: pass

    failure=None
    for step in iterator:
        ah=etdrk4_step(ah,prep,etd,include_nonlinearity,include_heterogeneity)
        if not np.all(np.isfinite(ah)):
            failure={"reason":"NONFINITE_SPECTRAL_STATE","step":int(step),"s":float(step*spec.dt)}; break
        output_now=(step%spec.output_every_steps==0 or step==steps)
        checkpoint_now=(paths is not None and (step%spec.checkpoint_every_steps==0 or step==steps))
        if output_now or checkpoint_now:
            state_now=_runtime_state_check(ah,prep)
            if not state_now["pass"]:
                failure={"reason":"RUNTIME_STATE_CHECK_FAILED","step":int(step),"s":float(step*spec.dt),"state_check":state_now}; break
        if output_now:
            rec=_record(prep,ah,step)
            if not _record_is_finite(rec):
                failure={"reason":"NONFINITE_DIAGNOSTIC","step":int(step),"s":float(step*spec.dt)}; break
            _history_append(history,rec)
            if rec["R"]>peak["R"]: peak={"R":rec["R"],"step":step,"ahat":ah.copy()}
        if checkpoint_now:
            save_checkpoint(prep,paths,step,ah,history,peak)

    hist=finalize_history(history); statecheck=_runtime_state_check(ah,prep)
    reached_final=(failure is None); runtime_valid=bool(reached_final and statecheck["pass"])
    if runtime_valid:
        numerical_status="VALID"; termination_reason=None; termination_step=int(steps); termination_s=float(spec.T_final)
        imax=int(np.argmax(hist["R"])); R_max=float(hist["R"][imax]); s_R_max=float(hist["s"][imax]); I2_final=float(hist["I2"][-1])
        G_bal_max=float(np.max(hist["G_bal"])); eta_tail_max=float(np.max(hist["eta_tail"])); chi_h_max=float(np.max(hist["chi_h"]))
        max_abs_RI1_inst=float(np.max(np.abs(hist["R_I1_inst"]))); max_abs_RE_inst=float(np.max(np.abs(hist["R_E_inst"])))
        max_abs_RI1_integrated=float(np.max(np.abs(hist.get("RI1_integrated",np.array([0.0])))))
        max_abs_RE_integrated=float(np.max(np.abs(hist.get("RE_integrated",np.array([0.0])))))
        max_abs_G_balance_vs_fd=float(np.max(np.abs(hist["G_bal"]-hist.get("G_fd",hist["G_bal"]))))
    else:
        reason=(failure or {}).get("reason","FINAL_RUNTIME_STATE_INVALID")
        numerical_status="UNSTABLE" if reason in {"NONFINITE_SPECTRAL_STATE","NONFINITE_DIAGNOSTIC"} else "INVALID"
        termination_reason=reason; termination_step=int((failure or {}).get("step",steps)); termination_s=float((failure or {}).get("s",termination_step*spec.dt))
        R_max=s_R_max=I2_final=G_bal_max=eta_tail_max=chi_h_max=max_abs_RI1_inst=max_abs_RE_inst=max_abs_RI1_integrated=max_abs_RE_integrated=max_abs_G_balance_vs_fd=None
    summary={"case_id":prep.case_id,"parent_case_id":prep.parent_case_id,"completed":bool(reached_final),"runtime_valid":runtime_valid,
             "numerical_status":numerical_status,"termination_reason":termination_reason,"termination_step":termination_step,"termination_s":termination_s,
             "R_max":R_max,"s_R_max":s_R_max,"I2_final":I2_final,"G_bal_max":G_bal_max,"eta_tail_max":eta_tail_max,
             "chi_h_max":chi_h_max,"max_abs_RI1_inst":max_abs_RI1_inst,"max_abs_RE_inst":max_abs_RE_inst,
             "max_abs_RI1_integrated":max_abs_RI1_integrated,"max_abs_RE_integrated":max_abs_RE_integrated,
             "max_abs_G_balance_vs_fd":max_abs_G_balance_vs_fd,
             "admissibility":prep.admissibility,"state_check":statecheck}
    budget_peak=modal_energy_budget(peak["ahat"],prep) if (spec.mechanism and runtime_valid) else None
    if paths is not None:
        outdir=case_result_subdir(prep,paths); atomic_write_json(outdir/"summary.json",summary)
        if runtime_valid:
            arrays={"xi":prep.grid.xi,"psi_D_raw":prep.psi_D_raw,"psi_D":prep.psi_D,"b":prep.b,"g":prep.g,
                    "bhat_norm":np.fft.fft(prep.b)/prep.grid.N,"ghat_norm":np.fft.fft(prep.g)/prep.grid.N,
                    "ahat_initial":initial_ahat,"ahat_peak":peak["ahat"],"ahat_final":ah}
            arrays.update({f"hist_{k}":v for k,v in hist.items()})
            if budget_peak is not None:
                for k,v in budget_peak.items(): arrays[f"budget_peak_{k}"]=v
            atomic_save_npz(outdir/"result.npz",**arrays)
            with np.load(outdir/"result.npz",allow_pickle=False) as z: _=z["ahat_final"].shape
            failed_marker=outdir/"FAILED.json"
            if failed_marker.exists(): failed_marker.unlink()
            atomic_write_json(outdir/"COMPLETED.json",{"case_id":prep.case_id,"runtime_valid":True,"summary_hash":stable_hash(summary),"archive_sha256":file_sha256(outdir/"result.npz")})
        else:
            completed_marker=outdir/"COMPLETED.json"
            if completed_marker.exists(): completed_marker.unlink()
            atomic_write_json(outdir/"FAILED.json",{"case_id":prep.case_id,"runtime_valid":False,"numerical_status":numerical_status,
                                                    "termination_reason":termination_reason,"termination_step":termination_step,"termination_s":termination_s,
                                                    "summary_hash":stable_hash(summary)})
    return {"prep":prep,"history":hist,"summary":summary,"ahat_final":ah,"ahat_peak":peak["ahat"],"budget_peak":budget_peak}


def verify_restart_equivalence() -> Dict[str,Any]:
    """Compare uninterrupted and checkpoint/restart state and saved diagnostics."""
    spec=CaseSpec("P1",Wo0=5,N=64,dt=.002,T_final=.08,eps_b=.12,eps_g=.1,q=1.0,k0=.5,
                  output_every_steps=2,checkpoint_every_steps=10)
    prep=prepare_case(spec); uninterrupted=run_case(prep,paths=None,resume=False,progress=False)
    steps=int(round(spec.T_final/spec.dt)); mid=steps//2
    ah=project_hat(np.fft.fft(initial_condition(spec,prep.grid)),prep.grid); etd=etd_coefficients(prep)
    history={}; rec=_record(prep,ah,0); _history_append(history,rec); peak={"R":rec["R"],"step":0,"ahat":ah.copy()}
    for step in range(1,mid+1):
        ah=etdrk4_step(ah,prep,etd)
        if step%spec.output_every_steps==0 or step==mid:
            rec=_record(prep,ah,step); _history_append(history,rec)
            if rec["R"]>peak["R"]: peak={"R":rec["R"],"step":step,"ahat":ah.copy()}
    with tempfile.TemporaryDirectory() as td:
        paths=init_project_paths(td); save_checkpoint(prep,paths,mid,ah,history,peak)
        resumed=run_case(prep,paths=paths,resume=True,progress=False)
    state_err=float(np.linalg.norm(resumed["ahat_final"]-uninterrupted["ahat_final"])/max(np.linalg.norm(uninterrupted["ahat_final"]),1e-30))
    keys=("I1","I2","E","B1","BE","R","eta_tail","chi_h","RI1_integrated","RE_integrated")
    diagnostic_error=0.0
    for key in keys:
        a=np.asarray(uninterrupted["history"][key]); b=np.asarray(resumed["history"][key])
        diagnostic_error=max(diagnostic_error,float(np.max(np.abs(a-b))))
    return {"name":"restart_equivalence","relative_state_error":state_err,"max_diagnostic_error":diagnostic_error,
            "pass":bool(state_err<1e-15 and diagnostic_error<1e-14)}


__all__ = [name for name in globals() if not name.startswith("_")]
