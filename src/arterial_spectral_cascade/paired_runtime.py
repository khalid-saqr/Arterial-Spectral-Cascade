from __future__ import annotations

from .core import *
from .core import _jsonable
from .storage import *
from .storage import _record, _history_append, _runtime_state_check, _record_is_finite

def _pair_checkpoint_path(parent: PreparedCase, mm: PreparedCase, paths: ProjectPaths) -> Path:
    d=paths.checkpoints/f"PAIR-{parent.case_id}__{mm.case_id}"; d.mkdir(parents=True,exist_ok=True); return d/"latest.npz"


def _pair_integrity_payload(parent: PreparedCase, mm: PreparedCase) -> Dict[str,Any]:
    return {
        "parent":parent.case_id,"mm":mm.case_id,"parent_spec":_jsonable(parent.spec),
        "psi_D":array_sha256(parent.psi_D),"b":array_sha256(parent.b),"g":array_sha256(parent.g),
        "mm_b":array_sha256(mm.b),"mm_g":array_sha256(mm.g),
        "k":array_sha256(parent.grid.k),"mask":array_sha256(parent.grid.mask),
        "ic":array_sha256(initial_condition(parent.spec,parent.grid)),
    }


def _save_pair_checkpoint(parent: PreparedCase, mm: PreparedCase, paths: ProjectPaths, step: int,
                          ahp: np.ndarray, ahm: np.ndarray, hp: Dict[str,List[float]], hm: Dict[str,List[float]], comp: Dict[str,List[float]],
                          peakp: Dict[str,Any], peakm: Dict[str,Any]) -> None:
    states=(ahp,ahm,np.asarray(peakp.get("ahat",ahp)),np.asarray(peakm.get("ahat",ahm)))
    if not all(np.all(np.isfinite(v)) for v in states):
        raise ValueError(f"Refusing to checkpoint non-finite paired state at step {step}.")
    scp=_runtime_state_check(ahp,parent); scm=_runtime_state_check(ahm,mm)
    if not scp["pass"] or not scm["pass"]:
        raise ValueError(f"Refusing to checkpoint numerically invalid paired state at step {step}.")
    path=_pair_checkpoint_path(parent,mm,paths)
    pairhash=stable_hash(_pair_integrity_payload(parent,mm),32)
    data={"step":np.array(step,dtype=np.int64),"ahat_parent":ahp,"ahat_mm":ahm,"pair_hash":np.array(pairhash),
          "peakp_R":np.array(float(peakp["R"])),"peakp_step":np.array(int(peakp["step"])),"peakp_ahat":peakp["ahat"],
          "peakm_R":np.array(float(peakm["R"])),"peakm_step":np.array(int(peakm["step"])),"peakm_ahat":peakm["ahat"]}
    for prefix,H in [("hp",hp),("hm",hm),("cmp",comp)]:
        for k,v in H.items(): data[f"{prefix}_{k}"]=np.asarray(v)
    atomic_save_npz(path,**data)


def _load_pair_checkpoint(parent: PreparedCase, mm: PreparedCase, paths: ProjectPaths) -> Optional[Dict[str,Any]]:
    path=_pair_checkpoint_path(parent,mm,paths)
    if not path.exists(): return None
    expected=stable_hash(_pair_integrity_payload(parent,mm),32)
    with np.load(path,allow_pickle=False) as z:
        if str(z["pair_hash"].item())!=expected: raise RuntimeError("Paired checkpoint case-metadata mismatch.")
        ahp=z["ahat_parent"].copy(); ahm=z["ahat_mm"].copy()
        pap=z["peakp_ahat"].copy(); pam=z["peakm_ahat"].copy()
        if not all(np.all(np.isfinite(v)) for v in (ahp,ahm,pap,pam)):
            raise RuntimeError("Paired checkpoint contains a non-finite spectral state.")
        if not _runtime_state_check(ahp,parent)["pass"] or not _runtime_state_check(ahm,mm)["pass"]:
            raise RuntimeError("Paired checkpoint fails runtime state validity checks.")
        def hist(prefix): return {k[len(prefix)+1:]:z[k].tolist() for k in z.keys() if k.startswith(prefix+"_")}
        return {"step":int(z["step"]),"ahat_parent":ahp,"ahat_mm":ahm,
                "hp":hist("hp"),"hm":hist("hm"),"comp":hist("cmp"),
                "peakp":{"R":float(z["peakp_R"]),"step":int(z["peakp_step"]),"ahat":pap},
                "peakm":{"R":float(z["peakm_R"]),"step":int(z["peakm_step"]),"ahat":pam}}


def _pair_failure(reason: str, step: int, spec: CaseSpec, **extra) -> Dict[str,Any]:
    out={"reason":reason,"step":int(step),"s":float(step*spec.dt)}
    out.update(extra); return out


def run_paired_case(parent: PreparedCase, paths: Optional[ProjectPaths]=None, resume: bool=True, progress: bool=False) -> Dict[str,Any]:
    if parent.spec.case_class not in {"P1","DL","DM","DR"}: raise ValueError("Paired runs require a heterogeneous P1/DL/DM/DR parent case.")
    if parent.admissibility["status"]!="ADMISSIBLE": raise RuntimeError(f"Parent heterogeneous case is not admissible: {parent.admissibility['status']}")
    mm=make_matched_mean(parent); sp=parent.spec; steps=int(round(sp.T_final/sp.dt))
    if abs(steps*sp.dt-sp.T_final)>1e-10*max(1.0,sp.T_final): raise ValueError("T_final/dt must be an integer.")
    if paths is not None: save_case_metadata(parent,paths); save_case_metadata(mm,paths)
    cp=_load_pair_checkpoint(parent,mm,paths) if (paths is not None and resume) else None
    if cp:
        start=cp["step"]; ahp=cp["ahat_parent"]; ahm=cp["ahat_mm"]; hp=cp["hp"]; hm=cp["hm"]; comp=cp["comp"]; peakp=cp["peakp"]; peakm=cp["peakm"]
    else:
        start=0; ahp=project_hat(np.fft.fft(initial_condition(sp,parent.grid)),parent.grid); ahm=ahp.copy(); hp={}; hm={}; comp={"step":[],"s":[],"DeltaR":[],"D2":[]}
        rp=_record(parent,ahp,0); rm=_record(mm,ahm,0)
        if not _record_is_finite(rp) or not _record_is_finite(rm): raise RuntimeError("Initial paired diagnostic record is non-finite.")
        _history_append(hp,rp); _history_append(hm,rm)
        comp["step"].append(0); comp["s"].append(0.0); comp["DeltaR"].append(rp["R"]-rm["R"]); comp["D2"].append(0.0)
        peakp={"R":rp["R"],"ahat":ahp.copy(),"step":0}; peakm={"R":rm["R"],"ahat":ahm.copy(),"step":0}
    ep=etd_coefficients(parent); em=etd_coefficients(mm)
    iterator=range(start+1,steps+1)
    if progress:
        try:
            from tqdm.auto import tqdm; iterator=tqdm(iterator,desc=f"PAIR {parent.case_id}",leave=False)
        except Exception: pass
    failure=None
    for step in iterator:
        ahp=etdrk4_step(ahp,parent,ep,True,True); ahm=etdrk4_step(ahm,mm,em,True,False)
        if not np.all(np.isfinite(ahp)) or not np.all(np.isfinite(ahm)):
            failure=_pair_failure("NONFINITE_SPECTRAL_STATE",step,sp); break
        output_now=(step%sp.output_every_steps==0 or step==steps)
        checkpoint_now=(paths is not None and (step%sp.checkpoint_every_steps==0 or step==steps))
        if output_now or checkpoint_now:
            scp=_runtime_state_check(ahp,parent); scm=_runtime_state_check(ahm,mm)
            if not scp["pass"] or not scm["pass"]:
                failure=_pair_failure("RUNTIME_STATE_CHECK_FAILED",step,sp,state_check_het=scp,state_check_mm=scm); break
        if output_now:
            rp=_record(parent,ahp,step); rm=_record(mm,ahm,step)
            if not _record_is_finite(rp) or not _record_is_finite(rm):
                failure=_pair_failure("NONFINITE_DIAGNOSTIC",step,sp); break
            ap=np.fft.ifft(ahp).real; am=np.fft.ifft(ahm).real
            D=np.sqrt(parent.grid.dx*np.sum((ap-am)**2))/max(np.sqrt(parent.grid.dx*np.sum(am**2)),np.finfo(float).eps)
            delta=rp["R"]-rm["R"]
            if not np.isfinite(D) or not np.isfinite(delta):
                failure=_pair_failure("NONFINITE_COMPARISON_DIAGNOSTIC",step,sp); break
            _history_append(hp,rp); _history_append(hm,rm)
            comp["step"].append(step); comp["s"].append(step*sp.dt); comp["DeltaR"].append(float(delta)); comp["D2"].append(float(D))
            if rp["R"]>peakp["R"]: peakp={"R":rp["R"],"ahat":ahp.copy(),"step":step}
            if rm["R"]>peakm["R"]: peakm={"R":rm["R"],"ahat":ahm.copy(),"step":step}
        if checkpoint_now:
            _save_pair_checkpoint(parent,mm,paths,step,ahp,ahm,hp,hm,comp,peakp,peakm)

    HP=finalize_history(hp); HM=finalize_history(hm); C={k:np.asarray(v) for k,v in comp.items()}
    scp=_runtime_state_check(ahp,parent); scm=_runtime_state_check(ahm,mm)
    reached_final=(failure is None); runtime_valid=bool(reached_final and scp["pass"] and scm["pass"])
    if runtime_valid:
        numerical_status="VALID"; termination_reason=None; termination_step=steps; termination_s=float(sp.T_final)
        R_max_het=float(np.max(HP["R"])); R_max_mm=float(np.max(HM["R"]))
        Delta_R_maxima=float(R_max_het-R_max_mm); DeltaR_max_timewise=float(np.max(C["DeltaR"])); D2_max=float(np.max(C["D2"]))
        eta_tail_max_het=float(np.max(HP["eta_tail"])); chi_h_max_het=float(np.max(HP["chi_h"])); chi_h_max_mm=float(np.max(HM["chi_h"]))
        max_abs_RI1_integrated_het=float(np.max(np.abs(HP.get("RI1_integrated",np.array([0.0])))))
        max_abs_RE_integrated_het=float(np.max(np.abs(HP.get("RE_integrated",np.array([0.0])))))
        max_abs_G_balance_vs_fd_het=float(np.max(np.abs(HP["G_bal"]-HP.get("G_fd",HP["G_bal"]))))
    else:
        reason=(failure or {}).get("reason","FINAL_RUNTIME_STATE_INVALID")
        numerical_status="UNSTABLE" if reason in {"NONFINITE_SPECTRAL_STATE","NONFINITE_DIAGNOSTIC","NONFINITE_COMPARISON_DIAGNOSTIC"} else "INVALID"
        termination_reason=reason; termination_step=int((failure or {}).get("step",steps)); termination_s=float((failure or {}).get("s",termination_step*sp.dt))
        R_max_het=R_max_mm=Delta_R_maxima=DeltaR_max_timewise=D2_max=eta_tail_max_het=chi_h_max_het=chi_h_max_mm=max_abs_RI1_integrated_het=max_abs_RE_integrated_het=max_abs_G_balance_vs_fd_het=None
    summary={"case_id":parent.case_id,"matched_mean_case_id":mm.case_id,"completed":bool(reached_final),"runtime_valid":runtime_valid,
             "numerical_status":numerical_status,"termination_reason":termination_reason,"termination_step":int(termination_step),"termination_s":termination_s,
             "R_max_het":R_max_het,"R_max_mm":R_max_mm,"Delta_R_maxima":Delta_R_maxima,
             "DeltaR_max_timewise":DeltaR_max_timewise,"D2_max":D2_max,"eta_tail_max_het":eta_tail_max_het,
             "chi_h_max_het":chi_h_max_het,"chi_h_max_mm":chi_h_max_mm,
             "max_abs_RI1_integrated_het":max_abs_RI1_integrated_het,"max_abs_RE_integrated_het":max_abs_RE_integrated_het,
             "max_abs_G_balance_vs_fd_het":max_abs_G_balance_vs_fd_het,
             "state_check_het":scp,"state_check_mm":scm}
    budget_peak=modal_energy_budget(peakp["ahat"],parent) if (parent.spec.mechanism and runtime_valid) else None
    if paths is not None:
        outdir=case_result_subdir(parent,paths); outdir.mkdir(parents=True,exist_ok=True)
        atomic_write_json(outdir/"paired_summary.json",summary)
        if runtime_valid:
            arrays={"xi":parent.grid.xi,"psi_D_raw":parent.psi_D_raw,"psi_D":parent.psi_D,"b":parent.b,"g":parent.g,
                    "ahat_het_peak":peakp["ahat"],"ahat_mm_peak":peakm["ahat"],"ahat_het_final":ahp,"ahat_mm_final":ahm}
            arrays.update({f"het_{k}":v for k,v in HP.items()}); arrays.update({f"mm_{k}":v for k,v in HM.items()}); arrays.update({f"cmp_{k}":v for k,v in C.items()})
            if budget_peak is not None:
                for k,v in budget_peak.items(): arrays[f"budget_peak_{k}"]=v
            atomic_save_npz(outdir/"paired_result.npz",**arrays)
            with np.load(outdir/"paired_result.npz",allow_pickle=False) as z: _=z["ahat_het_final"].shape
            failed_marker=outdir/"PAIRED_FAILED.json"
            if failed_marker.exists(): failed_marker.unlink()
            atomic_write_json(outdir/"PAIRED_COMPLETED.json",{"case_id":parent.case_id,"mm_case_id":mm.case_id,"runtime_valid":True,
                              "summary_hash":stable_hash(summary),"archive_sha256":file_sha256(outdir/"paired_result.npz")})
            mm_dir=case_result_subdir(mm,paths)
            atomic_write_json(mm_dir/"PARENT_LINK.json",{"matched_mean_case_id":mm.case_id,"parent_case_id":parent.case_id,"paired_archive":str(outdir/"paired_result.npz")})
        else:
            completed_marker=outdir/"PAIRED_COMPLETED.json"
            if completed_marker.exists(): completed_marker.unlink()
            atomic_write_json(outdir/"PAIRED_FAILED.json",{"case_id":parent.case_id,"mm_case_id":mm.case_id,"runtime_valid":False,
                              "numerical_status":numerical_status,"termination_reason":termination_reason,
                              "termination_step":int(termination_step),"termination_s":termination_s,"summary_hash":stable_hash(summary)})
    return {"parent":parent,"matched_mean":mm,"het_history":HP,"mm_history":HM,"comparison":C,"summary":summary,
            "ahat_het_peak":peakp["ahat"],"ahat_mm_peak":peakm["ahat"],"ahat_het_final":ahp,"ahat_mm_final":ahm,
            "budget_peak":budget_peak}


def verify_paired_restart_equivalence() -> Dict[str,Any]:
    """Verify paired restart equivalence for states and saved comparison diagnostics."""
    spec=CaseSpec("DL",Wo0=10,N=48,dt=.002,T_final=.04,k0=.5,chi_b=.01,chi_g=.01,w=3.0,p=1,
                  R0_over_L0=.01,slow_variation_limit=.1,output_every_steps=2,checkpoint_every_steps=5)
    parent_case=prepare_case(spec); mm=make_matched_mean(parent_case)
    uninterrupted=run_paired_case(parent_case,paths=None,resume=False,progress=False)
    ep=etd_coefficients(parent_case); em=etd_coefficients(mm)
    ahp=project_hat(np.fft.fft(initial_condition(spec,parent_case.grid)),parent_case.grid); ahm=ahp.copy()
    steps=int(round(spec.T_final/spec.dt)); mid=steps//2
    hp={}; hm={}; comp={"step":[],"s":[],"DeltaR":[],"D2":[]}
    rp=_record(parent_case,ahp,0); rm=_record(mm,ahm,0); _history_append(hp,rp); _history_append(hm,rm)
    comp["step"].append(0); comp["s"].append(0.0); comp["DeltaR"].append(rp["R"]-rm["R"]); comp["D2"].append(0.0)
    peakp={"R":rp["R"],"step":0,"ahat":ahp.copy()}; peakm={"R":rm["R"],"step":0,"ahat":ahm.copy()}
    for step in range(1,mid+1):
        ahp=etdrk4_step(ahp,parent_case,ep,True,True); ahm=etdrk4_step(ahm,mm,em,True,False)
        if step%spec.output_every_steps==0 or step==mid:
            rp=_record(parent_case,ahp,step); rm=_record(mm,ahm,step); _history_append(hp,rp); _history_append(hm,rm)
            ap=np.fft.ifft(ahp).real; am=np.fft.ifft(ahm).real
            D=np.sqrt(parent_case.grid.dx*np.sum((ap-am)**2))/max(np.sqrt(parent_case.grid.dx*np.sum(am**2)),np.finfo(float).eps)
            comp["step"].append(step); comp["s"].append(step*spec.dt); comp["DeltaR"].append(rp["R"]-rm["R"]); comp["D2"].append(float(D))
            if rp["R"]>peakp["R"]: peakp={"R":rp["R"],"step":step,"ahat":ahp.copy()}
            if rm["R"]>peakm["R"]: peakm={"R":rm["R"],"step":step,"ahat":ahm.copy()}
    with tempfile.TemporaryDirectory() as td:
        paths=init_project_paths(td); _save_pair_checkpoint(parent_case,mm,paths,mid,ahp,ahm,hp,hm,comp,peakp,peakm)
        resumed=run_paired_case(parent_case,paths=paths,resume=True,progress=False)
    ep_err=float(np.linalg.norm(resumed["ahat_het_final"]-uninterrupted["ahat_het_final"])/max(np.linalg.norm(uninterrupted["ahat_het_final"]),1e-30))
    em_err=float(np.linalg.norm(resumed["ahat_mm_final"]-uninterrupted["ahat_mm_final"])/max(np.linalg.norm(uninterrupted["ahat_mm_final"]),1e-30))
    diagnostic_error=0.0
    for key in ("I2","R","chi_h","RI1_integrated","RE_integrated"):
        diagnostic_error=max(diagnostic_error,float(np.max(np.abs(resumed["het_history"][key]-uninterrupted["het_history"][key]))))
    for key in ("DeltaR","D2"):
        diagnostic_error=max(diagnostic_error,float(np.max(np.abs(resumed["comparison"][key]-uninterrupted["comparison"][key]))))
    return {"name":"paired_restart_equivalence","relative_error_heterogeneous":ep_err,
            "relative_error_matched_mean":em_err,"max_diagnostic_error":diagnostic_error,
            "pass":bool(max(ep_err,em_err)<1e-15 and diagnostic_error<1e-14)}

__all__ = [name for name in globals() if not name.startswith("_")]
