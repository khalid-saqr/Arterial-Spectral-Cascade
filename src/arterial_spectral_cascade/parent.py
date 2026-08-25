from __future__ import annotations

from .core import *
from .core import _jsonable
from .storage import *
from .storage import _record, _history_append, _runtime_state_check


# ----------------------- Parent reference audit ------------------------------
PARENT_WO_SWEEP=(2.0,5.0,10.0,15.0,20.0)

def parent_sweep_specs(N: int=512, dt: float=2e-4, T_final: float=60.0) -> List[CaseSpec]:
    return [
        CaseSpec(
            "P0", Wo0=wo, N=N, dt=dt, T_final=T_final, k0=1.0,
            output_every_steps=max(1,int(round(0.02/dt))),
            checkpoint_every_steps=max(1000,int(round(1.0/dt)))
        )
        for wo in PARENT_WO_SWEEP
    ]


def run_parent_reference_audit(
    paths: Optional[ProjectPaths]=None,
    N: int=512,
    dt: float=2e-4,
    T_final: float=60.0,
    progress: bool=True
) -> Dict[str,Any]:
    """Audit the Solver Design parent baseline without forcing legacy resonance topology.

    Hard status:
      * each Solver Design P0 trajectory completes and remains numerically valid;
      * required scalar diagnostics are finite;
      * I2_final > 0 and R_max >= 0;
      * the constant-coefficient global balance remains dissipative within tolerance.

    Non-controlling diagnostic:
      * whether the Solver Design baseline reproduces the legacy published discrete
        topology with an interior maximum at Wo=15.
    """
    rows=[]
    required_scalar_keys=("R_max","s_R_max","I2_final","G_bal_max","eta_tail_max",
                          "max_abs_RI1_inst","max_abs_RE_inst")

    for spec in parent_sweep_specs(N,dt,T_final):
        res=run_case(prepare_case(spec),paths=paths,resume=True,progress=progress)
        summary=res["summary"]
        finite=bool(all(np.isfinite(float(summary[k])) for k in required_scalar_keys))
        completed=bool(summary.get("completed",False))
        runtime_valid=bool(summary.get("runtime_valid",False))
        I2_positive=bool(float(summary["I2_final"])>0.0)
        R_nonnegative=bool(float(summary["R_max"])>=0.0)
        global_stability=bool(float(summary["G_bal_max"])<=1e-10)
        numerical_valid=bool(completed and runtime_valid and finite and I2_positive and R_nonnegative and global_stability)

        rows.append({
            "Wo":float(spec.Wo0),
            "R_max":float(summary["R_max"]),
            "s_R_max":float(summary["s_R_max"]),
            "I2_final":float(summary["I2_final"]),
            "G_bal_max":float(summary["G_bal_max"]),
            "eta_tail_max":float(summary["eta_tail_max"]),
            "max_abs_RI1_inst":float(summary["max_abs_RI1_inst"]),
            "max_abs_RE_inst":float(summary["max_abs_RE_inst"]),
            "completed":completed,
            "runtime_valid":runtime_valid,
            "finite_diagnostics":finite,
            "I2_positive":I2_positive,
            "R_nonnegative":R_nonnegative,
            "pass_global_stability":global_stability,
            "pass_numerical":numerical_valid,
        })

    R_values=np.asarray([r["R_max"] for r in rows],dtype=float)
    peak_index=int(np.argmax(R_values))
    peak_Wo=float(rows[peak_index]["Wo"])
    pass_global_stability=bool(all(r["pass_global_stability"] for r in rows))
    pass_numerical=bool(all(r["pass_numerical"] for r in rows))

    by_wo={float(r["Wo"]):r for r in rows}
    legacy_topology_match=bool(
        15.0 in by_wo and 10.0 in by_wo and 20.0 in by_wo
        and peak_Wo==15.0
        and by_wo[15.0]["R_max"]>by_wo[10.0]["R_max"]
        and by_wo[15.0]["R_max"]>by_wo[20.0]["R_max"]
    )

    return {
        "reference_schema":PARENT_REFERENCE_SCHEMA,
        "solver_design_parent_baseline":{
            "parameters":{
                "Wo":list(PARENT_WO_SWEEP),
                "N":int(N),
                "dt":float(dt),
                "T_final":float(T_final),
                "k0":1.0,
            },
            "rows":rows,
            "peak_Wo":peak_Wo,
            "pass_global_stability":pass_global_stability,
            "pass_numerical":pass_numerical,
        },
        "legacy_reference_audit":{
            "reported_peak_Wo":15.0,
            "topology_match":legacy_topology_match,
            "status":"MATCH" if legacy_topology_match else "DIFFERENT",
            "acceptance_controlling":False,
            "criterion":"peak_Wo == 15 and R_max(15) > R_max(10), R_max(20)",
        },
        "pass":pass_numerical,
    }




def run_parent_detailed_case(paths: Optional[ProjectPaths]=None, progress: bool=True) -> Dict[str,Any]:
    spec=CaseSpec("P0",Wo0=10,N=512,dt=5e-5,T_final=200.0,k0=.5,output_every_steps=400,checkpoint_every_steps=20000)
    return run_case(prepare_case(spec),paths=paths,resume=True,progress=progress)



# Override with interruption-safe paired disease/MM runner.
def _pair_checkpoint_path(parent: PreparedCase, mm: PreparedCase, paths: ProjectPaths) -> Path:
    d=paths.checkpoints/f"PAIR-{parent.case_id}__{mm.case_id}"; d.mkdir(parents=True,exist_ok=True); return d/"latest.npz"


def _save_pair_checkpoint(parent: PreparedCase, mm: PreparedCase, paths: ProjectPaths, step: int,
                          ahp: np.ndarray, ahm: np.ndarray, hp: Dict[str,List[float]], hm: Dict[str,List[float]], comp: Dict[str,List[float]],
                          peakp: Dict[str,Any], peakm: Dict[str,Any]) -> None:
    path=_pair_checkpoint_path(parent,mm,paths)
    pairhash=stable_hash({"parent":parent.case_id,"mm":mm.case_id,"parent_spec":_jsonable(parent.spec),"psi_D":array_sha256(parent.psi_D),"b":array_sha256(parent.b),"g":array_sha256(parent.g),"k":array_sha256(parent.grid.k),"mask":array_sha256(parent.grid.mask),"ic":array_sha256(initial_condition(parent.spec,parent.grid))},32)
    data={"step":np.array(step,dtype=np.int64),"ahat_parent":ahp,"ahat_mm":ahm,"pair_hash":np.array(pairhash),
          "peakp_R":np.array(float(peakp["R"])),"peakp_step":np.array(int(peakp["step"])),"peakp_ahat":peakp["ahat"],
          "peakm_R":np.array(float(peakm["R"])),"peakm_step":np.array(int(peakm["step"])),"peakm_ahat":peakm["ahat"]}
    for prefix,H in [("hp",hp),("hm",hm),("cmp",comp)]:
        for k,v in H.items(): data[f"{prefix}_{k}"]=np.asarray(v)
    atomic_save_npz(path,**data)


def _load_pair_checkpoint(parent: PreparedCase, mm: PreparedCase, paths: ProjectPaths) -> Optional[Dict[str,Any]]:
    path=_pair_checkpoint_path(parent,mm,paths)
    if not path.exists(): return None
    expected=stable_hash({"parent":parent.case_id,"mm":mm.case_id,"parent_spec":_jsonable(parent.spec),"psi_D":array_sha256(parent.psi_D),"b":array_sha256(parent.b),"g":array_sha256(parent.g),"k":array_sha256(parent.grid.k),"mask":array_sha256(parent.grid.mask),"ic":array_sha256(initial_condition(parent.spec,parent.grid))},32)
    with np.load(path,allow_pickle=False) as z:
        if str(z["pair_hash"].item())!=expected: raise RuntimeError("Paired checkpoint case-metadata mismatch.")
        def hist(prefix): return {k[len(prefix)+1:]:z[k].tolist() for k in z.keys() if k.startswith(prefix+"_")}
        return {"step":int(z["step"]),"ahat_parent":z["ahat_parent"].copy(),"ahat_mm":z["ahat_mm"].copy(),
                "hp":hist("hp"),"hm":hist("hm"),"comp":hist("cmp"),
                "peakp":{"R":float(z["peakp_R"]),"step":int(z["peakp_step"]),"ahat":z["peakp_ahat"].copy()},
                "peakm":{"R":float(z["peakm_R"]),"step":int(z["peakm_step"]),"ahat":z["peakm_ahat"].copy()}}


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
        rp=_record(parent,ahp,0); rm=_record(mm,ahm,0); _history_append(hp,rp); _history_append(hm,rm)
        comp["step"].append(0); comp["s"].append(0.0); comp["DeltaR"].append(rp["R"]-rm["R"]); comp["D2"].append(0.0)
        peakp={"R":rp["R"],"ahat":ahp.copy(),"step":0}; peakm={"R":rm["R"],"ahat":ahm.copy(),"step":0}
    ep=etd_coefficients(parent); em=etd_coefficients(mm)
    iterator=range(start+1,steps+1)
    if progress:
        try:
            from tqdm.auto import tqdm; iterator=tqdm(iterator,desc=f"PAIR {parent.case_id}",leave=False)
        except Exception: pass
    for step in iterator:
        ahp=etdrk4_step(ahp,parent,ep,True,True); ahm=etdrk4_step(ahm,mm,em,True,False)
        if step%sp.output_every_steps==0 or step==steps:
            rp=_record(parent,ahp,step); rm=_record(mm,ahm,step); _history_append(hp,rp); _history_append(hm,rm)
            ap=np.fft.ifft(ahp).real; am=np.fft.ifft(ahm).real
            D=np.sqrt(parent.grid.dx*np.sum((ap-am)**2))/max(np.sqrt(parent.grid.dx*np.sum(am**2)),np.finfo(float).eps)
            comp["step"].append(step); comp["s"].append(step*sp.dt); comp["DeltaR"].append(rp["R"]-rm["R"]); comp["D2"].append(float(D))
            if rp["R"]>peakp["R"]: peakp={"R":rp["R"],"ahat":ahp.copy(),"step":step}
            if rm["R"]>peakm["R"]: peakm={"R":rm["R"],"ahat":ahm.copy(),"step":step}
        if paths is not None and (step%sp.checkpoint_every_steps==0 or step==steps):
            _save_pair_checkpoint(parent,mm,paths,step,ahp,ahm,hp,hm,comp,peakp,peakm)
    HP=finalize_history(hp); HM=finalize_history(hm); C={k:np.asarray(v) for k,v in comp.items()}
    scp=_runtime_state_check(ahp,parent); scm=_runtime_state_check(ahm,mm)
    summary={"case_id":parent.case_id,"matched_mean_case_id":mm.case_id,"completed":True,
             "runtime_valid":bool(scp["pass"] and scm["pass"]),"R_max_het":float(np.max(HP["R"])),"R_max_mm":float(np.max(HM["R"])),
             "Delta_R_maxima":float(np.max(HP["R"])-np.max(HM["R"])),"DeltaR_max_timewise":float(np.max(C["DeltaR"])),"D2_max":float(np.max(C["D2"])),
             "state_check_het":scp,"state_check_mm":scm}
    budget_peak=modal_energy_budget(peakp["ahat"],parent) if parent.spec.mechanism else None
    if paths is not None:
        outdir=case_result_subdir(parent,paths); outdir.mkdir(parents=True,exist_ok=True)
        arrays={"xi":parent.grid.xi,"psi_D_raw":parent.psi_D_raw,"psi_D":parent.psi_D,"b":parent.b,"g":parent.g,
                "ahat_het_peak":peakp["ahat"],"ahat_mm_peak":peakm["ahat"],"ahat_het_final":ahp,"ahat_mm_final":ahm}
        arrays.update({f"het_{k}":v for k,v in HP.items()}); arrays.update({f"mm_{k}":v for k,v in HM.items()}); arrays.update({f"cmp_{k}":v for k,v in C.items()})
        if budget_peak is not None:
            for k,v in budget_peak.items(): arrays[f"budget_peak_{k}"]=v
        atomic_save_npz(outdir/"paired_result.npz",**arrays); atomic_write_json(outdir/"paired_summary.json",summary)
        with np.load(outdir/"paired_result.npz",allow_pickle=False) as z: _=z["ahat_het_final"].shape
        atomic_write_json(outdir/"PAIRED_COMPLETED.json",{"case_id":parent.case_id,"mm_case_id":mm.case_id,"runtime_valid":summary["runtime_valid"],"summary_hash":stable_hash(summary),"archive_sha256":file_sha256(outdir/"paired_result.npz")})
        mm_dir=case_result_subdir(mm,paths)
        atomic_write_json(mm_dir/"PARENT_LINK.json",{"matched_mean_case_id":mm.case_id,"parent_case_id":parent.case_id,"paired_archive":str(outdir/"paired_result.npz")})
    return {"parent":parent,"matched_mean":mm,"het_history":HP,"mm_history":HM,"comparison":C,"summary":summary,
            "ahat_het_peak":peakp["ahat"],"ahat_mm_peak":peakm["ahat"],"budget_peak":budget_peak}




def verify_paired_restart_equivalence() -> Dict[str,Any]:
    """Verify exact restart equivalence for the heterogeneous/matched-mean pair."""
    spec=CaseSpec("DL",Wo0=10,N=48,dt=.002,T_final=.04,k0=.5,chi_b=.01,chi_g=.01,w=2.0,p=1,
                  R0_over_L0=.01,slow_variation_limit=.1,output_every_steps=2,checkpoint_every_steps=5)
    parent_case=prepare_case(spec); mm=make_matched_mean(parent_case)
    ep=etd_coefficients(parent_case); em=etd_coefficients(mm)
    ah0=np.fft.fft(initial_condition(spec,parent_case.grid)); steps=int(round(spec.T_final/spec.dt)); mid=steps//2
    up=ah0.copy(); um=ah0.copy()
    for _ in range(steps):
        up=etdrk4_step(up,parent_case,ep,True,True); um=etdrk4_step(um,mm,em,True,False)
    rp=ah0.copy(); rm=ah0.copy(); hp={}; hm={}; comp={"step":[],"s":[],"DeltaR":[],"D2":[]}
    peakp={"R":-np.inf,"step":0,"ahat":rp.copy()}; peakm={"R":-np.inf,"step":0,"ahat":rm.copy()}
    for _ in range(1,mid+1):
        rp=etdrk4_step(rp,parent_case,ep,True,True); rm=etdrk4_step(rm,mm,em,True,False)
    with tempfile.TemporaryDirectory() as td:
        paths=init_project_paths(td)
        _save_pair_checkpoint(parent_case,mm,paths,mid,rp,rm,hp,hm,comp,peakp,peakm)
        cp=_load_pair_checkpoint(parent_case,mm,paths)
        rp=cp["ahat_parent"]; rm=cp["ahat_mm"]
    for _ in range(mid,steps):
        rp=etdrk4_step(rp,parent_case,ep,True,True); rm=etdrk4_step(rm,mm,em,True,False)
    ep_err=float(np.linalg.norm(rp-up)/max(np.linalg.norm(up),1e-30))
    em_err=float(np.linalg.norm(rm-um)/max(np.linalg.norm(um),1e-30))
    return {"name":"paired_restart_equivalence","relative_error_heterogeneous":ep_err,
            "relative_error_matched_mean":em_err,"pass":max(ep_err,em_err)<1e-15}


def full_verification_suite() -> Dict[str,Any]:
    """Run the complete independent numerical verification suite."""
    report=core_verification_suite(True)
    report["tests"].append(verify_restart_equivalence())
    report["tests"].append(verify_paired_restart_equivalence())
    report["pass"]=all(t["pass"] for t in report["tests"])
    return report

__all__ = [name for name in globals() if not name.startswith("_")]
