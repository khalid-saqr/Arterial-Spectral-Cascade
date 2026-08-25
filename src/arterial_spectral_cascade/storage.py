from __future__ import annotations

from .core import *
from .core import _jsonable

# ----------------------------- Persistence ---------------------------------
@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    metadata: Path
    checkpoints: Path
    verification: Path
    results: Path
    tables: Path
    figures: Path
    logs: Path


def init_project_paths(root: os.PathLike) -> ProjectPaths:
    root=Path(root)
    dirs={name:root/name for name in ["metadata","checkpoints","verification","results","tables","figures","logs"]}
    for p in dirs.values(): p.mkdir(parents=True,exist_ok=True)
    for sub in ["parent","localized","multiple","distributed","matched_mean","mechanism","optional"]:
        (dirs["results"]/sub).mkdir(parents=True,exist_ok=True)
    return ProjectPaths(root=root,**dirs)


def atomic_write_json(path: os.PathLike, payload: Any) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(_jsonable(payload),indent=2,sort_keys=True,allow_nan=False))
        json.loads(Path(tmp).read_text())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.remove(tmp)


def atomic_save_npz(path: os.PathLike, **arrays) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent); os.close(fd)
    try:
        with open(tmp,"wb") as f: np.savez_compressed(f,**arrays)
        with np.load(tmp,allow_pickle=False) as z:
            _=list(z.keys())
            for key in z.keys(): _=z[key].shape
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.remove(tmp)


def case_result_subdir(prep: PreparedCase, paths: ProjectPaths) -> Path:
    c=prep.spec.case_class
    if prep.parent_case_id is not None or c=="MM": base=paths.results/"matched_mean"
    elif c in {"H0","P0","P1"}: base=paths.results/"parent"
    elif c=="DL": base=paths.results/"localized"
    elif c=="DM": base=paths.results/"multiple"
    elif c=="DR": base=paths.results/"distributed"
    else: base=paths.results/"optional"
    p=base/prep.case_id; p.mkdir(parents=True,exist_ok=True); return p


def _integrity_payload(prep: PreparedCase) -> Dict[str,Any]:
    return {
        "case_id":prep.case_id,
        "spec":_jsonable(prep.spec),
        "psi_D":array_sha256(prep.psi_D),
        "b":array_sha256(prep.b),"g":array_sha256(prep.g),
        "k":array_sha256(prep.grid.k),"mask":array_sha256(prep.grid.mask),
        "ic":array_sha256(initial_condition(prep.spec,prep.grid)),
    }


def save_case_metadata(prep: PreparedCase, paths: ProjectPaths) -> Path:
    a0=initial_condition(prep.spec,prep.grid)
    payload={
        "case_id":prep.case_id,"parent_case_id":prep.parent_case_id,"spec":_jsonable(prep.spec),
        "morphology":{"provenance":prep.morphology_provenance,"projection_error":prep.morphology_error,
                      "hash_raw":array_sha256(prep.psi_D_raw),"hash_resolved":array_sha256(prep.psi_D)},
        "admissibility":prep.admissibility,"b_bar":prep.b_bar,"g_bar":prep.g_bar,"coeff_error":prep.coeff_error,
        "hashes":{"b":array_sha256(prep.b),"g":array_sha256(prep.g),"k":array_sha256(prep.grid.k),
                  "mask":array_sha256(prep.grid.mask),"initial_condition":array_sha256(a0)},
        "fourier_convention":"f(xi)=sum fhat_l exp(-i k_l xi); NumPy code k=-2*pi*fftfreq",
        "dealiasing":"symmetric two-thirds projector",
        "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
        "schemas":{"model":prep.spec.model_schema,"solver":prep.spec.solver_schema,"results":prep.spec.result_schema},
    }
    path=paths.metadata/f"{prep.case_id}.json"; atomic_write_json(path,payload); return path


def save_checkpoint(prep: PreparedCase, paths: ProjectPaths, step: int, ahat: np.ndarray,
                    history: Dict[str,List[float]], peak: Dict[str,Any]) -> Path:
    if not np.all(np.isfinite(ahat)):
        raise ValueError(f"Refusing to checkpoint non-finite spectral state for {prep.case_id} at step {step}.")
    peak_ahat=np.asarray(peak.get("ahat",ahat))
    if not np.all(np.isfinite(peak_ahat)):
        raise ValueError(f"Refusing to checkpoint non-finite peak state for {prep.case_id} at step {step}.")
    cdir=paths.checkpoints/prep.case_id; cdir.mkdir(parents=True,exist_ok=True)
    path=cdir/"latest.npz"
    integrity=stable_hash(_integrity_payload(prep),32)
    payload={"step":np.array(step,dtype=np.int64),"ahat":ahat,
             "case_metadata_hash":np.array(stable_hash({"case_id":prep.case_id,"spec":_jsonable(prep.spec)})),
             "integrity_hash":np.array(integrity),"peak_R":np.array(float(peak.get("R",-np.inf))),
             "peak_step":np.array(int(peak.get("step",0)),dtype=np.int64),"peak_ahat":peak_ahat}
    for k,v in history.items(): payload[f"hist_{k}"]=np.asarray(v)
    atomic_save_npz(path,**payload); return path


def load_checkpoint(prep: PreparedCase, paths: ProjectPaths) -> Optional[Dict[str,Any]]:
    path=paths.checkpoints/prep.case_id/"latest.npz"
    if not path.exists(): return None
    with np.load(path,allow_pickle=False) as z:
        expected=stable_hash({"case_id":prep.case_id,"spec":_jsonable(prep.spec)})
        got=str(z["case_metadata_hash"].item())
        if got!=expected: raise RuntimeError(f"Checkpoint case-metadata mismatch for {prep.case_id}.")
        expected_integrity=stable_hash(_integrity_payload(prep),32)
        if "integrity_hash" not in z or str(z["integrity_hash"].item())!=expected_integrity:
            raise RuntimeError(f"Checkpoint morphology/coefficient/grid/initial-condition integrity mismatch for {prep.case_id}.")
        ahat=z["ahat"].copy(); peak_ahat=z["peak_ahat"].copy()
        if not np.all(np.isfinite(ahat)) or not np.all(np.isfinite(peak_ahat)):
            raise RuntimeError(f"Checkpoint contains a non-finite spectral state for {prep.case_id}; remove it and rerun the case.")
        hist={k[5:]:z[k].tolist() for k in z.keys() if k.startswith("hist_")}
        return {"step":int(z["step"]),"ahat":ahat,"history":hist,
                "peak":{"R":float(z["peak_R"]),"step":int(z["peak_step"]),"ahat":peak_ahat}}


def _finite_float_or_none(value: Any) -> Optional[float]:
    try: v=float(value)
    except (TypeError,ValueError): return None
    return v if np.isfinite(v) else None


def _runtime_state_check(ahat: np.ndarray, prep: PreparedCase) -> Dict[str,Any]:
    finite=bool(np.all(np.isfinite(ahat)))
    if not finite:
        return {"finite":False,"relative_imaginary":None,"forbidden_mode_fraction":None,"pass":False}
    a=np.fft.ifft(ahat)
    real_den=max(float(np.linalg.norm(a.real)),1e-30)
    ah_den=max(float(np.linalg.norm(ahat)),1e-30)
    real_ratio=_finite_float_or_none(np.linalg.norm(a.imag)/real_den)
    forbidden=_finite_float_or_none(np.linalg.norm(ahat[~prep.grid.mask])/ah_den)
    passed=(real_ratio is not None and forbidden is not None and real_ratio<1e-11 and forbidden<1e-13)
    return {"finite":True,"relative_imaginary":real_ratio,"forbidden_mode_fraction":forbidden,"pass":bool(passed)}


def _record(prep: PreparedCase, ahat: np.ndarray, step: int) -> Dict[str,float]:
    rhs=full_rhs_hat(ahat,prep)
    bal=integrals_and_balance(ahat,prep,rhs); sp=spectral_broadening(ahat,prep)
    return {"step":int(step),"s":float(step*prep.spec.dt),**bal,**sp}


def _record_is_finite(rec: Dict[str,Any]) -> bool:
    for value in rec.values():
        if isinstance(value,(int,np.integer)): continue
        try:
            if not np.isfinite(float(value)): return False
        except (TypeError,ValueError): return False
    return True


def _history_append(history: Dict[str,List[float]], rec: Dict[str,float]) -> None:
    for k,v in rec.items(): history.setdefault(k,[]).append(v)


def finalize_history(history: Dict[str,List[float]]) -> Dict[str,np.ndarray]:
    out={k:np.asarray(v) for k,v in history.items()}
    if "s" in out and len(out["s"])>=2 and "I2" in out:
        out["G_fd"]=np.gradient(np.log(np.maximum(out["I2"],np.finfo(float).tiny)),out["s"])
        out["RI1_integrated"]=out["I1"]-out["I1"][0]-np.concatenate([[0.0],np.cumsum(0.5*np.diff(out["s"])*(out["B1"][:-1]+out["B1"][1:]))])
        out["RE_integrated"]=out["E"]-out["E"][0]-np.concatenate([[0.0],np.cumsum(0.5*np.diff(out["s"])*(out["BE"][:-1]+out["BE"][1:]))])
    return out


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
        if step%spec.output_every_steps==0 or step==steps:
            state_now=_runtime_state_check(ah,prep)
            if not state_now["pass"]:
                failure={"reason":"RUNTIME_STATE_CHECK_FAILED","step":int(step),"s":float(step*spec.dt),"state_check":state_now}; break
            rec=_record(prep,ah,step)
            if not _record_is_finite(rec):
                failure={"reason":"NONFINITE_DIAGNOSTIC","step":int(step),"s":float(step*spec.dt)}; break
            _history_append(history,rec)
            if rec["R"]>peak["R"]: peak={"R":rec["R"],"step":step,"ahat":ah.copy()}
        if paths is not None and (step%spec.checkpoint_every_steps==0 or step==steps):
            save_checkpoint(prep,paths,step,ah,history,peak)

    hist=finalize_history(history); statecheck=_runtime_state_check(ah,prep)
    reached_final=(failure is None); runtime_valid=bool(reached_final and statecheck["pass"])
    if runtime_valid:
        numerical_status="VALID"; termination_reason=None; termination_step=int(steps); termination_s=float(spec.T_final)
        imax=int(np.argmax(hist["R"])); R_max=float(hist["R"][imax]); s_R_max=float(hist["s"][imax]); I2_final=float(hist["I2"][-1])
        G_bal_max=float(np.max(hist["G_bal"])); eta_tail_max=float(np.max(hist["eta_tail"]))
        max_abs_RI1_inst=float(np.max(np.abs(hist["R_I1_inst"]))); max_abs_RE_inst=float(np.max(np.abs(hist["R_E_inst"])))
    else:
        reason=(failure or {}).get("reason","FINAL_RUNTIME_STATE_INVALID")
        numerical_status="UNSTABLE" if reason in {"NONFINITE_SPECTRAL_STATE","NONFINITE_DIAGNOSTIC"} else "INVALID"
        termination_reason=reason; termination_step=int((failure or {}).get("step",steps)); termination_s=float((failure or {}).get("s",termination_step*spec.dt))
        R_max=s_R_max=I2_final=G_bal_max=eta_tail_max=max_abs_RI1_inst=max_abs_RE_inst=None
    summary={"case_id":prep.case_id,"parent_case_id":prep.parent_case_id,"completed":bool(reached_final),"runtime_valid":runtime_valid,
             "numerical_status":numerical_status,"termination_reason":termination_reason,"termination_step":termination_step,"termination_s":termination_s,
             "R_max":R_max,"s_R_max":s_R_max,"I2_final":I2_final,"G_bal_max":G_bal_max,"eta_tail_max":eta_tail_max,
             "max_abs_RI1_inst":max_abs_RI1_inst,"max_abs_RE_inst":max_abs_RE_inst,
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
            atomic_write_json(outdir/"COMPLETED.json",{"case_id":prep.case_id,"runtime_valid":True,"summary_hash":stable_hash(summary),"archive_sha256":file_sha256(outdir/"result.npz")})
        else:
            atomic_write_json(outdir/"FAILED.json",{"case_id":prep.case_id,"runtime_valid":False,"numerical_status":numerical_status,
                                                    "termination_reason":termination_reason,"termination_step":termination_step,"termination_s":termination_s,
                                                    "summary_hash":stable_hash(summary)})
    return {"prep":prep,"history":hist,"summary":summary,"ahat_final":ah,"ahat_peak":peak["ahat"],"budget_peak":budget_peak}


def verify_restart_equivalence() -> Dict[str,Any]:
    spec=CaseSpec("P1",Wo0=5,N=64,dt=.002,T_final=.08,eps_b=.12,eps_g=.1,q=1.0,k0=.5,output_every_steps=5)
    prep=prepare_case(spec); ah0=np.fft.fft(initial_condition(spec,prep.grid)); etd=etd_coefficients(prep); steps=int(round(spec.T_final/spec.dt)); mid=steps//2
    a=ah0.copy()
    for _ in range(steps): a=etdrk4_step(a,prep,etd)
    b=ah0.copy()
    for _ in range(mid): b=etdrk4_step(b,prep,etd)
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/"c.npz"; atomic_save_npz(f,ahat=b,step=np.array(mid))
        with np.load(f,allow_pickle=False) as z: b=z["ahat"].copy(); start=int(z["step"])
    for _ in range(start,steps): b=etdrk4_step(b,prep,etd)
    err=float(np.linalg.norm(a-b)/np.linalg.norm(a))
    return {"name":"restart_equivalence","relative_error":err,"pass":err<1e-15}


__all__ = [name for name in globals() if not name.startswith("_")]
