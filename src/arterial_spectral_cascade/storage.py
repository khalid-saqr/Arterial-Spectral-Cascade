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
    for sub in ["parent","stenosis","dilation","matched_mean","mechanism","width","optional"]:
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
    if prep.parent_case_id is not None: base=paths.results/"matched_mean"
    elif c in {"H0","P0","P1"}: base=paths.results/"parent"
    elif c=="DS": base=paths.results/"stenosis"
    elif c=="DA": base=paths.results/"dilation"
    elif c=="DM": base=paths.results/"optional"
    else: base=paths.results/"optional"
    p=base/prep.case_id; p.mkdir(parents=True,exist_ok=True); return p


def save_case_metadata(prep: PreparedCase, paths: ProjectPaths) -> Path:
    a0=initial_condition(prep.spec,prep.grid)
    payload={"case_id":prep.case_id,"parent_case_id":prep.parent_case_id,"spec":_jsonable(prep.spec),
             "admissibility":prep.admissibility,"b_bar":prep.b_bar,"g_bar":prep.g_bar,"coeff_error":prep.coeff_error,
             "hashes":{"b":array_sha256(prep.b),"g":array_sha256(prep.g),"k":array_sha256(prep.grid.k),"mask":array_sha256(prep.grid.mask),"initial_condition":array_sha256(a0)},
             "fourier_convention":"f(xi)=sum fhat_l exp(-i k_l xi); NumPy code k=-2*pi*fftfreq",
             "dealiasing":"symmetric two-thirds projector",
             "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
             "schemas":{"model":MODEL_SCHEMA,"solver":SOLVER_SCHEMA,"results":RESULT_SCHEMA}}
    path=paths.metadata/f"{prep.case_id}.json"; atomic_write_json(path,payload); return path


def save_checkpoint(prep: PreparedCase, paths: ProjectPaths, step: int, ahat: np.ndarray, history: Dict[str,List[float]], peak: Dict[str,Any]) -> Path:
    cdir=paths.checkpoints/prep.case_id; cdir.mkdir(parents=True,exist_ok=True)
    path=cdir/"latest.npz"
    integrity=stable_hash({"case_id":prep.case_id,"spec":_jsonable(prep.spec),"b":array_sha256(prep.b),"g":array_sha256(prep.g),"k":array_sha256(prep.grid.k),"mask":array_sha256(prep.grid.mask),"ic":array_sha256(initial_condition(prep.spec,prep.grid))},32)
    payload={"step":np.array(step,dtype=np.int64),"ahat":ahat,
             "case_metadata_hash":np.array(stable_hash({"case_id":prep.case_id,"spec":_jsonable(prep.spec)})),"integrity_hash":np.array(integrity),
             "peak_R":np.array(float(peak.get("R",-np.inf))),"peak_step":np.array(int(peak.get("step",0)),dtype=np.int64),
             "peak_ahat":np.asarray(peak.get("ahat",ahat))}
    for k,v in history.items(): payload[f"hist_{k}"]=np.asarray(v)
    atomic_save_npz(path,**payload); return path


def load_checkpoint(prep: PreparedCase, paths: ProjectPaths) -> Optional[Dict[str,Any]]:
    path=paths.checkpoints/prep.case_id/"latest.npz"
    if not path.exists(): return None
    with np.load(path,allow_pickle=False) as z:
        expected=stable_hash({"case_id":prep.case_id,"spec":_jsonable(prep.spec)})
        got=str(z["case_metadata_hash"].item())
        if got!=expected: raise RuntimeError(f"Checkpoint case-metadata mismatch for {prep.case_id}.")
        expected_integrity=stable_hash({"case_id":prep.case_id,"spec":_jsonable(prep.spec),"b":array_sha256(prep.b),"g":array_sha256(prep.g),"k":array_sha256(prep.grid.k),"mask":array_sha256(prep.grid.mask),"ic":array_sha256(initial_condition(prep.spec,prep.grid))},32)
        if "integrity_hash" not in z or str(z["integrity_hash"].item())!=expected_integrity:
            raise RuntimeError(f"Checkpoint coefficient/grid/initial-condition integrity mismatch for {prep.case_id}.")
        hist={k[5:]:z[k].tolist() for k in z.keys() if k.startswith("hist_")}
        return {"step":int(z["step"]),"ahat":z["ahat"].copy(),"history":hist,
                "peak":{"R":float(z["peak_R"]),"step":int(z["peak_step"]),"ahat":z["peak_ahat"].copy()}}


def _runtime_state_check(ahat: np.ndarray, prep: PreparedCase) -> Dict[str,Any]:
    finite=bool(np.all(np.isfinite(ahat)))
    a=np.fft.ifft(ahat)
    real_ratio=float(np.linalg.norm(a.imag)/max(np.linalg.norm(a.real),1e-30))
    forbidden=float(np.linalg.norm(ahat[~prep.grid.mask])/max(np.linalg.norm(ahat),1e-30))
    return {"finite":finite,"relative_imaginary":real_ratio,"forbidden_mode_fraction":forbidden,
            "pass":finite and real_ratio<1e-11 and forbidden<1e-13}


def _record(prep: PreparedCase, ahat: np.ndarray, step: int) -> Dict[str,float]:
    rhs=full_rhs_hat(ahat,prep)
    bal=integrals_and_balance(ahat,prep,rhs); sp=spectral_broadening(ahat,prep)
    return {"step":int(step),"s":float(step*prep.spec.dt),**bal,**sp}


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
    if abs(steps*spec.dt-spec.T_final)>1e-10*max(1.0,spec.T_final): raise ValueError("T_final/dt must be an integer for deterministic fixed-step integration.")
    if paths is not None: save_case_metadata(prep,paths)
    cp=load_checkpoint(prep,paths) if (paths is not None and resume) else None
    if cp:
        start=cp["step"]; ah=cp["ahat"]; history=cp["history"]; peak=cp["peak"]
    else:
        start=0; ah=project_hat(np.fft.fft(initial_condition(spec,prep.grid)),prep.grid); history={}; rec=_record(prep,ah,0); _history_append(history,rec); peak={"R":rec["R"],"step":0,"ahat":ah.copy()}
    initial_ahat=project_hat(np.fft.fft(initial_condition(spec,prep.grid)),prep.grid)
    etd=etd_coefficients(prep)
    iterator=range(start+1,steps+1)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator=tqdm(iterator,desc=prep.case_id,leave=False)
        except Exception: pass
    for step in iterator:
        ah=etdrk4_step(ah,prep,etd,include_nonlinearity,include_heterogeneity)
        if step%spec.output_every_steps==0 or step==steps:
            rec=_record(prep,ah,step); _history_append(history,rec)
            if rec["R"]>peak["R"]: peak={"R":rec["R"],"step":step,"ahat":ah.copy()}
        if paths is not None and (step%spec.checkpoint_every_steps==0 or step==steps):
            save_checkpoint(prep,paths,step,ah,history,peak)
    hist=finalize_history(history); statecheck=_runtime_state_check(ah,prep)
    imax=int(np.argmax(hist["R"])); summary={
        "case_id":prep.case_id,"parent_case_id":prep.parent_case_id,"completed":True,"runtime_valid":bool(statecheck["pass"]),
        "R_max":float(hist["R"][imax]),"s_R_max":float(hist["s"][imax]),"I2_final":float(hist["I2"][-1]),
        "G_bal_max":float(np.max(hist["G_bal"])),"eta_tail_max":float(np.max(hist["eta_tail"])),
        "max_abs_RI1_inst":float(np.max(np.abs(hist["R_I1_inst"]))),"max_abs_RE_inst":float(np.max(np.abs(hist["R_E_inst"]))),
        "admissibility":prep.admissibility,"state_check":statecheck}
    budget_peak=modal_energy_budget(peak["ahat"],prep) if spec.mechanism else None
    if paths is not None:
        outdir=case_result_subdir(prep,paths)
        arrays={"xi":prep.grid.xi,"r":prep.r,"Wo_R":prep.Wo_R,"b":prep.b,"g":prep.g,
                "bhat_norm":np.fft.fft(prep.b)/prep.grid.N,"ghat_norm":np.fft.fft(prep.g)/prep.grid.N,
                "ahat_initial":initial_ahat,"ahat_peak":peak["ahat"],"ahat_final":ah}
        arrays.update({f"hist_{k}":v for k,v in hist.items()})
        if budget_peak is not None:
            for k,v in budget_peak.items(): arrays[f"budget_peak_{k}"]=v
        atomic_save_npz(outdir/"result.npz",**arrays)
        atomic_write_json(outdir/"summary.json",summary)
        # Completion marker is written last, after archive read-back.
        with np.load(outdir/"result.npz",allow_pickle=False) as z: _=z["ahat_final"].shape
        atomic_write_json(outdir/"COMPLETED.json",{"case_id":prep.case_id,"runtime_valid":summary["runtime_valid"],"summary_hash":stable_hash(summary),"archive_sha256":file_sha256(outdir/"result.npz")})
    return {"prep":prep,"history":hist,"summary":summary,"ahat_final":ah,"ahat_peak":peak["ahat"],"budget_peak":budget_peak}


def verify_restart_equivalence() -> Dict[str,Any]:
    spec=CaseSpec("P1",Wo0=5,N=64,dt=.002,T_final=.08,eps_b=.12,eps_g=.1,q=1.0,k0=.5,output_every_steps=5)
    prep=prepare_case(spec); ah0=np.fft.fft(initial_condition(spec,prep.grid)); etd=etd_coefficients(prep); steps=int(round(spec.T_final/spec.dt)); mid=steps//2
    a=ah0.copy()
    for _ in range(steps): a=etdrk4_step(a,prep,etd)
    b=ah0.copy()
    for _ in range(mid): b=etdrk4_step(b,prep,etd)
    # mimic checkpoint serialization exactly
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/"c.npz"; atomic_save_npz(f,ahat=b,step=np.array(mid));
        with np.load(f,allow_pickle=False) as z: b=z["ahat"].copy(); start=int(z["step"])
    for _ in range(start,steps): b=etdrk4_step(b,prep,etd)
    err=float(np.linalg.norm(a-b)/np.linalg.norm(a))
    return {"name":"restart_equivalence","relative_error":err,"pass":err<1e-15}




__all__ = [name for name in globals() if not name.startswith("_")]
