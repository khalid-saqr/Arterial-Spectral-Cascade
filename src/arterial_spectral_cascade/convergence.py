from __future__ import annotations

from .core import *
from .storage import run_case
from .parent import run_paired_case

def _finite_number(value: Any) -> bool:
    if value is None: return False
    try: return bool(np.isfinite(float(value)))
    except (TypeError,ValueError): return False


def _model_rejection_row(axis_name: str, axis_value: Any, prep: PreparedCase) -> Dict[str,Any]:
    return {axis_name:axis_value,"status":prep.admissibility["status"],"model_status":prep.admissibility["status"],
            "numerical_status":"NOT_RUN","runtime_valid":False,"R_max":None,"s_R_max":None,"I2_final":None,
            "morphology_error":prep.morphology_error,"coeff_error":prep.coeff_error,"morphology_limit":prep.spec.morphology_projection_limit,"coeff_limit":prep.spec.coeff_projection_limit,"chi_h_max":None}


def _single_trial_row(axis_name: str, axis_value: Any, prep: PreparedCase, result: Dict[str,Any]) -> Dict[str,Any]:
    summary=result["summary"]; valid=bool(summary.get("runtime_valid",False))
    finite=all(_finite_number(summary.get(k)) for k in ("R_max","s_R_max","I2_final"))
    if valid and finite:
        hist=result["history"]
        return {axis_name:axis_value,"status":"ADMISSIBLE","model_status":"ADMISSIBLE","numerical_status":"VALID","runtime_valid":True,
                "R_max":float(summary["R_max"]),"s_R_max":float(summary["s_R_max"]),"I2_final":float(summary["I2_final"]),
                "morphology_error":prep.morphology_error,"coeff_error":prep.coeff_error,"morphology_limit":prep.spec.morphology_projection_limit,"coeff_limit":prep.spec.coeff_projection_limit,"eta_tail_max":summary.get("eta_tail_max"),
                "chi_h_max":summary.get("chi_h_max"),"max_abs_RI1_integrated":summary.get("max_abs_RI1_integrated"),
                "max_abs_RE_integrated":summary.get("max_abs_RE_integrated"),
                "history":{"s":hist["s"].tolist(),"I2":hist["I2"].tolist(),"R":hist["R"].tolist(),
                           "eta_tail":hist["eta_tail"].tolist(),"RI1_integrated":hist.get("RI1_integrated",np.zeros_like(hist["s"])).tolist(),
                           "RE_integrated":hist.get("RE_integrated",np.zeros_like(hist["s"])).tolist()}}
    numerical_status=str(summary.get("numerical_status") or "INVALID")
    return {axis_name:axis_value,"status":"NUMERICALLY_UNSTABLE" if numerical_status=="UNSTABLE" else "NUMERICALLY_INVALID",
            "model_status":"ADMISSIBLE","numerical_status":numerical_status,"runtime_valid":False,
            "R_max":None,"s_R_max":None,"I2_final":None,"morphology_error":prep.morphology_error,"coeff_error":prep.coeff_error,"morphology_limit":prep.spec.morphology_projection_limit,"coeff_limit":prep.spec.coeff_projection_limit,
            "eta_tail_max":None,"chi_h_max":None,"termination_reason":summary.get("termination_reason"),
            "termination_step":summary.get("termination_step"),"termination_s":summary.get("termination_s")}


def _paired_trial_row(axis_name: str, axis_value: Any, prep: PreparedCase, result: Dict[str,Any]) -> Dict[str,Any]:
    summary=result["summary"]; valid=bool(summary.get("runtime_valid",False))
    required=("R_max_het","R_max_mm","DeltaR_max_timewise","D2_max")
    finite=all(_finite_number(summary.get(k)) for k in required)
    if valid and finite:
        hh=result["het_history"]; mh=result["mm_history"]; cmp=result["comparison"]
        imax=int(np.argmax(hh["R"]))
        return {axis_name:axis_value,"status":"ADMISSIBLE","model_status":"ADMISSIBLE","numerical_status":"VALID","runtime_valid":True,
                "R_max":float(summary["R_max_het"]),"s_R_max":float(hh["s"][imax]),"I2_final":float(hh["I2"][-1]),
                "R_max_mm":float(summary["R_max_mm"]),"DeltaR_max_timewise":float(summary["DeltaR_max_timewise"]),"D2_max":float(summary["D2_max"]),
                "morphology_error":prep.morphology_error,"coeff_error":prep.coeff_error,"morphology_limit":prep.spec.morphology_projection_limit,"coeff_limit":prep.spec.coeff_projection_limit,
                "eta_tail_max":float(np.max(hh["eta_tail"])),"chi_h_max":float(np.max(hh["chi_h"])),
                "max_abs_RI1_integrated":float(np.max(np.abs(hh.get("RI1_integrated",np.array([0.0]))))),
                "max_abs_RE_integrated":float(np.max(np.abs(hh.get("RE_integrated",np.array([0.0]))))),
                "history":{"s":hh["s"].tolist(),"I2":hh["I2"].tolist(),"R":hh["R"].tolist(),
                           "I2_mm":mh["I2"].tolist(),"R_mm":mh["R"].tolist(),"DeltaR":cmp["DeltaR"].tolist(),"D2":cmp["D2"].tolist(),
                           "eta_tail":hh["eta_tail"].tolist(),"RI1_integrated":hh.get("RI1_integrated",np.zeros_like(hh["s"])).tolist(),
                           "RE_integrated":hh.get("RE_integrated",np.zeros_like(hh["s"])).tolist()}}
    numerical_status=str(summary.get("numerical_status") or "INVALID")
    return {axis_name:axis_value,"status":"NUMERICALLY_UNSTABLE" if numerical_status=="UNSTABLE" else "NUMERICALLY_INVALID",
            "model_status":"ADMISSIBLE","numerical_status":numerical_status,"runtime_valid":False,"R_max":None,"s_R_max":None,"I2_final":None,
            "R_max_mm":None,"DeltaR_max_timewise":None,"D2_max":None,"eta_tail_max":None,"chi_h_max":None,
            "morphology_error":prep.morphology_error,"coeff_error":prep.coeff_error,"morphology_limit":prep.spec.morphology_projection_limit,"coeff_limit":prep.spec.coeff_projection_limit,"termination_reason":summary.get("termination_reason"),
            "termination_step":summary.get("termination_step"),"termination_s":summary.get("termination_s")}


def _valid_convergence_row(row: Dict[str,Any]) -> bool:
    return (row.get("model_status")=="ADMISSIBLE" and row.get("numerical_status")=="VALID"
            and row.get("runtime_valid") is True and _finite_number(row.get("R_max"))
            and _finite_number(row.get("I2_final")) and isinstance(row.get("history"),dict))


def _interp_error(coarse: Dict[str,Any], fine: Dict[str,Any], key: str, relative: bool=True, scale_floor: float=1e-12) -> float:
    sc=np.asarray(coarse["history"]["s"],float); sf=np.asarray(fine["history"]["s"],float)
    yc=np.asarray(coarse["history"][key],float); yf=np.asarray(fine["history"][key],float)
    ref=np.interp(sc,sf,yf); err=float(np.max(np.abs(yc-ref)))
    if not relative: return err
    return err/max(float(np.max(np.abs(ref))),float(scale_floor))


def _nonincreasing_or_floor(fine: float, coarse: float, floor: float=1e-10, slack: float=1.05) -> bool:
    if max(abs(fine),abs(coarse)) <= floor: return True
    return abs(fine) <= slack*max(abs(coarse),floor)


def _attach_convergence_changes(rows: List[Dict[str,Any]], axis_name: str) -> None:
    prev=None
    for row in rows:
        if not _valid_convergence_row(row): continue
        if prev is not None:
            row[f"prev_{axis_name}"]=prev[axis_name]
            row["epsilon_I2_history"]=_interp_error(prev,row,"I2",True)
            row["epsilon_I2_mm_history"]=_interp_error(prev,row,"I2_mm",True) if "I2_mm" in row["history"] else 0.0
            row["rel_R_history_change"]=_interp_error(prev,row,"R",True)
            row["rel_R_mm_history_change"]=_interp_error(prev,row,"R_mm",True) if "R_mm" in row["history"] else 0.0
            row["rel_I2_change_vs_prev"]=row["epsilon_I2_history"]
            row["rel_Rmax_change_vs_prev"]=abs(row["R_max"]-prev["R_max"])/max(abs(row["R_max"]),np.finfo(float).eps)
            T=max(float(row["history"]["s"][-1]),np.finfo(float).eps)
            row["rel_sRmax_change_vs_prev"]=abs(row["s_R_max"]-prev["s_R_max"])/T
            if "DeltaR" in row["history"] and "DeltaR" in prev["history"]:
                row["rel_DeltaR_history_change"]=_interp_error(prev,row,"DeltaR",True)
                row["rel_D2_history_change"]=_interp_error(prev,row,"D2",True)
            else:
                row["rel_DeltaR_history_change"]=0.0; row["rel_D2_history_change"]=0.0
            row["rel_eta_tail_change"]=_interp_error(prev,row,"eta_tail",True,scale_floor=1e-8)
            r1f=float(row.get("max_abs_RI1_integrated") or 0.0); r1c=float(prev.get("max_abs_RI1_integrated") or 0.0)
            ref=float(row.get("max_abs_RE_integrated") or 0.0); rec=float(prev.get("max_abs_RE_integrated") or 0.0)
            row["rel_integrated_balance_change"]=max(abs(r1f-r1c)/max(abs(r1f),abs(r1c),1e-10),
                                                        abs(ref-rec)/max(abs(ref),abs(rec),1e-10))
            row["integrated_balance_refinement_pass"]=bool(_nonincreasing_or_floor(r1f,r1c) and _nonincreasing_or_floor(ref,rec))
            row["morphology_projection_stable"]=bool(row["morphology_error"] <= row["morphology_limit"] and
                                                       (axis_name!="N" or row["morphology_error"] <= max(1.05*prev["morphology_error"],1e-14)))
            row["coefficient_projection_stable"]=bool(row["coeff_error"] <= row["coeff_limit"] and
                                                        (axis_name!="N" or row["coeff_error"] <= max(1.05*prev["coeff_error"],1e-14)))
            row["chi_h_ratio_vs_prev"]=float(row["chi_h_max"])/max(float(prev["chi_h_max"]),1e-30) if _finite_number(row.get("chi_h_max")) and _finite_number(prev.get("chi_h_max")) else None
        prev=row


def spatial_convergence(case: CaseSpec, N_values: Sequence[int], progress: bool=False) -> Dict[str,Any]:
    rows=[]; paired=case.case_class in {"P1","DL","DM","DR"}
    for N in N_values:
        prep=prepare_case(replace(case,N=int(N)))
        if prep.admissibility["status"]!="ADMISSIBLE": rows.append(_model_rejection_row("N",N,prep)); continue
        res=run_paired_case(prep,paths=None,resume=False,progress=progress) if paired else run_case(prep,paths=None,resume=False,progress=progress)
        rows.append(_paired_trial_row("N",N,prep,res) if paired else _single_trial_row("N",N,prep,res))
    _attach_convergence_changes(rows,"N")
    return {"rows":rows,"paired":paired,"criterion":"full-history Solver Design spatial refinement"}


def temporal_convergence(case: CaseSpec, dt_values: Sequence[float], progress: bool=False) -> Dict[str,Any]:
    rows=[]; paired=case.case_class in {"P1","DL","DM","DR"}
    out_interval=case.output_every_steps*case.dt; cp_interval=case.checkpoint_every_steps*case.dt
    for dt in dt_values:
        dt=float(dt); spec=replace(case,dt=dt,output_every_steps=max(1,int(round(out_interval/dt))),
                                   checkpoint_every_steps=max(1,int(round(cp_interval/dt))))
        prep=prepare_case(spec)
        if prep.admissibility["status"]!="ADMISSIBLE": rows.append(_model_rejection_row("dt",dt,prep)); continue
        res=run_paired_case(prep,paths=None,resume=False,progress=progress) if paired else run_case(prep,paths=None,resume=False,progress=progress)
        rows.append(_paired_trial_row("dt",dt,prep,res) if paired else _single_trial_row("dt",dt,prep,res))
    _attach_convergence_changes(rows,"dt")
    return {"rows":rows,"paired":paired,"criterion":"full-history Solver Design temporal refinement"}


def convergence_pair_pass(row: Dict[str,Any], i2tol: float, obstol: float, spatial: bool) -> bool:
    if not _valid_convergence_row(row) or "epsilon_I2_history" not in row: return False
    metrics=(row.get("rel_R_history_change",np.inf),row.get("rel_R_mm_history_change",np.inf),
             row.get("rel_Rmax_change_vs_prev",np.inf),row.get("rel_sRmax_change_vs_prev",np.inf),
             row.get("rel_DeltaR_history_change",np.inf),row.get("rel_D2_history_change",np.inf))
    if not (row["epsilon_I2_history"] < i2tol and row.get("epsilon_I2_mm_history",0.0) < i2tol
            and all(float(v) < obstol for v in metrics)): return False
    if spatial:
        if not (row.get("rel_integrated_balance_change",np.inf) < obstol): return False
    elif not row.get("integrated_balance_refinement_pass",False): return False
    if not row.get("morphology_projection_stable",False) or not row.get("coefficient_projection_stable",False): return False
    if spatial:
        if not (row.get("rel_eta_tail_change",np.inf) < obstol): return False
        if not (_finite_number(row.get("eta_tail_max")) and float(row["eta_tail_max"]) < obstol): return False
    return True


def convergence_acceptance(rows: Sequence[Dict[str,Any]], axis_name: str, i2tol: float, obstol: float,
                           spatial: Optional[bool]=None):
    if spatial is None: spatial=(axis_name=="N")
    valid=[r for r in rows if _valid_convergence_row(r)]
    for i in range(1,len(valid)):
        if convergence_pair_pass(valid[i],i2tol,obstol,spatial): return valid[i-1][axis_name]
    return None

__all__=[name for name in globals() if not name.startswith("_")]
