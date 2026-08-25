from __future__ import annotations

import os, sys, json, time
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

from .core import *
from .core import _jsonable
from .storage import *
from .planning import *
from .parent import *

# Notebook-level study orchestration. This module declares parameter cases only;
# it does not redefine the Mathematical Model or Solver Design.

MORPHOLOGY_CLASSES = {
    "DL": {
        "name":"single localized morphology",
        "field":"Psi_L",
        "required_parameters":("xi_c","w","p","chi_b","chi_g"),
    },
    "DM": {
        "name":"normalized multiple-lesion morphology",
        "field":"Psi_M",
        "required_parameters":("lesions","chi_b","chi_g"),
    },
    "DR": {
        "name":"distributed or geometry-derived morphology",
        "field":"Psi_R or sampled Psi_D",
        "required_parameters":("chi_b","chi_g"),
    },
}

# No clinical severity-to-coefficient law exists in the Mathematical Model.
# Consequently the package does not ship a default clinical disease library.
# Users must provide coefficient-space cases explicitly or via anatomical
# calibration external to this package.
STUDY_CONFIG = {
    "RUN_MODE": os.environ.get("POF_RUN_MODE", "FULL_STUDY"),
    "MOUNT_DRIVE": True,
    "PROJECT_ROOT": None,
    "DISEASE_CASES": (),

    "R0_OVER_L0": 0.05,
    "SLOW_VARIATION_LIMIT": 0.10,
    "MORPHOLOGY_PROJECTION_LIMIT": 1e-8,
    "COEFF_PROJECTION_LIMIT": 1e-8,

    "COARSE_WO": (2.0,5.0,10.0,15.0,20.0),
    "REFINE_MIN_SPACING": 0.625,
    "MAX_REFINE_ROUNDS": 2,

    "STUDY_N": 512,
    "STUDY_DT": 2e-4,
    "STUDY_T_FINAL": 60.0,
    "OUTPUT_INTERVAL": 0.05,
    "CHECKPOINT_INTERVAL": 2.0,

    "RUN_PARENT_DETAILED": False,
    "SELECTION_N_VALUES": (256,512,1024),
    "SELECTION_DT_VALUES": (8e-4,4e-4,2e-4),
    "SELECTION_T_FINAL": 60.0,
    "I2_CONVERGENCE_TOL": 1e-5,
    "OBSERVABLE_CONVERGENCE_TOL": 1e-3,

    # Optional morphology-scale factors. Only analytical DL/DM cases are scaled
    # automatically; geometry-derived DR fields require a new supplied field.
    "SCALE_FACTORS": (),
}

ALLOWED_MODES={"QUICK_CHECK","VERIFICATION","PARAMETER_SELECTION","FIGURES","FULL_STUDY"}


def validate_study_config(cfg=STUDY_CONFIG, disease_required=False):
    if cfg["RUN_MODE"] not in ALLOWED_MODES:
        raise ValueError(f"RUN_MODE must be one of {sorted(ALLOWED_MODES)}")
    if disease_required:
        missing=[k for k in ["R0_OVER_L0","SLOW_VARIATION_LIMIT"] if cfg.get(k) is None]
        if missing: raise RuntimeError("Disease calculations require explicit reduced-order consistency inputs: "+", ".join(missing))
        if not cfg.get("DISEASE_CASES"):
            raise RuntimeError(
                "No disease coefficient-space cases are configured. Supply DISEASE_CASES with morphology parameters and externally justified or parametrically varied chi_b/chi_g values."
            )
    return True


def configured_root(cfg=STUDY_CONFIG):
    if cfg.get("PROJECT_ROOT"): return Path(cfg["PROJECT_ROOT"]).expanduser()
    if "google.colab" in sys.modules: return Path("/content/drive/MyDrive/PoF_ArterialSpectralCascade")
    return Path.cwd()/"PoF_ArterialSpectralCascade_local"


def output_stride(dt, cfg=STUDY_CONFIG): return max(1,int(round(cfg["OUTPUT_INTERVAL"]/dt)))
def checkpoint_stride(dt, cfg=STUDY_CONFIG): return max(1,int(round(cfg["CHECKPOINT_INTERVAL"]/dt)))


def morphology_class_table() -> pd.DataFrame:
    return pd.DataFrame([{"case_class":k,**v} for k,v in MORPHOLOGY_CLASSES.items()])


def _normalize_lesions(items) -> Tuple[Lesion,...]:
    out=[]
    for x in items:
        if isinstance(x,Lesion): out.append(x)
        elif isinstance(x,dict): out.append(Lesion(**x))
        else: raise TypeError("DM lesions must be Lesion objects or dictionaries.")
    return tuple(out)


def _normalize_modes(items) -> Tuple[DistributedMode,...]:
    out=[]
    for x in items:
        if isinstance(x,DistributedMode): out.append(x)
        elif isinstance(x,dict): out.append(DistributedMode(**x))
        else: raise TypeError("DR distributed_modes must be DistributedMode objects or dictionaries.")
    return tuple(out)


def case_record_to_spec(record: Dict[str,Any], Wo: float, N: Optional[int]=None, dt: Optional[float]=None,
                        T_final: Optional[float]=None, mechanism: bool=False, cfg=STUDY_CONFIG) -> CaseSpec:
    """Convert one explicit coefficient-space study record into a CaseSpec.

    Required disease sensitivities are never inferred from anatomy or labels.
    Primary disease-only records use eps_b=eps_g=0. A combined parent-background
    modulation must set ``combined_parent_background=True`` explicitly.
    """
    cls=str(record.get("case_class",""))
    if cls not in {"DL","DM","DR"}: raise ValueError("Disease case_class must be DL, DM, or DR.")
    if "case_id" not in record or not str(record["case_id"]).strip(): raise ValueError("Each configured disease record requires a non-empty case_id.")
    if "chi_b" not in record or "chi_g" not in record: raise ValueError("Each disease record must explicitly supply chi_b and chi_g.")
    eps_b=float(record.get("eps_b",0.0)); eps_g=float(record.get("eps_g",0.0))
    if (eps_b!=0 or eps_g!=0) and not bool(record.get("combined_parent_background",False)):
        raise ValueError("Disease-only cases require eps_b=eps_g=0 unless combined_parent_background=True is declared explicitly.")
    N=cfg["STUDY_N"] if N is None else int(N); dt=cfg["STUDY_DT"] if dt is None else float(dt)
    T_final=cfg["STUDY_T_FINAL"] if T_final is None else float(T_final)
    common=dict(
        case_class=cls,Wo0=float(Wo),N=N,dt=dt,T_final=T_final,k0=float(record.get("k0",1.0)),
        eps_b=eps_b,eps_g=eps_g,q=float(record.get("q",1.0)),chi_b=float(record["chi_b"]),chi_g=float(record["chi_g"]),
        output_every_steps=output_stride(dt,cfg),checkpoint_every_steps=checkpoint_stride(dt,cfg),mechanism=bool(mechanism),
        R0_over_L0=float(cfg["R0_OVER_L0"]),slow_variation_limit=float(cfg["SLOW_VARIATION_LIMIT"]),
        morphology_projection_limit=float(cfg["MORPHOLOGY_PROJECTION_LIMIT"]),coeff_projection_limit=float(cfg["COEFF_PROJECTION_LIMIT"]),
        notes=str(record.get("notes",record["case_id"])),
    )
    if cls=="DL":
        common.update(xi_c=float(record.get("xi_c",2*np.pi)),w=float(record["w"]),p=int(record.get("p",1)))
    elif cls=="DM":
        common.update(lesions=_normalize_lesions(record.get("lesions",())))
    else:
        modes=_normalize_modes(record.get("distributed_modes",()))
        sampled=tuple(float(v) for v in record.get("sampled_psi",()))
        common.update(distributed_modes=modes,sampled_psi=sampled,
                      morphology_provenance=str(record.get("morphology_provenance","")),
                      morphology_scale=None if record.get("morphology_scale") is None else float(record["morphology_scale"]))
    return CaseSpec(**common)


def configured_case_table(cfg=STUDY_CONFIG) -> pd.DataFrame:
    rows=[]
    for record in cfg.get("DISEASE_CASES",()):
        r=dict(record)
        row={"case_id":str(r.get("case_id","")),"case_class":str(r.get("case_class","")),
             "chi_b":r.get("chi_b"),"chi_g":r.get("chi_g"),"combined_parent_background":bool(r.get("combined_parent_background",False)),
             "morphology_provenance":str(r.get("morphology_provenance","analytic morphology")),"notes":str(r.get("notes",""))}
        rows.append(row)
    return pd.DataFrame(rows)


def geometry_derived_spec(psi_D: Sequence[float], provenance: str, Wo: float, chi_b: float, chi_g: float,
                          morphology_scale: float, N: Optional[int]=None, dt: Optional[float]=None,
                          T_final: Optional[float]=None, cfg=STUDY_CONFIG, notes: str="") -> CaseSpec:
    values=tuple(float(v) for v in psi_D); N=len(values) if N is None else int(N)
    if N!=len(values): raise ValueError("N must equal the number of supplied morphology samples; no hidden interpolation is performed.")
    rec={"case_id":"geometry-derived","case_class":"DR","chi_b":float(chi_b),"chi_g":float(chi_g),
         "sampled_psi":values,"morphology_provenance":str(provenance),"morphology_scale":float(morphology_scale),"notes":notes}
    return case_record_to_spec(rec,Wo,N=N,dt=dt,T_final=T_final,cfg=cfg)


def preflight_model_cases(paths, cfg=STUDY_CONFIG):
    validate_study_config(cfg,disease_required=True)
    rows=[]
    for record in cfg["DISEASE_CASES"]:
        for wo in cfg["COARSE_WO"]:
            prep=prepare_case(case_record_to_spec(record,wo,N=cfg["STUDY_N"],dt=cfg["STUDY_DT"],T_final=cfg["STUDY_T_FINAL"],cfg=cfg))
            ar=prep.admissibility
            rows.append({"case_id":record["case_id"],"case_class":record["case_class"],"Wo":float(wo),
                         "chi_b":float(record["chi_b"]),"chi_g":float(record["chi_g"]),"status":ar["status"],
                         "psi_mean":ar["psi_mean"],"b_heterogeneity":ar["b_heterogeneity"],"g_heterogeneity":ar["g_heterogeneity"],
                         "morphology_error":prep.morphology_error,"coeff_error":prep.coeff_error,
                         "R0_over_ellD":ar["long_wave"]["R0_over_ellD"],"morphology_provenance":prep.morphology_provenance})
    pf=pd.DataFrame(rows)
    tmp=paths.tables/"model_case_preflight.csv.tmp"; pf.to_csv(tmp,index=False); os.replace(tmp,paths.tables/"model_case_preflight.csv")
    atomic_write_json(paths.verification/"MODEL_PREFLIGHT.json",{"pass":bool((pf.status=="ADMISSIBLE").all()),
                      "model_schema":MODEL_SCHEMA,"solver_schema":SOLVER_SCHEMA,"case_count":len(cfg["DISEASE_CASES"])})
    if not (pf.status=="ADMISSIBLE").all():
        raise RuntimeError("At least one configured coefficient-space disease case failed Mathematical Model admissibility/resolution preflight.")
    return pf


def convergence_acceptance(rows, x_name, i2tol, obstol):
    valid=[r for r in rows if r.get("status")=="ADMISSIBLE"]
    for i in range(1,len(valid)):
        r=valid[i]
        if r.get("rel_I2_change_vs_prev",np.inf)<i2tol and r.get("rel_Rmax_change_vs_prev",np.inf)<obstol:
            return valid[i-1][x_name]
    return None


def _demanding_record_per_class(preflight: pd.DataFrame, cfg=STUDY_CONFIG):
    records={str(r["case_id"]):r for r in cfg["DISEASE_CASES"]}
    selected=[]
    for cls in sorted(preflight.case_class.unique()):
        sub=preflight[preflight.case_class==cls].copy()
        sub["worst_heterogeneity"]=sub[["b_heterogeneity","g_heterogeneity"]].max(axis=1)
        row=sub.loc[sub.worst_heterogeneity.idxmax()]
        selected.append(records[str(row.case_id)])
    return selected


def run_parameter_selection(paths, cfg=STUDY_CONFIG, progress=True):
    validate_study_config(cfg,disease_required=True); pf=preflight_model_cases(paths,cfg)
    parameter_selection={"model_schema":MODEL_SCHEMA,"solver_schema":SOLVER_SCHEMA,"result_schema":RESULT_SCHEMA,
                         "preflight_pass":True,"convergence":{},"runtime":{},"pass":True}
    rep_wo=15.0 if 15.0 in cfg["COARSE_WO"] else float(cfg["COARSE_WO"][len(cfg["COARSE_WO"])//2])
    for record in _demanding_record_per_class(pf,cfg):
        cls=record["case_class"]; cid=record["case_id"]
        template=case_record_to_spec(record,rep_wo,N=cfg["STUDY_N"],dt=cfg["STUDY_DT"],T_final=cfg["SELECTION_T_FINAL"],cfg=cfg)
        parameter_selection["runtime"][cls]=estimate_runtime(template,benchmark_steps=100)
        spatial=spatial_convergence(template,cfg["SELECTION_N_VALUES"],progress=progress)
        temporal=temporal_convergence(template,cfg["SELECTION_DT_VALUES"],progress=progress)
        Nacc=convergence_acceptance(spatial["rows"],"N",cfg["I2_CONVERGENCE_TOL"],cfg["OBSERVABLE_CONVERGENCE_TOL"])
        dtacc=convergence_acceptance(temporal["rows"],"dt",cfg["I2_CONVERGENCE_TOL"],cfg["OBSERVABLE_CONVERGENCE_TOL"])
        parameter_selection["convergence"][cls]={"case_id":cid,"spatial":spatial,"temporal":temporal,"accepted_N":Nacc,"accepted_dt":dtacc}
        if Nacc is None or dtacc is None: parameter_selection["pass"]=False
    if parameter_selection["pass"]:
        parameter_selection["recommended_N"]=int(max(v["accepted_N"] for v in parameter_selection["convergence"].values()))
        parameter_selection["recommended_dt"]=float(min(v["accepted_dt"] for v in parameter_selection["convergence"].values()))
    atomic_write_json(paths.verification/"PARAMETER_SELECTION_REPORT.json",parameter_selection); return parameter_selection


def verification_status(paths):
    f=paths.verification/"VERIFICATION_STATUS.json"
    return {"pass":False,"reason":"VERIFICATION_STATUS.json not found"} if not f.exists() else json.loads(f.read_text())


def parameter_selection_status(paths):
    f=paths.verification/"PARAMETER_SELECTION_REPORT.json"
    return {"pass":False,"reason":"PARAMETER_SELECTION_REPORT.json not found"} if not f.exists() else json.loads(f.read_text())


def _status_compatible(status,kind):
    if not status.get("pass",False): return False
    if kind=="verification":
        return (status.get("schemas",{}).get("model")==MODEL_SCHEMA and status.get("schemas",{}).get("solver")==SOLVER_SCHEMA
                and status.get("parent_reference_schema")==PARENT_REFERENCE_SCHEMA)
    if kind=="parameter_selection":
        return status.get("model_schema")==MODEL_SCHEMA and status.get("solver_schema")==SOLVER_SCHEMA
    return False


def assert_full_study_ready(paths):
    vg=verification_status(paths); pg=parameter_selection_status(paths)
    if not _status_compatible(vg,"verification"): raise RuntimeError("Full study blocked: compatible verification status is not PASS.")
    if not _status_compatible(pg,"parameter_selection"): raise RuntimeError("Full study blocked: compatible parameter-selection status is not PASS.")
    return vg,pg


def _record_by_id(case_id,cfg=STUDY_CONFIG):
    for record in cfg["DISEASE_CASES"]:
        if str(record["case_id"])==str(case_id): return record
    raise KeyError(case_id)


def run_resonance_case(case_id, paths, cfg=STUDY_CONFIG, progress=True):
    assert_full_study_ready(paths); record=_record_by_id(case_id,cfg); Wo_values=list(cfg["COARSE_WO"]); rows_by_wo={}
    for round_idx in range(cfg["MAX_REFINE_ROUNDS"]+1):
        for wo in sorted(Wo_values):
            if wo in rows_by_wo: continue
            prep=prepare_case(case_record_to_spec(record,wo,cfg=cfg)); pair=run_paired_case(prep,paths=paths,resume=True,progress=progress)
            rows_by_wo[wo]={"study_case_id":case_id,"case_class":record["case_class"],"chi_b":float(record["chi_b"]),"chi_g":float(record["chi_g"]),
                            "Wo":wo,**pair["summary"]}
        xs=sorted(rows_by_wo); ys=[rows_by_wo[x]["R_max_het"] for x in xs]
        prop=propose_refinement_points(xs,ys,cfg["REFINE_MIN_SPACING"]); new=[float(v) for v in prop["new_Wo"] if v not in rows_by_wo]
        if not new or round_idx==cfg["MAX_REFINE_ROUNDS"]: break
        Wo_values.extend(new)
    df=pd.DataFrame([rows_by_wo[x] for x in sorted(rows_by_wo)]); desc=resonance_descriptors(df["Wo"],df["R_max_het"]); desc["study_case_id"]=case_id
    return df,desc


def run_primary_study(paths,cfg=STUDY_CONFIG,progress=True):
    assert_full_study_ready(paths); all_rows=[]; descriptors={}
    for record in cfg["DISEASE_CASES"]:
        cid=record["case_id"]; df,desc=run_resonance_case(cid,paths,cfg,progress); all_rows.append(df); descriptors[cid]=desc
    full=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
    tmp=paths.tables/"primary_resonance.csv.tmp"; full.to_csv(tmp,index=False); os.replace(tmp,paths.tables/"primary_resonance.csv")
    atomic_write_json(paths.tables/"primary_resonance_descriptors.json",descriptors); return full,descriptors


def choose_mechanism_cases(primary_df,paths):
    rows=[]
    for cls in sorted(primary_df.case_class.unique()):
        sub=primary_df[primary_df.case_class==cls].copy(); sub["abs_delta"]=np.abs(sub["Delta_R_maxima"])
        row=sub.loc[sub.abs_delta.idxmax()]
        rows.append({"case_class":cls,"study_case_id":row.study_case_id,"Wo":float(row.Wo),
                     "selection_metric":"max_abs_Rmax_het_minus_Rmax_mm","selection_value":float(row.abs_delta),"source_case_id":row.case_id})
    sel=pd.DataFrame(rows); tmp=paths.tables/"mechanism_selection.csv.tmp"; sel.to_csv(tmp,index=False); os.replace(tmp,paths.tables/"mechanism_selection.csv"); return sel


def run_selected_mechanism_cases(primary_df,paths,cfg=STUDY_CONFIG,progress=True):
    sel=choose_mechanism_cases(primary_df,paths); outputs=[]
    for _,row in sel.iterrows():
        record=_record_by_id(row.study_case_id,cfg); prep=prepare_case(case_record_to_spec(record,row.Wo,mechanism=True,cfg=cfg))
        res=run_paired_case(prep,paths=paths,resume=True,progress=progress)
        outputs.append({"study_case_id":row.study_case_id,"Wo":float(row.Wo),"case_id":res["summary"]["case_id"],"summary":res["summary"]})
    atomic_write_json(paths.tables/"mechanism_runs.json",outputs); return sel,outputs


def run_morphology_scale_study(primary_df,paths,cfg=STUDY_CONFIG,progress=True):
    # Automatic rescaling is defined only for analytical DL/DM morphologies. DR
    # geometry-derived fields must be supplied as new morphology inputs, not warped silently.
    factors=tuple(float(x) for x in cfg.get("SCALE_FACTORS",()))
    if not factors: return {}
    out={}
    for record in cfg["DISEASE_CASES"]:
        if record["case_class"] not in {"DL","DM"}: continue
        cid=record["case_id"]; sub=primary_df[primary_df.study_case_id==cid]
        if sub.empty: continue
        Wo=float(sub.loc[sub.R_max_het.idxmax(),"Wo"]); rows=[]
        for factor in factors:
            rr=deepcopy(record)
            if rr["case_class"]=="DL": rr["w"]=float(rr["w"])*factor
            else:
                rr["lesions"]=[{**dict(x),"w":float(dict(x)["w"])*factor} for x in rr["lesions"]]
            prep=prepare_case(case_record_to_spec(rr,Wo,cfg=cfg))
            if prep.admissibility["status"]!="ADMISSIBLE": rows.append({"study_case_id":cid,"factor":factor,"status":prep.admissibility["status"]}); continue
            pair=run_paired_case(prep,paths=paths,resume=True,progress=progress)
            rows.append({"study_case_id":cid,"factor":factor,"status":"ADMISSIBLE",**pair["summary"]})
        df=pd.DataFrame(rows); out[cid]=df; target=paths.tables/f"morphology_scale_{cid}.csv"; tmp=Path(str(target)+".tmp"); df.to_csv(tmp,index=False); os.replace(tmp,target)
    return out


def ensure_verification(paths,cfg=STUDY_CONFIG,progress=True):
    status=verification_status(paths)
    if _status_compatible(status,"verification"): print("VERIFICATION: reusing compatible PASS status."); return status
    print("VERIFICATION: running core solver verification..."); core=full_verification_suite(); atomic_write_json(paths.verification/"CORE_VERIFICATION.json",core)
    if not core["pass"]: raise RuntimeError("Core verification failed; study stopped.")
    print("VERIFICATION: running Solver Design parent reference audit...")
    parent=run_parent_reference_audit(paths=paths,N=512,dt=2e-4,T_final=60.0,progress=progress); atomic_write_json(paths.verification/"PARENT_REFERENCE_AUDIT.json",parent)
    detailed_ok=True
    if cfg["RUN_PARENT_DETAILED"]:
        detailed=run_parent_detailed_case(paths=paths,progress=progress); detailed_ok=bool(detailed["summary"]["runtime_valid"])
        atomic_write_json(paths.verification/"PARENT_DETAILED_SUMMARY.json",detailed["summary"])
    parent_baseline_pass=bool(parent["solver_design_parent_baseline"]["pass_numerical"]); legacy_topology_match=bool(parent["legacy_reference_audit"]["topology_match"])
    status={"pass":bool(core["pass"] and parent_baseline_pass and detailed_ok),"core_pass":bool(core["pass"]),
            "parent_baseline_pass":parent_baseline_pass,"legacy_parent_topology_match":legacy_topology_match,
            "legacy_parent_topology_is_acceptance_criterion":False,"parent_detailed_required":bool(cfg["RUN_PARENT_DETAILED"]),
            "parent_detailed_pass":bool(detailed_ok),"parent_reference_schema":PARENT_REFERENCE_SCHEMA,
            "schemas":{"model":MODEL_SCHEMA,"solver":SOLVER_SCHEMA,"results":RESULT_SCHEMA}}
    atomic_write_json(paths.verification/"VERIFICATION_STATUS.json",status)
    if not status["pass"]: raise RuntimeError("VERIFICATION status failed; study stopped.")
    return status


def ensure_parameter_selection(paths,cfg=STUDY_CONFIG,progress=True):
    pg=parameter_selection_status(paths)
    if _status_compatible(pg,"parameter_selection"):
        cfg["STUDY_N"]=int(pg["recommended_N"]); cfg["STUDY_DT"]=float(pg["recommended_dt"]); print("PARAMETER_SELECTION: reusing compatible PASS status."); return pg
    print("PARAMETER_SELECTION: Mathematical Model preflight and convergence..."); pg=run_parameter_selection(paths,cfg,progress=progress)
    if not pg["pass"]: raise RuntimeError("PARAMETER_SELECTION did not establish converged numerical settings for the main calculations.")
    cfg["STUDY_N"]=int(pg["recommended_N"]); cfg["STUDY_DT"]=float(pg["recommended_dt"]); return pg


def run_full_study(paths,cfg=STUDY_CONFIG,progress=True):
    validate_study_config(cfg,disease_required=True)
    atomic_write_json(paths.logs/"FULL_STUDY_START.json",{"run_mode":"FULL_STUDY","model_schema":MODEL_SCHEMA,"solver_schema":SOLVER_SCHEMA,"started_unix":time.time()})
    print("[1/7] QUICK_CHECK"); quick=quick_numerical_check(); atomic_write_json(paths.verification/"QUICK_CHECK_REPORT.json",quick)
    if not quick["pass"]: raise RuntimeError("QUICK_CHECK failed; full study stopped.")
    print("[2/7] VERIFICATION"); vg=ensure_verification(paths,cfg,progress)
    print("[3/7] PARAMETER_SELECTION + Mathematical Model preflight"); pg=ensure_parameter_selection(paths,cfg,progress)
    print(f"      study N={cfg['STUDY_N']}, dt={cfg['STUDY_DT']}")
    print("[4/7] Primary coefficient-morphology study"); primary,desc=run_primary_study(paths,cfg,progress)
    print("[5/7] Objective mechanism cases"); mech_sel,mech=run_selected_mechanism_cases(primary,paths,cfg,progress)
    print("[6/7] Morphology-scale study"); scale=run_morphology_scale_study(primary,paths,cfg,progress)
    print("[7/7] Publication figures")
    from .plotting import regenerate_available_figures
    figs=regenerate_available_figures(paths,cfg)
    study_summary={"pass":True,"configured_cases":[str(r["case_id"]) for r in cfg["DISEASE_CASES"]],"STUDY_N":cfg["STUDY_N"],"STUDY_DT":cfg["STUDY_DT"],
                   "model_schema":MODEL_SCHEMA,"solver_schema":SOLVER_SCHEMA,"verification":vg,"parameter_selection":pg,
                   "primary_rows":int(len(primary)),"mechanism_cases":int(len(mech_sel)),"scale_studies":list(scale),"figure_files":figs}
    atomic_write_json(paths.root/"FULL_STUDY_COMPLETED.json",study_summary); return study_summary


def run_study_mode(paths,cfg=STUDY_CONFIG,progress=True):
    mode=cfg["RUN_MODE"]; validate_study_config(cfg,disease_required=(mode in {"PARAMETER_SELECTION","FULL_STUDY"}))
    if mode=="QUICK_CHECK": return quick_numerical_check()
    if mode=="VERIFICATION": return ensure_verification(paths,cfg,progress)
    if mode=="PARAMETER_SELECTION": return ensure_parameter_selection(paths,cfg,progress)
    if mode=="FIGURES":
        from .plotting import regenerate_available_figures
        return {"figure_files":regenerate_available_figures(paths,cfg),"pass":True}
    if mode=="FULL_STUDY": return run_full_study(paths,cfg,progress=progress)
    raise ValueError(f"Unsupported RUN_MODE: {mode}")


__all__=[name for name in globals() if not name.startswith("_")]
