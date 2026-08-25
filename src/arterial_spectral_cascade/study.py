from __future__ import annotations

import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from .core import *
from .core import _jsonable
from .storage import *
from .planning import *
from .parent import *

# Notebook-level study orchestration. These functions do not change the Stage 1 model or Stage 2 solver.
import pandas as pd

# ---------------------------------------------------------------------------
# Evidence registry
# ---------------------------------------------------------------------------
EVIDENCE_REFERENCES = {
    "chen2024": {
        "citation": "Chen X, Cao H, Li Y, et al. Front Bioeng Biotechnol. 2024;12:1439846.",
        "doi": "10.3389/fbioe.2024.1439846",
        "url": "https://doi.org/10.3389/fbioe.2024.1439846",
        "role": "diameter-reduction stenosis levels; study modeled 10,20,30,40,50,60,70% DS",
    },
    "long2001": {
        "citation": "Long Q, Xu XY, Ramnarine KV, Hoskins P. J Biomech. 2001;34:1229-1242.",
        "doi": "10.1016/S0021-9290(01)00100-2",
        "url": "https://doi.org/10.1016/S0021-9290(01)00100-2",
        "role": "established idealized pulsatile stenosis precedent; 25,50,75% area reduction",
    },
    "ullery2018": {
        "citation": "Ullery BW, Hallett RL, Fleischmann D. Abdom Radiol. 2018;43:1032-1043.",
        "doi": "10.1007/s00261-017-1450-7",
        "url": "https://doi.org/10.1007/s00261-017-1450-7",
        "role": "AAA definition context: focal dilation >=1.5 times adjacent normal diameter; amplitude anchor only",
    },
    "abdelhamid2025": {
        "citation": "Abdelhamid T, Rahma AG. Discov Appl Sci. 2025;7:1367.",
        "doi": "10.1007/s42452-024-05993-0",
        "url": "https://doi.org/10.1007/s42452-024-05993-0",
        "role": "idealized fusiform benchmark dilation ratios 1.0,1.2,1.6,2.0,2.4,2.6",
    },
}

# Full-resolution profiles use fixed amplitude definitions from the literature.  Labels are
# study identifiers, not universal clinical risk categories.
EVIDENCE_PROFILES = {
    "S10": {"case_class":"DS","sigma":0.10,"severity_measure":"diameter_reduction_fraction","severity_value":0.10,
            "display_value":10.0,"display_unit":"% diameter stenosis","label":"S10","representation_name":"smooth distributed narrowing","source":"chen2024"},
    "S20": {"case_class":"DS","sigma":0.20,"severity_measure":"diameter_reduction_fraction","severity_value":0.20,
            "display_value":20.0,"display_unit":"% diameter stenosis","label":"S20","representation_name":"smooth distributed narrowing","source":"chen2024"},
    "S30": {"case_class":"DS","sigma":0.30,"severity_measure":"diameter_reduction_fraction","severity_value":0.30,
            "display_value":30.0,"display_unit":"% diameter stenosis","label":"S30","representation_name":"smooth distributed narrowing","source":"chen2024"},
    "D20": {"case_class":"DA","sigma":0.20,"severity_measure":"maximum_diameter_ratio","severity_value":1.20,
            "display_value":1.20,"display_unit":"Dmax/D0","label":"D20","representation_name":"smooth distributed dilation","source":"abdelhamid2025"},
    "D50": {"case_class":"DA","sigma":0.50,"severity_measure":"maximum_diameter_ratio","severity_value":1.50,
            "display_value":1.50,"display_unit":"Dmax/D0","label":"D50","representation_name":"smooth distributed dilation","source":"ullery2018"},
    "D60": {"case_class":"DA","sigma":0.60,"severity_measure":"maximum_diameter_ratio","severity_value":1.60,
            "display_value":1.60,"display_unit":"Dmax/D0","label":"D60","representation_name":"smooth distributed dilation","source":"abdelhamid2025"},
}
PRIMARY_PROFILE_IDS = ("S10","S20","S30","D20","D50","D60")
WIDTH_PROFILE_IDS = {"DS":"S20","DA":"D50"}

STUDY_CONFIG = {
    # Environment variable permits local/quick testing without changing the artifact.
    # In Google Colab the default is therefore FULL_STUDY.
    "RUN_MODE": os.environ.get("POF_RUN_MODE", "FULL_STUDY"),
    "MOUNT_DRIVE": True,
    "PROJECT_ROOT": None,                 # None -> MyDrive/PoF_DiseaseCascade in Colab.

    # Reduced-order validity bookkeeping. These are declared model-study scalings, not
    # clinical severity evidence and they do not alter b(r), g(r), or the PDE evolution.
    "R0_OVER_L0": 0.05,
    "SLOW_VARIATION_LIMIT": 0.10,

    # Fixed morphology for the primary severity comparison is selected automatically.
    "PRIMARY_WIDTH": None,
    "PRIMARY_WIDTH_CANDIDATES": (2.5,3.0,3.5,4.0,4.25,4.5,5.0,6.0),
    "HETEROGENEITY_DESIGN_FRACTION": 0.95,  # 5% headroom below Stage-1 0.30 ceiling
    "PRIMARY_P": 1,
    "XI_C": 2*np.pi,
    "WIDTH_CANDIDATES": (3.5,4.25,5.0,6.0),

    # Coarse disease resonance grid is the parent-published Wo set; local refinement
    # prevents the study from assuming the disease resonance remains at Wo=15.
    "COARSE_WO": (2.0,5.0,10.0,15.0,20.0),
    "REFINE_MIN_SPACING": 0.625,
    "MAX_REFINE_ROUNDS": 2,

    # PARAMETER_SELECTION replaces these initial numerical settings with the coarsest verified N and largest
    # verified dt common to the highest evidence grade in each branch.
    "STUDY_N": 512,
    "STUDY_DT": 2e-4,
    "STUDY_T_FINAL": 60.0,
    "OUTPUT_INTERVAL": 0.05,
    "CHECKPOINT_INTERVAL": 2.0,
    "COEFF_PROJECTION_LIMIT": 1e-8,

    "RUN_PARENT_DETAILED": False,
    "SELECTION_N_VALUES": (256,512,1024),
    "SELECTION_DT_VALUES": (8e-4,4e-4,2e-4),
    "SELECTION_T_FINAL": 60.0,
    "I2_CONVERGENCE_TOL": 1e-5,
    "OBSERVABLE_CONVERGENCE_TOL": 1e-3,
}

ALLOWED_MODES={"QUICK_CHECK","VERIFICATION","PARAMETER_SELECTION","FIGURES","FULL_STUDY"}


def validate_study_config(cfg=STUDY_CONFIG, disease_required=False):
    if cfg["RUN_MODE"] not in ALLOWED_MODES:
        raise ValueError(f"RUN_MODE must be one of {sorted(ALLOWED_MODES)}")
    if disease_required:
        missing=[k for k in ["R0_OVER_L0","SLOW_VARIATION_LIMIT"] if cfg.get(k) is None]
        if missing:
            raise RuntimeError("Disease calculations require explicit reduced-order validity inputs: "+", ".join(missing))
    if not (0 < cfg["HETEROGENEITY_DESIGN_FRACTION"] <= 1):
        raise ValueError("HETEROGENEITY_DESIGN_FRACTION must lie in (0,1].")
    return True


def configured_root(cfg=STUDY_CONFIG):
    if cfg.get("PROJECT_ROOT"):
        return Path(cfg["PROJECT_ROOT"]).expanduser()
    if "google.colab" in sys.modules:
        return Path("/content/drive/MyDrive/PoF_ArterialSpectralCascade")
    return Path.cwd()/"PoF_ArterialSpectralCascade_local"


def output_stride(dt, cfg=STUDY_CONFIG):
    return max(1,int(round(cfg["OUTPUT_INTERVAL"]/dt)))


def checkpoint_stride(dt, cfg=STUDY_CONFIG):
    return max(1,int(round(cfg["CHECKPOINT_INTERVAL"]/dt)))


def evidence_profile_table() -> pd.DataFrame:
    rows=[]
    for pid in PRIMARY_PROFILE_IDS:
        p=EVIDENCE_PROFILES[pid]; ref=EVIDENCE_REFERENCES[p["source"]]
        sigma=float(p["sigma"])
        if p["case_class"]=="DS":
            diameter_ratio=1-sigma; area_ratio=diameter_ratio**2
        else:
            diameter_ratio=1+sigma; area_ratio=diameter_ratio**2
        rows.append({"profile_id":pid,"case_class":p["case_class"],"sigma":sigma,
                     "severity_measure":p["severity_measure"],"severity_value":p["severity_value"],
                     "display_value":p["display_value"],"display_unit":p["display_unit"],"representation_name":p["representation_name"],
                     "throat_or_peak_diameter_ratio":diameter_ratio,"throat_or_peak_area_ratio":area_ratio,
                     "source_key":p["source"],"source_doi":ref["doi"],"source_url":ref.get("url",""),"source_citation":ref["citation"]})
    return pd.DataFrame(rows)


def _profile_metadata(profile_id):
    if profile_id not in EVIDENCE_PROFILES: raise KeyError(f"Unknown evidence profile {profile_id}")
    p=EVIDENCE_PROFILES[profile_id]; ref=EVIDENCE_REFERENCES[p["source"]]
    return p,ref


def disease_spec(case_class, Wo, sigma, width=None, N=None, dt=None, T_final=None, mechanism=False,
                 cfg=STUDY_CONFIG, profile_id="", severity_measure="", severity_value=None,
                 evidence_source="", evidence_doi=""):
    validate_study_config(cfg,disease_required=True)
    if case_class not in {"DS","DA"}: raise ValueError("Primary disease_spec supports DS or DA.")
    width=cfg.get("PRIMARY_WIDTH") if width is None else width
    if width is None: raise RuntimeError("Primary width has not yet been resolved by evidence-profile preflight.")
    N=cfg["STUDY_N"] if N is None else int(N); dt=cfg["STUDY_DT"] if dt is None else float(dt); T_final=cfg["STUDY_T_FINAL"] if T_final is None else float(T_final)
    return CaseSpec(case_class,Wo0=float(Wo),N=N,dt=dt,T_final=T_final,k0=1.0,sigma=float(sigma),xi_c=float(cfg["XI_C"]),w=float(width),p=int(cfg["PRIMARY_P"]),
                    eps_b=0.0,eps_g=0.0,q=1.0,output_every_steps=output_stride(dt,cfg),checkpoint_every_steps=checkpoint_stride(dt,cfg),mechanism=bool(mechanism),
                    R0_over_L0=float(cfg["R0_OVER_L0"]),slow_variation_limit=float(cfg["SLOW_VARIATION_LIMIT"]),coeff_projection_limit=float(cfg["COEFF_PROJECTION_LIMIT"]),
                    profile_id=str(profile_id),severity_measure=str(severity_measure),severity_value=severity_value,
                    evidence_source=str(evidence_source),evidence_doi=str(evidence_doi),
                    notes=(f"Evidence profile {profile_id}" if profile_id else ""))


def profile_spec(profile_id, Wo, width=None, N=None, dt=None, T_final=None, mechanism=False, cfg=STUDY_CONFIG):
    p,ref=_profile_metadata(profile_id)
    return disease_spec(p["case_class"],Wo,p["sigma"],width=width,N=N,dt=dt,T_final=T_final,mechanism=mechanism,cfg=cfg,
                        profile_id=profile_id,severity_measure=p["severity_measure"],severity_value=p["severity_value"],
                        evidence_source=ref["citation"],evidence_doi=ref["doi"])


def resolve_primary_width(cfg=STUDY_CONFIG, N=512):
    """Choose the smallest common admissible fixed width with numerical headroom."""
    validate_study_config(cfg,disease_required=True)
    reports=[]; design_limit=CONSTS.heterogeneity_limit*cfg["HETEROGENEITY_DESIGN_FRACTION"]
    for w in cfg["PRIMARY_WIDTH_CANDIDATES"]:
        all_ok=True; worst=0.0; details=[]
        for pid in PRIMARY_PROFILE_IDS:
            for wo in cfg["COARSE_WO"]:
                p,ref=_profile_metadata(pid)
                sp=disease_spec(p["case_class"],wo,p["sigma"],width=float(w),N=N,dt=cfg["STUDY_DT"],T_final=cfg["STUDY_T_FINAL"],cfg=cfg,
                                profile_id=pid,severity_measure=p["severity_measure"],severity_value=p["severity_value"],evidence_source=ref["citation"],evidence_doi=ref["doi"])
                prep=prepare_case(sp); het=max(prep.admissibility["b_heterogeneity"],prep.admissibility["g_heterogeneity"])
                worst=max(worst,float(het))
                good=(prep.admissibility["status"]=="ADMISSIBLE" and het<=design_limit+1e-14)
                H=lesion_kernel(prep.grid.xi,prep.spec.xi_c,prep.spec.w,prep.spec.p,prep.spec.Lg)
                hmin=float(np.min(H)); far_dev=float(abs(prep.spec.sigma)*hmin)
                details.append({"profile_id":pid,"Wo":wo,"status":prep.admissibility["status"],"heterogeneity":float(het),"coeff_error":prep.coeff_error,
                                "kernel_floor":hmin,"far_field_radius_deviation":far_dev})
                all_ok &= good
        reports.append({"w":float(w),"all_ok_with_headroom":bool(all_ok),"worst_heterogeneity":worst,"details":details})
        if all_ok:
            return {"width":float(w),"design_limit":design_limit,"reports":reports}
    raise RuntimeError("No PRIMARY_WIDTH_CANDIDATE makes all evidence profiles Stage-1 admissible with the requested design headroom.")


def preflight_evidence_profiles(paths, cfg=STUDY_CONFIG):
    resolved=resolve_primary_width(cfg,N=max(512,min(cfg["SELECTION_N_VALUES"])))
    cfg["PRIMARY_WIDTH"]=float(resolved["width"])
    df=evidence_profile_table(); rows=[]
    for _,base in df.iterrows():
        pid=base.profile_id
        for wo in cfg["COARSE_WO"]:
            prep=prepare_case(profile_spec(pid,wo,width=cfg["PRIMARY_WIDTH"],N=512,dt=cfg["STUDY_DT"],T_final=cfg["STUDY_T_FINAL"],cfg=cfg))
            ar=prep.admissibility
            H=lesion_kernel(prep.grid.xi,prep.spec.xi_c,prep.spec.w,prep.spec.p,prep.spec.Lg)
            hmin=float(np.min(H)); far_dev=float(abs(prep.spec.sigma)*hmin)
            rows.append({**base.to_dict(),"Wo":float(wo),"primary_width":cfg["PRIMARY_WIDTH"],"status":ar["status"],
                         "b_heterogeneity":ar["b_heterogeneity"],"g_heterogeneity":ar["g_heterogeneity"],"coeff_error":prep.coeff_error,
                         "r_min":ar["r_min"],"r_max":ar["r_max"],"kernel_floor":hmin,"far_field_radius_deviation":far_dev,
                         "R0_over_ellD":ar["slow_variation"]["R0_over_ellD"],"max_abs_dRdx":ar["slow_variation"]["max_abs_dRdx"]})
    pf=pd.DataFrame(rows)
    tmp=paths.tables/"evidence_profile_preflight.csv.tmp"; pf.to_csv(tmp,index=False); os.replace(tmp,paths.tables/"evidence_profile_preflight.csv")
    tmp2=paths.tables/"evidence_profile_definitions.csv.tmp"; df.to_csv(tmp2,index=False); os.replace(tmp2,paths.tables/"evidence_profile_definitions.csv")
    refs=pd.DataFrame([{"source_key":k,**v} for k,v in EVIDENCE_REFERENCES.items()])
    tmp3=paths.tables/"evidence_references.csv.tmp"; refs.to_csv(tmp3,index=False); os.replace(tmp3,paths.tables/"evidence_references.csv")
    atomic_write_json(paths.verification/"EVIDENCE_PREFLIGHT.json",{"pass":bool((pf.status=="ADMISSIBLE").all()),"primary_width":cfg["PRIMARY_WIDTH"],
                      "design_heterogeneity_limit":resolved["design_limit"],"width_scan":resolved["reports"],"profile_schema":RESULT_SCHEMA})
    if not (pf.status=="ADMISSIBLE").all():
        raise RuntimeError("At least one evidence profile failed Stage-1 preflight at the selected fixed width.")
    return pf,resolved


def convergence_acceptance(rows, x_name, i2tol, obstol):
    valid=[r for r in rows if r.get("status")=="ADMISSIBLE"]
    for i in range(1,len(valid)):
        r=valid[i]
        if r.get("rel_I2_change_vs_prev",np.inf)<i2tol and r.get("rel_Rmax_change_vs_prev",np.inf)<obstol:
            return valid[i-1][x_name]
    return None


def run_parameter_selection(paths, cfg=STUDY_CONFIG, progress=True):
    validate_study_config(cfg,disease_required=True)
    pf,resolved=preflight_evidence_profiles(paths,cfg)
    parameter_selection={"evidence_profile_schema":RESULT_SCHEMA,"primary_width":cfg["PRIMARY_WIDTH"],"preflight_pass":True,"convergence":{},"runtime":{},"pass":True}
    hardest={"DS":"S30","DA":"D60"}
    for cls,pid in hardest.items():
        template=profile_spec(pid,15.0,width=cfg["PRIMARY_WIDTH"],N=cfg["STUDY_N"],dt=cfg["STUDY_DT"],T_final=cfg["SELECTION_T_FINAL"],cfg=cfg)
        parameter_selection["runtime"][cls]=estimate_runtime(template,benchmark_steps=100)
        spatial=spatial_convergence(template,cfg["SELECTION_N_VALUES"],progress=progress)
        temporal=temporal_convergence(template,cfg["SELECTION_DT_VALUES"],progress=progress)
        Nacc=convergence_acceptance(spatial["rows"],"N",cfg["I2_CONVERGENCE_TOL"],cfg["OBSERVABLE_CONVERGENCE_TOL"])
        dtacc=convergence_acceptance(temporal["rows"],"dt",cfg["I2_CONVERGENCE_TOL"],cfg["OBSERVABLE_CONVERGENCE_TOL"])
        parameter_selection["convergence"][cls]={"profile_id":pid,"spatial":spatial,"temporal":temporal,"accepted_N":Nacc,"accepted_dt":dtacc}
        if Nacc is None or dtacc is None: parameter_selection["pass"]=False
    if parameter_selection["pass"]:
        parameter_selection["recommended_N"]=int(max(parameter_selection["convergence"]["DS"]["accepted_N"],parameter_selection["convergence"]["DA"]["accepted_N"]))
        parameter_selection["recommended_dt"]=float(min(parameter_selection["convergence"]["DS"]["accepted_dt"],parameter_selection["convergence"]["DA"]["accepted_dt"]))
    atomic_write_json(paths.verification/"PARAMETER_SELECTION_REPORT.json",parameter_selection)
    return parameter_selection


def verification_status(paths):
    f=paths.verification/"VERIFICATION_STATUS.json"
    if not f.exists(): return {"pass":False,"reason":"VERIFICATION_STATUS.json not found"}
    return json.loads(f.read_text())


def parameter_selection_status(paths):
    f=paths.verification/"PARAMETER_SELECTION_REPORT.json"
    if not f.exists(): return {"pass":False,"reason":"PARAMETER_SELECTION_REPORT.json not found"}
    return json.loads(f.read_text())


def assert_full_study_ready(paths):
    vg=verification_status(paths); pg=parameter_selection_status(paths)
    if not _status_compatible(vg,"verification"):
        raise RuntimeError("Full study blocked: compatible verification status is not PASS.")
    if not pg.get("pass",False):
        raise RuntimeError("Full study blocked: parameter-selection/convergence status is not PASS.")
    if pg.get("evidence_profile_schema")!=RESULT_SCHEMA:
        raise RuntimeError("Full study blocked: parameter-selection results belong to a different evidence/result schema.")
    return vg,pg


def run_resonance_profile(profile_id, paths, cfg=STUDY_CONFIG, progress=True):
    assert_full_study_ready(paths)
    p,_=_profile_metadata(profile_id); Wo_values=list(cfg["COARSE_WO"]); rows_by_wo={}
    for round_idx in range(cfg["MAX_REFINE_ROUNDS"]+1):
        for wo in sorted(Wo_values):
            if wo in rows_by_wo: continue
            prep=prepare_case(profile_spec(profile_id,wo,width=cfg["PRIMARY_WIDTH"],cfg=cfg))
            pair=run_paired_case(prep,paths=paths,resume=True,progress=progress)
            rows_by_wo[wo]={"profile_id":profile_id,"case_class":p["case_class"],"sigma":p["sigma"],"severity_measure":p["severity_measure"],
                            "severity_value":p["severity_value"],"severity_display":p["display_value"],"severity_unit":p["display_unit"],"Wo":wo,**pair["summary"]}
        xs=sorted(rows_by_wo); ys=[rows_by_wo[x]["R_max_het"] for x in xs]
        prop=propose_refinement_points(xs,ys,cfg["REFINE_MIN_SPACING"])
        new=[float(v) for v in prop["new_Wo"] if v not in rows_by_wo]
        if not new or round_idx==cfg["MAX_REFINE_ROUNDS"]: break
        Wo_values.extend(new)
    df=pd.DataFrame([rows_by_wo[x] for x in sorted(rows_by_wo)])
    desc=resonance_descriptors(df["Wo"],df["R_max_het"]); desc["profile_id"]=profile_id
    return df,desc


def run_primary_study(paths, cfg=STUDY_CONFIG, progress=True):
    vg,pg=assert_full_study_ready(paths)
    cfg["PRIMARY_WIDTH"]=float(pg["primary_width"])
    all_rows=[]; descriptors={}
    for pid in PRIMARY_PROFILE_IDS:
        df,desc=run_resonance_profile(pid,paths,cfg,progress)
        all_rows.append(df); descriptors[pid]=desc
    full=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
    tmp=paths.tables/"primary_resonance.csv.tmp"; full.to_csv(tmp,index=False); os.replace(tmp,paths.tables/"primary_resonance.csv")
    atomic_write_json(paths.tables/"primary_resonance_descriptors.json",descriptors)
    return full,descriptors


def choose_mechanism_cases(primary_df, paths):
    rows=[]
    for cls in ["DS","DA"]:
        sub=primary_df[primary_df.case_class==cls].copy()
        if sub.empty: continue
        sub["abs_delta"]=np.abs(sub["Delta_R_maxima"])
        row=sub.loc[sub.abs_delta.idxmax()]
        rows.append({"case_class":cls,"profile_id":row.profile_id,"Wo":float(row.Wo),"sigma":float(row.sigma),
                     "selection_metric":"max_abs_Rmax_het_minus_Rmax_mm","selection_value":float(row.abs_delta),"source_case_id":row.case_id})
    sel=pd.DataFrame(rows); tmp=paths.tables/"mechanism_selection.csv.tmp"; sel.to_csv(tmp,index=False); os.replace(tmp,paths.tables/"mechanism_selection.csv")
    return sel


def run_selected_mechanism_cases(primary_df, paths, cfg=STUDY_CONFIG, progress=True):
    sel=choose_mechanism_cases(primary_df,paths); outputs=[]
    for _,row in sel.iterrows():
        prep=prepare_case(profile_spec(row.profile_id,row.Wo,width=cfg["PRIMARY_WIDTH"],mechanism=True,cfg=cfg))
        res=run_paired_case(prep,paths=paths,resume=True,progress=progress)
        outputs.append({"profile_id":row.profile_id,"Wo":float(row.Wo),"case_id":res["summary"]["case_id"],"summary":res["summary"]})
    atomic_write_json(paths.tables/"mechanism_runs.json",outputs)
    return sel,outputs


def representative_Wo_for_profile(primary_df, profile_id):
    sub=primary_df[primary_df.profile_id==profile_id]
    if sub.empty: raise RuntimeError(f"No primary results for {profile_id}")
    return float(sub.loc[sub.R_max_het.idxmax(),"Wo"])


def run_width_study_profile(profile_id, widths, primary_df, paths, cfg=STUDY_CONFIG, progress=True):
    assert_full_study_ready(paths); p,_=_profile_metadata(profile_id); Wo=representative_Wo_for_profile(primary_df,profile_id); rows=[]
    for w in widths:
        prep=prepare_case(profile_spec(profile_id,Wo,width=float(w),cfg=cfg))
        if prep.admissibility["status"]!="ADMISSIBLE":
            rows.append({"profile_id":profile_id,"case_class":p["case_class"],"Wo":Wo,"w":float(w),"status":prep.admissibility["status"]}); continue
        pair=run_paired_case(prep,paths=paths,resume=True,progress=progress)
        bh=np.fft.fft(prep.b_tilde)/prep.grid.N; gh=np.fft.fft(prep.g_tilde)/prep.grid.N
        weight=np.abs(bh)**2+np.abs(gh)**2; bandwidth=float(np.sum(np.abs(prep.grid.k)*weight)/max(np.sum(weight),1e-30))
        rows.append({"profile_id":profile_id,"case_class":p["case_class"],"Wo":Wo,"w":float(w),"status":"ADMISSIBLE","coupling_bandwidth":bandwidth,**pair["summary"]})
    df=pd.DataFrame(rows); out=paths.tables/f"width_{p['case_class']}.csv"; tmp=Path(str(out)+".tmp"); df.to_csv(tmp,index=False); os.replace(tmp,out); return df


def run_axial_scale_study(primary_df, paths, cfg=STUDY_CONFIG, progress=True):
    out={}
    for cls,pid in WIDTH_PROFILE_IDS.items():
        out[cls]=run_width_study_profile(pid,cfg["WIDTH_CANDIDATES"],primary_df,paths,cfg,progress)
    return out

# ---------------------------------------------------------------------------
# Figure layer: all paper figures are regenerated from persisted validated data.
# ---------------------------------------------------------------------------
def _status_compatible(status, kind):
    if not status.get("pass",False): return False
    if kind=="verification":
        return (
            status.get("schemas",{}).get("model")==MODEL_SCHEMA
            and status.get("schemas",{}).get("solver")==SOLVER_SCHEMA
            and status.get("parent_reference_schema")==PARENT_REFERENCE_SCHEMA
        )
    if kind=="parameter_selection":
        return status.get("evidence_profile_schema")==RESULT_SCHEMA
    return False


def ensure_verification(paths, cfg=STUDY_CONFIG, progress=True):
    status=verification_status(paths)
    if _status_compatible(status,"verification"):
        print("VERIFICATION: reusing compatible PASS status."); return status

    print("VERIFICATION: running core solver verification...")
    core=full_verification_suite()
    atomic_write_json(paths.verification/"CORE_VERIFICATION.json",core)
    if not core["pass"]:
        raise RuntimeError("Core verification failed; study stopped.")

    print("VERIFICATION: running Stage-2 parent reference audit...")
    parent=run_parent_reference_audit(paths=paths,N=512,dt=2e-4,T_final=60.0,progress=progress)
    atomic_write_json(paths.verification/"PARENT_REFERENCE_AUDIT.json",parent)

    detailed_ok=True
    if cfg["RUN_PARENT_DETAILED"]:
        detailed=run_parent_detailed_case(paths=paths,progress=progress)
        detailed_ok=bool(detailed["summary"]["runtime_valid"])
        atomic_write_json(paths.verification/"PARENT_DETAILED_SUMMARY.json",detailed["summary"])

    parent_baseline_pass=bool(parent["stage2_parent_baseline"]["pass_numerical"])
    legacy_topology_match=bool(parent["legacy_reference_audit"]["topology_match"])

    status={
        "pass":bool(core["pass"] and parent_baseline_pass and detailed_ok),
        "core_pass":bool(core["pass"]),
        "parent_baseline_pass":parent_baseline_pass,
        "legacy_parent_topology_match":legacy_topology_match,
        "legacy_parent_topology_is_acceptance_criterion":False,
        "parent_detailed_required":bool(cfg["RUN_PARENT_DETAILED"]),
        "parent_detailed_pass":bool(detailed_ok),
        "parent_reference_schema":PARENT_REFERENCE_SCHEMA,
        "schemas":{"model":MODEL_SCHEMA,"solver":SOLVER_SCHEMA,"results":RESULT_SCHEMA},
    }
    atomic_write_json(paths.verification/"VERIFICATION_STATUS.json",status)
    if not status["pass"]:
        raise RuntimeError("VERIFICATION status failed; study stopped.")
    return status


def ensure_parameter_selection(paths, cfg=STUDY_CONFIG, progress=True):
    pg=parameter_selection_status(paths)
    if _status_compatible(pg,"parameter_selection"):
        cfg["PRIMARY_WIDTH"]=float(pg["primary_width"]); cfg["STUDY_N"]=int(pg["recommended_N"]); cfg["STUDY_DT"]=float(pg["recommended_dt"])
        print("PARAMETER_SELECTION: reusing compatible PASS status."); return pg
    print("PARAMETER_SELECTION: evidence-profile preflight and convergence...")
    pg=run_parameter_selection(paths,cfg,progress=progress)
    if not pg["pass"]: raise RuntimeError("PARAMETER_SELECTION did not establish converged numerical settings for the main calculations.")
    cfg["PRIMARY_WIDTH"]=float(pg["primary_width"]); cfg["STUDY_N"]=int(pg["recommended_N"]); cfg["STUDY_DT"]=float(pg["recommended_dt"])
    return pg


def run_full_study(paths, cfg=STUDY_CONFIG, progress=True):
    validate_study_config(cfg,disease_required=True)
    atomic_write_json(paths.logs/"FULL_STUDY_START.json",{"run_mode":"FULL_STUDY","result_schema":RESULT_SCHEMA,"parent_reference_schema":PARENT_REFERENCE_SCHEMA,"started_unix":time.time()})
    print("[1/7] QUICK_CHECK")
    quick=quick_numerical_check(); atomic_write_json(paths.verification/"QUICK_CHECK_REPORT.json",quick)
    if not quick["pass"]: raise RuntimeError("QUICK_CHECK failed; full study stopped.")
    print("[2/7] VERIFICATION")
    vg=ensure_verification(paths,cfg,progress)
    print("[3/7] PARAMETER_SELECTION + evidence preflight")
    pg=ensure_parameter_selection(paths,cfg,progress)
    print(f"      primary width = {cfg['PRIMARY_WIDTH']}; study N={cfg['STUDY_N']}, dt={cfg['STUDY_DT']}")
    print("[4/7] R1-R3 primary evidence-severity study")
    primary,desc=run_primary_study(paths,cfg,progress)
    print("[5/7] R4 objective mechanism cases")
    mech_sel,mech_runs=run_selected_mechanism_cases(primary,paths,cfg,progress)
    print("[6/7] R5 axial-scale study")
    width=run_axial_scale_study(primary,paths,cfg,progress)
    print("[7/7] publication figures")
    from .plotting import regenerate_available_figures
    made=regenerate_available_figures(paths,cfg)
    study_summary={"pass":True,"profiles":list(PRIMARY_PROFILE_IDS),"primary_width":cfg["PRIMARY_WIDTH"],"STUDY_N":cfg["STUDY_N"],"STUDY_DT":cfg["STUDY_DT"],
              "verification_status":vg,"parameter_selection_pass":pg["pass"],"primary_rows":int(len(primary)),"mechanism_selection":_jsonable(mech_sel.to_dict(orient="records")),
              "figure_files":made,"completed_unix":time.time(),"schemas":{"model":MODEL_SCHEMA,"solver":SOLVER_SCHEMA,"results":RESULT_SCHEMA,"parent_reference":PARENT_REFERENCE_SCHEMA}}
    atomic_write_json(paths.root/"FULL_STUDY_COMPLETED.json",study_summary)
    print(f"FULL STUDY COMPLETE: {len(made)} figure files written to {paths.figures}")
    return study_summary


def run_study_mode(paths, cfg=STUDY_CONFIG, progress=True):
    """Execute the selected public study mode.

    `FULL_STUDY` is the default. `PARAMETER_SELECTION` first establishes a
    compatible verification status. `FIGURES` regenerates publication figures
    from persisted validated result archives without rerunning trajectories.
    """
    validate_study_config(cfg, disease_required=cfg["RUN_MODE"] in {"PARAMETER_SELECTION", "FULL_STUDY"})
    mode=cfg["RUN_MODE"]
    if mode=="QUICK_CHECK":
        report=quick_numerical_check()
        atomic_write_json(paths.verification/"QUICK_CHECK_REPORT.json",report)
        return report
    if mode=="VERIFICATION":
        return ensure_verification(paths,cfg,progress=progress)
    if mode=="PARAMETER_SELECTION":
        ensure_verification(paths,cfg,progress=progress)
        return ensure_parameter_selection(paths,cfg,progress=progress)
    if mode=="FIGURES":
        from .plotting import regenerate_available_figures
        return {"figure_files":regenerate_available_figures(paths,cfg),"pass":True}
    if mode=="FULL_STUDY":
        return run_full_study(paths,cfg,progress=progress)
    raise ValueError(f"Unsupported RUN_MODE: {mode}")
