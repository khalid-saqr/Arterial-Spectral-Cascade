from __future__ import annotations

from .study_base import *
from . import study_base as _base
from .study_base import _demanding_record_per_class
from .core import *
from .storage import *
from .planning import *
from .parent import *

_BASE_CASE_RECORD_TO_SPEC = _base.case_record_to_spec


def _class_numerical_settings(report):
    """Return verified per-morphology-class N/dt settings from parameter selection."""
    out={}
    for cls,entry in report.get("convergence",{}).items():
        N=entry.get("accepted_N"); dt=entry.get("accepted_dt")
        if N is not None and dt is not None:
            out[str(cls)]={"N":int(N),"dt":float(dt),"source_case_id":entry.get("case_id")}
    return out


def case_record_to_spec(record, Wo, N=None, dt=None, T_final=None, mechanism=False, cfg=STUDY_CONFIG):
    """Use class-specific converged settings when N/dt are not explicitly supplied.

    Explicit N/dt always win, which keeps preflight and convergence studies
    unchanged.  Main calculations can therefore use the coarsest resolution and
    largest timestep already demonstrated converged for their morphology class.
    """
    settings=cfg.get("_CLASS_NUMERICAL_SETTINGS",{})
    cls=str(record.get("case_class",""))
    if cls in settings:
        if N is None: N=int(settings[cls]["N"])
        if dt is None: dt=float(settings[cls]["dt"])
    return _BASE_CASE_RECORD_TO_SPEC(record,Wo,N=N,dt=dt,T_final=T_final,mechanism=mechanism,cfg=cfg)


def convergence_acceptance(rows, x_name, i2tol, obstol):
    """Compatibility wrapper for the Solver Design full-history acceptance function."""
    from .planning import convergence_acceptance as _accept
    return _accept(rows,x_name,i2tol,obstol,spatial=(x_name=="N"))


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
        Nacc=convergence_acceptance(spatial["rows"],"N",cfg["I2_CONVERGENCE_TOL"],cfg["OBSERVABLE_CONVERGENCE_TOL"])
        if Nacc is not None:
            temporal_template=replace(template,N=int(Nacc))
            temporal=temporal_convergence(temporal_template,cfg["SELECTION_DT_VALUES"],progress=progress)
            dtacc=convergence_acceptance(temporal["rows"],"dt",cfg["I2_CONVERGENCE_TOL"],cfg["OBSERVABLE_CONVERGENCE_TOL"])
        else:
            temporal={"rows":[],"paired":True,"criterion":"not run: no converged spatial resolution"}; dtacc=None
        parameter_selection["convergence"][cls]={"case_id":cid,"spatial":spatial,"temporal":temporal,"accepted_N":Nacc,"accepted_dt":dtacc}
        if Nacc is None or dtacc is None: parameter_selection["pass"]=False
    if parameter_selection["pass"]:
        settings=_class_numerical_settings(parameter_selection)
        parameter_selection["class_numerical_settings"]=settings
        # Conservative global values remain as fallbacks for code paths that do
        # not represent a disease morphology class (e.g. parent verification).
        parameter_selection["recommended_N"]=int(max(v["N"] for v in settings.values()))
        parameter_selection["recommended_dt"]=float(min(v["dt"] for v in settings.values()))
    atomic_write_json(paths.verification/"PARAMETER_SELECTION_REPORT.json",parameter_selection); return parameter_selection


def ensure_parameter_selection(paths,cfg=STUDY_CONFIG,progress=True):
    status=_base.parameter_selection_status(paths)
    if _base._status_compatible(status,"parameter_selection"):
        pg=status
        print("PARAMETER_SELECTION: reusing compatible PASS status.")
    else:
        print("PARAMETER_SELECTION: Mathematical Model preflight and convergence...")
        pg=run_parameter_selection(paths,cfg,progress=progress)
    if not pg.get("pass",False):
        raise RuntimeError("PARAMETER_SELECTION did not establish converged numerical settings for the main calculations.")
    settings=pg.get("class_numerical_settings") or _class_numerical_settings(pg)
    if not settings:
        raise RuntimeError("PARAMETER_SELECTION PASS contains no class-specific converged numerical settings.")
    cfg["_CLASS_NUMERICAL_SETTINGS"]=settings
    cfg["STUDY_N"]=int(pg.get("recommended_N",max(v["N"] for v in settings.values())))
    cfg["STUDY_DT"]=float(pg.get("recommended_dt",min(v["dt"] for v in settings.values())))
    return pg


# Patch the preserved orchestration module so its existing FULL_STUDY path resolves
# Solver Design v2 plus class-specific converged settings. Explicit convergence
# calls remain unchanged because they always pass N/dt explicitly.
_base.run_parameter_selection = run_parameter_selection
_base.ensure_parameter_selection = ensure_parameter_selection
_base.convergence_acceptance = convergence_acceptance
_base.case_record_to_spec = case_record_to_spec
_base.run_paired_case = run_paired_case
_base.spatial_convergence = spatial_convergence
_base.temporal_convergence = temporal_convergence
_base.estimate_runtime = estimate_runtime

run_full_study = _base.run_full_study
run_study_mode = _base.run_study_mode

__all__ = [name for name in globals() if not name.startswith("_")]
