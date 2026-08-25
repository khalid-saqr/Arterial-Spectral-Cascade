from __future__ import annotations

from .study_base import *
from . import study_base as _base
from .study_base import _demanding_record_per_class
from .core import *
from .storage import *
from .planning import *
from .parent import *

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
        parameter_selection["recommended_N"]=int(max(v["accepted_N"] for v in parameter_selection["convergence"].values()))
        parameter_selection["recommended_dt"]=float(min(v["accepted_dt"] for v in parameter_selection["convergence"].values()))
    atomic_write_json(paths.verification/"PARAMETER_SELECTION_REPORT.json",parameter_selection); return parameter_selection


# Patch the preserved orchestration module so its existing FULL_STUDY path resolves
# the Solver Design v2 paired runner and convergence implementation.
_base.run_parameter_selection = run_parameter_selection
_base.convergence_acceptance = convergence_acceptance
_base.run_paired_case = run_paired_case
_base.spatial_convergence = spatial_convergence
_base.temporal_convergence = temporal_convergence
_base.estimate_runtime = estimate_runtime

run_full_study = _base.run_full_study
run_study_mode = _base.run_study_mode

__all__ = [name for name in globals() if not name.startswith("_")]
