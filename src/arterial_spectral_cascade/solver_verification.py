from __future__ import annotations

from .core import *
from .storage import *
from .paired_runtime import run_paired_case, verify_paired_restart_equivalence
from .parent_base import run_parent_detailed_case

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
    required_scalar_keys=("R_max","s_R_max","I2_final","G_bal_max","eta_tail_max","chi_h_max",
                          "max_abs_RI1_inst","max_abs_RE_inst","max_abs_RI1_integrated","max_abs_RE_integrated")

    for spec in parent_sweep_specs(N,dt,T_final):
        res=run_case(prepare_case(spec),paths=paths,resume=True,progress=progress)
        summary=res["summary"]; hist=res["history"]
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
            "chi_h_max":float(summary["chi_h_max"]),
            "max_abs_RI1_inst":float(summary["max_abs_RI1_inst"]),
            "max_abs_RE_inst":float(summary["max_abs_RE_inst"]),
            "max_abs_RI1_integrated":float(summary["max_abs_RI1_integrated"]),
            "max_abs_RE_integrated":float(summary["max_abs_RE_integrated"]),
            "completed":completed,
            "runtime_valid":runtime_valid,
            "finite_diagnostics":finite,
            "I2_positive":I2_positive,
            "R_nonnegative":R_nonnegative,
            "pass_global_stability":global_stability,
            "pass_numerical":numerical_valid,
            "history":{"s":hist["s"].tolist(),"I2":hist["I2"].tolist(),"G":hist["G_bal"].tolist(),"R":hist["R"].tolist()},
            "final_spectrum_abs":np.abs(res["ahat_final"]/spec.N).tolist(),
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


def verify_matched_mean_control() -> Dict[str,Any]:
    spec=CaseSpec("DL",Wo0=10,N=64,dt=.002,T_final=.02,k0=.5,chi_b=.01,chi_g=-.005,w=3.0,p=1,
                  R0_over_L0=.01,slow_variation_limit=.1,output_every_steps=2,checkpoint_every_steps=10)
    parent=prepare_case(spec); mm=make_matched_mean(parent)
    mean_err=max(abs(mm.b_bar-parent.b_bar),abs(mm.g_bar-parent.g_bar))
    tilde=max(float(np.max(np.abs(mm.b_tilde))),float(np.max(np.abs(mm.g_tilde))))
    ic_err=float(np.linalg.norm(initial_condition(parent.spec,parent.grid)-initial_condition(mm.spec,mm.grid)))
    numeric_same=(parent.spec.N==mm.spec.N and parent.spec.dt==mm.spec.dt and parent.spec.T_final==mm.spec.T_final
                  and parent.spec.output_every_steps==mm.spec.output_every_steps and parent.spec.checkpoint_every_steps==mm.spec.checkpoint_every_steps)
    return {"name":"matched_mean_control","mean_error":float(mean_err),"tilde_max":tilde,"initial_condition_error":ic_err,
            "numerical_settings_identical":bool(numeric_same),"pass":bool(mean_err<1e-15 and tilde<1e-15 and ic_err<1e-15 and numeric_same)}


def verify_published_recovery_convergence() -> Dict[str,Any]:
    runs=[]
    for dt in (.01,.005):
        spec=CaseSpec("P0",Wo0=10,N=64,dt=dt,T_final=.1,k0=.5,output_every_steps=1,checkpoint_every_steps=1000)
        runs.append(run_case(prepare_case(spec),paths=None,resume=False,progress=False))
    coarse,fine=runs; sc=coarse["history"]["s"]; sf=fine["history"]["s"]
    errors={}
    for key in ("I2","G_bal","R"):
        ref=np.interp(sc,sf,fine["history"][key]); val=coarse["history"][key]
        errors[key]=float(np.max(np.abs(val-ref))/max(float(np.max(np.abs(ref))),1e-12))
    ac=coarse["ahat_final"]/coarse["prep"].grid.N; af=fine["ahat_final"]/fine["prep"].grid.N
    errors["spectrum"]=float(np.linalg.norm(ac-af)/max(np.linalg.norm(af),1e-30))
    passed=(errors["I2"]<1e-8 and errors["G_bal"]<1e-8 and errors["R"]<1e-5 and errors["spectrum"]<1e-8)
    return {"name":"published_recovery_convergence","comparison":"P0 constant-coefficient parent dynamics under timestep refinement",
            "errors":errors,"pass":bool(passed)}


def full_verification_suite() -> Dict[str,Any]:
    """Run the complete independent numerical verification suite."""
    report=core_verification_suite(True)
    report["tests"].append(verify_published_recovery_convergence())
    report["tests"].append(verify_matched_mean_control())
    report["tests"].append(verify_restart_equivalence())
    report["tests"].append(verify_paired_restart_equivalence())
    report["pass"]=all(t["pass"] for t in report["tests"])
    return report

__all__ = [name for name in globals() if not name.startswith("_")]
