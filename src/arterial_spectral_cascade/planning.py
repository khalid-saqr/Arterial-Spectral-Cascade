from __future__ import annotations

from .core import *
from .storage import run_case
from .convergence import *
from . import convergence as _convergence

# Preserve the public monkeypatch/test surface while keeping convergence logic
# in its focused module.
def spatial_convergence(*args, **kwargs):
    _convergence.run_case = globals()["run_case"]
    return _convergence.spatial_convergence(*args, **kwargs)

def temporal_convergence(*args, **kwargs):
    _convergence.run_case = globals()["run_case"]
    return _convergence.temporal_convergence(*args, **kwargs)


def sensitivity_admissibility_scan(template: CaseSpec, chi_b_values: Sequence[float],
                                   chi_g_values: Sequence[float]) -> Dict[str,Any]:
    if template.case_class not in {"DL","DM","DR"}:
        raise ValueError("Sensitivity scans require a DL, DM, or DR disease template.")
    rows=[]
    for cb in chi_b_values:
        for cg in chi_g_values:
            prep=prepare_case(replace(template,chi_b=float(cb),chi_g=float(cg)))
            rows.append({"chi_b":float(cb),"chi_g":float(cg),"status":prep.admissibility["status"],
                         "b_heterogeneity":prep.admissibility["b_heterogeneity"],
                         "g_heterogeneity":prep.admissibility["g_heterogeneity"],
                         "morphology_error":prep.morphology_error,"coeff_error":prep.coeff_error})
    return {"rows":rows,"all_admissible":all(r["status"]=="ADMISSIBLE" for r in rows)}


def estimate_runtime(case: CaseSpec, benchmark_steps: int=200) -> Dict[str,float]:
    prep=prepare_case(case); ah=np.fft.fft(initial_condition(case,prep.grid)); etd=etd_coefficients(prep)
    n=min(benchmark_steps,int(round(case.T_final/case.dt))); t0=time.perf_counter()
    for _ in range(n): ah=etdrk4_step(ah,prep,etd)
    elapsed=time.perf_counter()-t0; sps=n/max(elapsed,1e-12); total=int(round(case.T_final/case.dt))
    return {"steps_per_second":sps,"estimated_seconds":total/sps,"benchmark_steps":n}


def propose_refinement_points(Wo: Sequence[float], response: Sequence[float], min_spacing: float=0.5) -> Dict[str,Any]:
    x=np.asarray(sorted(set(float(v) for v in Wo))); ymap={float(a):float(b) for a,b in zip(Wo,response)}; y=np.array([ymap[v] for v in x])
    topo=topology_from_curve(x,y); new=[]
    for pk in topo.get("peaks",[]):
        i=int(np.where(x==pk["Wo"])[0][0])
        if i>0 and x[i]-x[i-1]>min_spacing: new.append((x[i]+x[i-1])/2)
        if i<len(x)-1 and x[i+1]-x[i]>min_spacing: new.append((x[i]+x[i+1])/2)
    return {"topology":topo,"new_Wo":sorted(set(new))}


def resonance_descriptors(Wo: Sequence[float], Rmax: Sequence[float]) -> Dict[str,Any]:
    x=np.asarray(Wo,float); y=np.asarray(Rmax,float); idx=np.argsort(x); x=x[idx]; y=y[idx]
    topo=topology_from_curve(x,y); out={"topology":topo,"R_global_max":float(np.max(y)),"Wo_global_max":float(x[np.argmax(y)])}
    if topo["class"]=="single interior peak":
        p=topo["peaks"][0]; out["R_star"]=p["value"]; out["Wo_star"]=p["Wo"]; out["width"]=half_prominence_width(x,y,p["index"])
    elif topo["class"]=="multiple peaks": out["peaks"]=topo["peaks"]
    return out

__all__=[name for name in globals() if not name.startswith("_")]
