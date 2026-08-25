from __future__ import annotations

from .core import *
from .storage import run_case

# Parameter selection and resonance-refinement planning
def find_sigma_admissibility_ceiling(template: CaseSpec, sigma_upper: float, iterations: int=32) -> Dict[str,Any]:
    if template.case_class not in {"DS","DA"}: raise ValueError("Template must be DS or DA.")
    if sigma_upper<=0: raise ValueError("sigma_upper is a search bound and must be >0.")
    def ok(sigma):
        try:
            p=prepare_case(replace(template,sigma=float(sigma)))
            return p.admissibility["status"]=="ADMISSIBLE",p
        except Exception:
            return False,None
    lo=0.0; hi=float(sigma_upper); okhi,phi=ok(hi)
    if okhi: return {"ceiling_at_least":hi,"bracketed":False,"note":"Increase sigma_upper if a tighter ceiling is required.","last_prepared":phi}
    last=None
    for _ in range(iterations):
        mid=.5*(lo+hi); good,p=ok(mid)
        if good: lo=mid; last=p
        else: hi=mid
    return {"ceiling":lo,"first_rejected":hi,"bracketed":True,"last_prepared":last}


def paired_severity_levels(stenosis_template: CaseSpec, dilation_template: CaseSpec, dilation_search_upper: float,
                           safety_factor: float=.9) -> Dict[str,Any]:
    S=find_sigma_admissibility_ceiling(stenosis_template,0.999)
    A=find_sigma_admissibility_ceiling(dilation_template,dilation_search_upper)
    if not S.get("bracketed",False) or not A.get("bracketed",False):
        return {"stenosis":S,"dilation":A,"ready":False,"reason":"One search ceiling was not bracketed; increase its search bound."}
    common=safety_factor*min(S["ceiling"],A["ceiling"])
    levels=[0.0,common/3,2*common/3,common]
    return {"stenosis":S,"dilation":A,"ready":True,"sigma_prod_max":common,"levels":levels,"safety_factor":safety_factor}


def spatial_convergence(case: CaseSpec, N_values: Sequence[int], progress: bool=False) -> Dict[str,Any]:
    rows=[]
    for N in N_values:
        prep=prepare_case(replace(case,N=int(N)))
        if prep.admissibility["status"]!="ADMISSIBLE":
            rows.append({"N":N,"status":prep.admissibility["status"],"coeff_error":prep.coeff_error})
            continue
        res=run_case(prep,paths=None,resume=False,progress=progress)
        rows.append({"N":N,"status":"ADMISSIBLE","R_max":res["summary"]["R_max"],"s_R_max":res["summary"]["s_R_max"],"I2_final":res["summary"]["I2_final"],"eta_tail_max":res["summary"]["eta_tail_max"],"coeff_error":prep.coeff_error})
    prev=None
    for r in rows:
        if r.get("status")!="ADMISSIBLE": continue
        if prev is not None:
            r["rel_I2_change_vs_prev"]=abs(r["I2_final"]-prev["I2_final"])/max(abs(r["I2_final"]),1e-30)
            r["rel_Rmax_change_vs_prev"]=abs(r["R_max"]-prev["R_max"])/max(abs(r["R_max"]),1e-30)
            r["prev_N"]=prev["N"]
        prev=r
    return {"rows":rows}


def temporal_convergence(case: CaseSpec, dt_values: Sequence[float], progress: bool=False) -> Dict[str,Any]:
    rows=[]
    out_interval=case.output_every_steps*case.dt; cp_interval=case.checkpoint_every_steps*case.dt
    for dt in dt_values:
        dt=float(dt)
        spec=replace(case,dt=dt,output_every_steps=max(1,int(round(out_interval/dt))),checkpoint_every_steps=max(1,int(round(cp_interval/dt))))
        prep=prepare_case(spec)
        if prep.admissibility["status"]!="ADMISSIBLE":
            rows.append({"dt":dt,"status":prep.admissibility["status"],"coeff_error":prep.coeff_error}); continue
        res=run_case(prep,paths=None,resume=False,progress=progress)
        rows.append({"dt":dt,"status":"ADMISSIBLE","R_max":res["summary"]["R_max"],"s_R_max":res["summary"]["s_R_max"],"I2_final":res["summary"]["I2_final"]})
    prev=None
    for r in rows:
        if r.get("status")!="ADMISSIBLE": continue
        if prev is not None:
            r["rel_I2_change_vs_prev"]=abs(r["I2_final"]-prev["I2_final"])/max(abs(r["I2_final"]),1e-30)
            r["rel_Rmax_change_vs_prev"]=abs(r["R_max"]-prev["R_max"])/max(abs(r["R_max"]),1e-30)
            r["prev_dt"]=prev["dt"]
        prev=r
    return {"rows":rows}


def estimate_runtime(case: CaseSpec, benchmark_steps: int=200) -> Dict[str,float]:
    prep=prepare_case(case)
    ah=np.fft.fft(initial_condition(case,prep.grid)); etd=etd_coefficients(prep)
    n=min(benchmark_steps,int(round(case.T_final/case.dt)))
    t0=time.perf_counter()
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
    elif topo["class"]=="multiple peaks":
        out["peaks"]=topo["peaks"]
    return out



__all__ = [name for name in globals() if not name.startswith("_")]
