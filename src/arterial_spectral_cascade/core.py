from __future__ import annotations

# Solver Design v2 compatibility layer over the Mathematical Model implementation.
from .core_base import *
from .core_base import _jsonable

SOLVER_SCHEMA = "solver-design-etdrk4-dealiased-v2"
RESULT_SCHEMA = "coefficient-disease-study-v2"
PARENT_REFERENCE_SCHEMA = "parent-reference-audit-v4"

# The Mathematical Model dataclass was created under the previous solver schema.
# Rebind only its solver-schema default while preserving the complete frozen model
# specification and all Mathematical Model fields.
from dataclasses import dataclass as _dataclass
_BaseCaseSpec = CaseSpec

@_dataclass(frozen=True)
class CaseSpec(_BaseCaseSpec):
    solver_schema: str = SOLVER_SCHEMA

def explicit_stiffness_screen(ahat: np.ndarray, prep: PreparedCase) -> float:
    """Solver Design screening quantity for explicitly treated nonlinear/heterogeneous scales."""
    ah=project_hat(ahat,prep.grid)
    a=np.fft.ifft(ah).real
    kret=float(prep.grid.k_ret); h=float(prep.spec.dt)
    return float(h*(np.max(np.abs(a))*kret
                    + np.max(np.abs(prep.b_tilde))*kret**3
                    + np.max(np.abs(prep.g_tilde))*kret))


def verify_kdv_refinement() -> Dict[str,Any]:
    def drift(N: int, dt: float, T: float=.6) -> float:
        spec=CaseSpec("H0",Wo0=10,N=N,dt=dt,T_final=T,k0=1.0,A1=1.5,A2_ratio=.5,A3_ratio=.3)
        grid=make_grid(N,spec.Lg); bval=.005; z=np.zeros(N)
        prep=_verification_prepared_from_fields(spec,np.full(N,bval),z,"kdv")
        ah=np.fft.fft(initial_condition(spec,grid)); etd=etd_coefficients(prep,dt)
        def inv(v):
            a=np.fft.ifft(v).real; ax=np.fft.ifft((-1j*grid.k)*v).real
            return np.array([grid.dx*np.sum(a),grid.dx*np.sum(a*a),grid.dx*np.sum(a**3/6-bval*ax*ax/2)])
        q0=inv(ah)
        for _ in range(int(round(T/dt))): ah=etdrk4_step(ah,prep,etd,True,False)
        q1=inv(ah); return float(np.linalg.norm(q1-q0)/max(np.linalg.norm(q0),1e-15))
    temporal_coarse=drift(64,.01); temporal_fine=drift(64,.005)
    spatial_coarse=drift(24,.001,T=.3); spatial_fine=drift(32,.001,T=.3)
    return {"name":"kdv_refinement","temporal_coarse":temporal_coarse,"temporal_fine":temporal_fine,
            "spatial_coarse":spatial_coarse,"spatial_fine":spatial_fine,
            "pass":bool(temporal_fine<temporal_coarse and spatial_fine<spatial_coarse)}


def _verification_prepared_from_fields(spec: CaseSpec, b: np.ndarray, gfield: np.ndarray,
                                       case_id: str="verification") -> PreparedCase:
    grid=make_grid(spec.N,spec.Lg); z=np.zeros(grid.N)
    b=np.asarray(b,float); gfield=np.asarray(gfield,float)
    return PreparedCase(spec=spec,case_id=case_id,grid=grid,psi_D_raw=z.copy(),psi_D=z.copy(),b=b,g=gfield,
                        b_bar=float(np.mean(b)),g_bar=float(np.mean(gfield)),b_tilde=b-np.mean(b),g_tilde=gfield-np.mean(gfield),
                        morphology_error=0.0,coeff_error=0.0,admissibility={"status":"ADMISSIBLE"})


def _integrated_balance_error(prep: PreparedCase) -> Dict[str,float]:
    spec=prep.spec; ah=np.fft.fft(initial_condition(spec,prep.grid)); etd=etd_coefficients(prep)
    rows=[]; steps=int(round(spec.T_final/spec.dt))
    for step in range(steps+1):
        bal=integrals_and_balance(ah,prep)
        rows.append((step*spec.dt,bal["I1"],bal["E"],bal["B1"],bal["BE"]))
        if step<steps: ah=etdrk4_step(ah,prep,etd,True,True)
    A=np.asarray(rows,float); t=A[:,0]
    q1=np.concatenate([[0.0],np.cumsum(0.5*np.diff(t)*(A[:-1,3]+A[1:,3]))])
    qE=np.concatenate([[0.0],np.cumsum(0.5*np.diff(t)*(A[:-1,4]+A[1:,4]))])
    r1=A[:,1]-A[0,1]-q1; rE=A[:,2]-A[0,2]-qE
    return {"max_abs_RI1":float(np.max(np.abs(r1))),"max_abs_RE":float(np.max(np.abs(rE)))}


def verify_fractional_burgers_refinement() -> Dict[str,Any]:
    N=64; Lg=4*np.pi; grid=make_grid(N,Lg); x=grid.xi
    gfield=0.01*(1+0.15*np.cos(x)); b=np.zeros(N)
    spec=CaseSpec("H0",Wo0=10,N=N,Lg=Lg,dt=.004,T_final=.08,k0=.5)
    prep=_verification_prepared_from_fields(spec,b,gfield,"burgers")
    a=0.7*np.sin(.5*x)+0.2*np.cos(x)-0.1*np.sin(1.5*x); ah=np.fft.fft(a)
    lam=lambda_from_hat(ah,grid)
    rhs_phys=-0.5*derivative_from_hat(np.fft.fft(a*a),grid,1)-gfield*lam
    phys=project_hat(np.fft.fft(rhs_phys),grid)/N
    modal=direct_modal_rhs_normalized(a,b,gfield,grid,True)
    split=full_rhs_hat(ah,prep)/N
    op_err=max(float(np.max(np.abs(phys-modal))),float(np.max(np.abs(phys-split))),float(np.max(np.abs(modal-split))))
    coarse=_integrated_balance_error(prep)
    fine_spec=replace(spec,dt=.002); fine=_integrated_balance_error(_verification_prepared_from_fields(fine_spec,b,gfield,"burgers-fine"))
    coarse_max=max(coarse.values()); fine_max=max(fine.values())
    return {"name":"fractional_burgers_refinement","operator_error":op_err,"coarse_balance":coarse,
            "fine_balance":fine,"refinement_ratio":fine_max/max(coarse_max,1e-30),
            "pass":bool(op_err<1e-11 and fine_max<0.35*coarse_max)}


def verify_parent_pointwise_coefficients() -> Dict[str,Any]:
    spec=CaseSpec("P1",Wo0=10,N=64,dt=.002,T_final=.02,k0=.5,eps_b=.12,eps_g=.1,q=1.0,chi_b=0.0,chi_g=0.0)
    prep=prepare_case(spec); a=initial_condition(spec,prep.grid); ah=np.fft.fft(a)
    modal=direct_modal_rhs_normalized(a,prep.b,prep.g,prep.grid,True)
    split=full_rhs_hat(ah,prep)/prep.grid.N
    err=float(np.max(np.abs(modal-split)))
    psi_zero=float(np.max(np.abs(prep.psi_D)))
    return {"name":"parent_pointwise_coefficients","modal_vs_split":err,"psi_D_max":psi_zero,
            "pass":bool(err<1e-11 and psi_zero<1e-15)}


def verify_field_resolution_refinement() -> Dict[str,Any]:
    base=CaseSpec("DL",Wo0=10,N=32,dt=.002,T_final=.02,k0=.5,chi_b=.02,chi_g=.02,w=2.0,p=1,
                  R0_over_L0=.01,slow_variation_limit=.1)
    c=prepare_case(base); f=prepare_case(replace(base,N=64))
    positive=(c.admissibility["b_min"]>0 and c.admissibility["g_min"]>0 and f.admissibility["b_min"]>0 and f.admissibility["g_min"]>0)
    decreased=(f.morphology_error<c.morphology_error and f.coeff_error<c.coeff_error)
    return {"name":"field_resolution_refinement","coarse_morphology_error":c.morphology_error,"fine_morphology_error":f.morphology_error,
            "coarse_coefficient_error":c.coeff_error,"fine_coefficient_error":f.coeff_error,
            "fine_status":f.admissibility["status"],"pass":bool(positive and decreased and f.admissibility["status"]=="ADMISSIBLE")}


def verify_heterogeneous_balance_refinement() -> Dict[str,Any]:
    base=CaseSpec("P1",Wo0=10,N=64,dt=.004,T_final=.08,k0=.5,eps_b=.1,eps_g=.08,q=1.0)
    coarse=_integrated_balance_error(prepare_case(base)); fine=_integrated_balance_error(prepare_case(replace(base,dt=.002)))
    ratios={k:fine[k]/max(coarse[k],1e-30) for k in coarse}
    return {"name":"heterogeneous_balance_refinement","coarse":coarse,"fine":fine,"ratios":ratios,
            "pass":bool(max(ratios.values())<0.35)}


def verify_aliasing_control() -> Dict[str,Any]:
    base=CaseSpec("P1",Wo0=10,N=32,dt=.002,T_final=.04,k0=.5,eps_b=.15,eps_g=.1,q=1.0)
    maxima=[]
    for N in (32,64):
        prep=prepare_case(replace(base,N=N)); ah=np.fft.fft(initial_condition(prep.spec,prep.grid)); etd=etd_coefficients(prep); mx=0.0
        for step in range(int(round(prep.spec.T_final/prep.spec.dt))+1):
            mx=max(mx,spectral_broadening(ah,prep)["eta_tail"])
            if step<int(round(prep.spec.T_final/prep.spec.dt)): ah=etdrk4_step(ah,prep,etd)
        maxima.append(float(mx))
    return {"name":"aliasing_control","eta_tail_N":maxima[0],"eta_tail_2N":maxima[1],
            "pass":bool(max(maxima)<1e-3 and maxima[1] <= maxima[0]+1e-12)}


def quick_numerical_check() -> Dict[str,Any]:
    tests=[verify_fft_sign(),verify_three_form(),verify_constant_linear(),verify_balance_identities(),verify_modal_budget()]
    return {"tests":tests,"pass":all(t["pass"] for t in tests)}


def core_verification_suite(include_kdv: bool=True) -> Dict[str,Any]:
    tests=[verify_fft_sign(),verify_three_form(),verify_constant_linear(),verify_heterogeneous_linear_order(),
           verify_balance_identities(),verify_heterogeneous_balance_refinement(),verify_fractional_burgers_refinement(),
           verify_parent_pointwise_coefficients(),verify_field_resolution_refinement(),verify_aliasing_control(),verify_modal_budget()]
    if include_kdv: tests.append(verify_kdv_refinement())
    return {"schema":SOLVER_SCHEMA,"tests":tests,"pass":all(t["pass"] for t in tests),"timestamp":time.time()}


__all__ = [name for name in globals() if not name.startswith("_")]
