from __future__ import annotations

"""Verified performance backend for Colab-class CPU execution.

The backend preserves the Solver Design semi-discrete equations and ETDRK4
coefficients.  It reduces transform overhead by packing the two real-valued
heterogeneous derivative reconstructions into one complex inverse FFT, caches
coefficient-only balance fields, and uses lightweight transactional checkpoint
archives.  The reference implementation remains available and is used for an
automatic equivalence check before the optimized backend is installed.
"""

import os
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import core as core
from . import core_base as reference


@dataclass(frozen=True)
class PerformanceBackendStatus:
    requested: str
    active: str
    verified: bool
    residual_error: Optional[float]
    step_error: Optional[float]
    trajectory_error: Optional[float]
    reason: str = ""


def _project_inplace_copy(v: np.ndarray, grid: core.SpectralGrid) -> np.ndarray:
    out=np.array(v,dtype=np.complex128,copy=True)
    out[~grid.mask]=0.0
    return out


def optimized_residual_hat(ahat: np.ndarray, prep: core.PreparedCase,
                           include_nonlinearity: bool=True,
                           include_heterogeneity: bool=True) -> np.ndarray:
    """Mathematically equivalent residual with one fewer IFFT per hetero call.

    The third derivative and fractional derivative of a real field are both
    real-valued.  Linearity of the inverse FFT therefore permits their spectra
    to be packed as A + i B and reconstructed in one complex transform.  The
    real and imaginary parts recover the two physical fields to roundoff.
    """
    grid=prep.grid
    ah=_project_inplace_copy(ahat,grid)
    out=np.zeros(grid.N,dtype=np.complex128)
    if include_nonlinearity:
        a=np.fft.ifft(ah)
        out += (1j*grid.k/2.0)*np.fft.fft(a*a)
    if include_heterogeneity:
        packed=np.fft.ifft((1j*grid.k**3)*ah + 1j*(np.abs(grid.k)*ah))
        axxx=packed.real
        lama=packed.imag
        out += np.fft.fft(-prep.b_tilde*axxx-prep.g_tilde*lama)
    out[~grid.mask]=0.0
    return out


def optimized_etdrk4_step(v: np.ndarray, prep: core.PreparedCase,
                          etd: core.ETDCoefficients,
                          include_nonlinearity: bool=True,
                          include_heterogeneity: bool=True) -> np.ndarray:
    """ETDRK4 with the verified optimized residual and unchanged coefficients."""
    grid=prep.grid
    v=_project_inplace_copy(v,grid)
    Nv=optimized_residual_hat(v,prep,include_nonlinearity,include_heterogeneity)
    a1=_project_inplace_copy(etd.E2*v+etd.Q*Nv,grid)
    Na=optimized_residual_hat(a1,prep,include_nonlinearity,include_heterogeneity)
    a2=_project_inplace_copy(etd.E2*v+etd.Q*Na,grid)
    Nb=optimized_residual_hat(a2,prep,include_nonlinearity,include_heterogeneity)
    a3=_project_inplace_copy(etd.E2*a1+etd.Q*(2*Nb-Nv),grid)
    Nc=optimized_residual_hat(a3,prep,include_nonlinearity,include_heterogeneity)
    return _project_inplace_copy(etd.E*v+etd.f1*Nv+2*etd.f2*(Na+Nb)+etd.f3*Nc,grid)


@dataclass
class _DiagnosticCache:
    b1: np.ndarray
    b3: np.ndarray
    lamg: np.ndarray
    btilde_max: float
    gtilde_max: float


def _diagnostic_cache(prep: core.PreparedCase) -> _DiagnosticCache:
    cache=getattr(prep,"_performance_diagnostic_cache",None)
    if cache is not None:
        return cache
    grid=prep.grid
    bh=np.fft.fft(prep.b); gh=np.fft.fft(prep.g)
    cache=_DiagnosticCache(
        b1=np.fft.ifft((-1j*grid.k)*bh).real,
        b3=np.fft.ifft((1j*grid.k**3)*bh).real,
        lamg=np.fft.ifft(np.abs(grid.k)*gh).real,
        btilde_max=float(np.max(np.abs(prep.b_tilde))),
        gtilde_max=float(np.max(np.abs(prep.g_tilde))),
    )
    setattr(prep,"_performance_diagnostic_cache",cache)
    return cache


def optimized_integrals_and_balance(ahat: np.ndarray, prep: core.PreparedCase,
                                    rhs_hat: Optional[np.ndarray]=None) -> Dict[str,float]:
    """Exact Solver Design balances with coefficient-only fields cached once."""
    grid=prep.grid; ah=_project_inplace_copy(ahat,grid)
    a=np.fft.ifft(ah).real
    if rhs_hat is None:
        rhs_hat=core.full_rhs_hat(ah,prep)
    rhs=np.fft.ifft(rhs_hat).real
    cache=_diagnostic_cache(prep)
    I1=grid.dx*np.sum(a); I2=grid.dx*np.sum(a*a); E=0.5*I2
    ax=np.fft.ifft((-1j*grid.k)*ah).real
    lama=np.fft.ifft(np.abs(grid.k)*ah).real
    B1=grid.dx*np.sum(a*cache.b3-a*cache.lamg)
    BE=grid.dx*np.sum(0.5*cache.b3*a*a-1.5*cache.b1*ax*ax-prep.g*a*lama)
    dI1_num=grid.dx*np.sum(rhs); dE_num=grid.dx*np.sum(a*rhs)
    G_bal=2*BE/I2 if I2>0 else np.nan
    return {"I1":float(I1),"I2":float(I2),"E":float(E),"B1":float(B1),"BE":float(BE),
            "dI1_num":float(dI1_num),"dE_num":float(dE_num),"R_I1_inst":float(dI1_num-B1),
            "R_E_inst":float(dE_num-BE),"G_bal":float(G_bal)}


def optimized_record(prep: core.PreparedCase, ahat: np.ndarray, step: int) -> Dict[str,float]:
    """One reconstruction for balance and chi_h diagnostics; cached coefficients."""
    ah=_project_inplace_copy(ahat,prep.grid)
    rhs=core.full_rhs_hat(ah,prep)
    bal=optimized_integrals_and_balance(ah,prep,rhs)
    sp=core.spectral_broadening(ah,prep)
    a=np.fft.ifft(ah).real
    cache=_diagnostic_cache(prep)
    kret=float(prep.grid.k_ret); h=float(prep.spec.dt)
    chi_h=h*(float(np.max(np.abs(a)))*kret+cache.btilde_max*kret**3+cache.gtilde_max*kret)
    return {"step":int(step),"s":float(step*prep.spec.dt),**bal,**sp,"chi_h":float(chi_h)}


def _atomic_save_checkpoint_npz(path: Path, **arrays) -> None:
    """Transactional uncompressed NPZ for frequent checkpoints only.

    Final scientific result archives remain compressed by the standard storage
    layer.  This checkpoint format trades temporary disk size for much lower CPU
    overhead and is still validated by reopening every stored array.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent); os.close(fd)
    try:
        with open(tmp,"wb") as f:
            np.savez(f,**arrays)
        with np.load(tmp,allow_pickle=False) as z:
            for key in z.keys():
                _=z[key].shape
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.remove(tmp)


def _fast_single_checkpoint(storage, prep, paths, step, ahat, history, peak):
    if not np.all(np.isfinite(ahat)):
        raise ValueError(f"Refusing to checkpoint non-finite spectral state for {prep.case_id} at step {step}.")
    peak_ahat=np.asarray(peak.get("ahat",ahat))
    if not np.all(np.isfinite(peak_ahat)):
        raise ValueError(f"Refusing to checkpoint non-finite peak state for {prep.case_id} at step {step}.")
    statecheck=storage._runtime_state_check(ahat,prep)
    if not statecheck["pass"]:
        raise ValueError(f"Refusing to checkpoint numerically invalid state for {prep.case_id} at step {step}.")
    cdir=paths.checkpoints/prep.case_id; cdir.mkdir(parents=True,exist_ok=True); path=cdir/"latest.npz"
    integrity=core.stable_hash(storage._integrity_payload(prep),32)
    payload={"step":np.array(step,dtype=np.int64),"ahat":ahat,
             "case_metadata_hash":np.array(core.stable_hash({"case_id":prep.case_id,"spec":storage._jsonable(prep.spec)})),
             "integrity_hash":np.array(integrity),"peak_R":np.array(float(peak.get("R",-np.inf))),
             "peak_step":np.array(int(peak.get("step",0)),dtype=np.int64),"peak_ahat":peak_ahat}
    for k,v in history.items(): payload[f"hist_{k}"]=np.asarray(v)
    _atomic_save_checkpoint_npz(path,**payload)
    return path


def _fast_pair_checkpoint(paired, storage, parent, mm, paths, step, ahp, ahm, hp, hm, comp, peakp, peakm):
    states=(ahp,ahm,np.asarray(peakp.get("ahat",ahp)),np.asarray(peakm.get("ahat",ahm)))
    if not all(np.all(np.isfinite(v)) for v in states):
        raise ValueError(f"Refusing to checkpoint non-finite paired state at step {step}.")
    scp=storage._runtime_state_check(ahp,parent); scm=storage._runtime_state_check(ahm,mm)
    if not scp["pass"] or not scm["pass"]:
        raise ValueError(f"Refusing to checkpoint numerically invalid paired state at step {step}.")
    path=paired._pair_checkpoint_path(parent,mm,paths)
    pairhash=core.stable_hash(paired._pair_integrity_payload(parent,mm),32)
    data={"step":np.array(step,dtype=np.int64),"ahat_parent":ahp,"ahat_mm":ahm,"pair_hash":np.array(pairhash),
          "peakp_R":np.array(float(peakp["R"])),"peakp_step":np.array(int(peakp["step"])),"peakp_ahat":peakp["ahat"],
          "peakm_R":np.array(float(peakm["R"])),"peakm_step":np.array(int(peakm["step"])),"peakm_ahat":peakm["ahat"]}
    for prefix,H in (("hp",hp),("hm",hm),("cmp",comp)):
        for k,v in H.items(): data[f"{prefix}_{k}"]=np.asarray(v)
    _atomic_save_checkpoint_npz(path,**data)


def verify_optimized_backend_equivalence() -> Dict[str,Any]:
    """Compare optimized and reference residual/step/trajectory at double precision."""
    spec=core.CaseSpec("DL",Wo0=10,N=48,dt=.001,T_final=.012,k0=.5,chi_b=.015,chi_g=-.01,
                       w=3.0,p=1,R0_over_L0=.01,slow_variation_limit=.1)
    prep=core.prepare_case(spec)
    v=core.project_hat(np.fft.fft(core.initial_condition(spec,prep.grid)),prep.grid)
    etd=core.etd_coefficients(prep)
    rr=reference.residual_hat(v,prep,True,True); ro=optimized_residual_hat(v,prep,True,True)
    residual_error=float(np.linalg.norm(ro-rr)/max(np.linalg.norm(rr),1e-30))
    sr=reference.etdrk4_step(v,prep,etd,True,True); so=optimized_etdrk4_step(v,prep,etd,True,True)
    step_error=float(np.linalg.norm(so-sr)/max(np.linalg.norm(sr),1e-30))
    vr=v.copy(); vo=v.copy()
    for _ in range(12):
        vr=reference.etdrk4_step(vr,prep,etd,True,True)
        vo=optimized_etdrk4_step(vo,prep,etd,True,True)
    trajectory_error=float(np.linalg.norm(vo-vr)/max(np.linalg.norm(vr),1e-30))
    passed=bool(residual_error<5e-13 and step_error<5e-13 and trajectory_error<5e-12)
    return {"name":"optimized_backend_equivalence","residual_error":residual_error,
            "step_error":step_error,"trajectory_error":trajectory_error,"pass":passed}


def benchmark_performance_backend(prep: core.PreparedCase, steps: int=200) -> Dict[str,float]:
    """Deterministic wall-clock comparison; timings are informational, not acceptance criteria."""
    n=max(1,int(steps)); v0=core.project_hat(np.fft.fft(core.initial_condition(prep.spec,prep.grid)),prep.grid)
    etd=core.etd_coefficients(prep)
    v=v0.copy(); t0=time.perf_counter()
    for _ in range(n): v=reference.etdrk4_step(v,prep,etd,True,True)
    reference_seconds=time.perf_counter()-t0
    vr=v
    v=v0.copy(); t0=time.perf_counter()
    for _ in range(n): v=optimized_etdrk4_step(v,prep,etd,True,True)
    optimized_seconds=time.perf_counter()-t0
    err=float(np.linalg.norm(v-vr)/max(np.linalg.norm(vr),1e-30))
    return {"steps":float(n),"reference_seconds":float(reference_seconds),
            "optimized_seconds":float(optimized_seconds),"speedup":float(reference_seconds/max(optimized_seconds,1e-30)),
            "relative_state_error":err}


def install_performance_backend(requested: Optional[str]=None) -> PerformanceBackendStatus:
    """Install only after an automatic equivalence check; otherwise retain reference."""
    global _INSTALL_RESULT, PERFORMANCE_BACKEND_STATUS
    requested=(requested or os.environ.get("ASC_BACKEND","optimized")).strip().lower()
    if _INSTALL_RESULT is not None and requested==_INSTALL_RESULT.requested:
        return _INSTALL_RESULT
    if requested not in {"optimized","reference"}:
        raise ValueError("ASC_BACKEND must be 'optimized' or 'reference'.")
    if requested=="reference":
        _INSTALL_RESULT=PerformanceBackendStatus(requested,"reference",True,None,None,None,"reference backend requested")
        PERFORMANCE_BACKEND_STATUS=_INSTALL_RESULT
        return _INSTALL_RESULT
    check=verify_optimized_backend_equivalence()
    if not check["pass"]:
        _INSTALL_RESULT=PerformanceBackendStatus(requested,"reference",False,check["residual_error"],check["step_error"],check["trajectory_error"],
                                        "optimized/reference equivalence check failed; reference backend retained")
        PERFORMANCE_BACKEND_STATUS=_INSTALL_RESULT
        return _INSTALL_RESULT

    from . import storage
    from . import paired_runtime
    from . import parent
    from . import solver_verification

    storage.etdrk4_step=optimized_etdrk4_step
    storage._record=optimized_record
    storage.save_checkpoint=lambda prep,paths,step,ahat,history,peak: _fast_single_checkpoint(storage,prep,paths,step,ahat,history,peak)

    paired_runtime.etdrk4_step=optimized_etdrk4_step
    paired_runtime._record=optimized_record
    paired_runtime._save_pair_checkpoint=lambda parent_case,mm,paths,step,ahp,ahm,hp,hm,comp,peakp,peakm: _fast_pair_checkpoint(
        paired_runtime,storage,parent_case,mm,paths,step,ahp,ahm,hp,hm,comp,peakp,peakm)
    # parent.run_paired_case deliberately mirrors its own etdrk4_step global into
    # paired_runtime before execution; keep that compatibility surface optimized.
    parent.etdrk4_step=optimized_etdrk4_step

    # Record the active backend and equivalence evidence in every case metadata file.
    original_save_case_metadata=storage.save_case_metadata
    backend_payload={"requested":requested,"active":"optimized","verified":True,
                     "residual_error":check["residual_error"],"step_error":check["step_error"],
                     "trajectory_error":check["trajectory_error"]}
    def save_case_metadata_with_backend(prep,paths):
        path=original_save_case_metadata(prep,paths)
        payload=json.loads(Path(path).read_text())
        payload["performance_backend"]=backend_payload
        storage.atomic_write_json(path,payload)
        return path
    storage.save_case_metadata=save_case_metadata_with_backend
    paired_runtime.save_case_metadata=save_case_metadata_with_backend

    # The optimized/reference equivalence check is an explicit verification item,
    # so a full verification record demonstrates both numerical validity and
    # performance-backend fidelity.
    original_full_verification=solver_verification.full_verification_suite
    def full_verification_with_backend():
        report=original_full_verification()
        eq=verify_optimized_backend_equivalence()
        report["tests"].append(eq)
        report["pass"]=bool(report.get("pass",True) and eq["pass"])
        report["performance_backend"]=backend_payload
        return report
    solver_verification.full_verification_suite=full_verification_with_backend
    parent.full_verification_suite=full_verification_with_backend

    _INSTALL_RESULT=PerformanceBackendStatus(requested,"optimized",True,check["residual_error"],check["step_error"],check["trajectory_error"],
                                    "optimized backend verified against reference and installed")
    PERFORMANCE_BACKEND_STATUS=_INSTALL_RESULT
    return _INSTALL_RESULT


PERFORMANCE_BACKEND_STATUS: Optional[PerformanceBackendStatus] = None
_INSTALL_RESULT: Optional[PerformanceBackendStatus] = None
