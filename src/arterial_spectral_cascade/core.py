from __future__ import annotations

import os, json, math, hashlib, tempfile, shutil, time, platform
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.signal import find_peaks, peak_prominences

# Authoritative schemas: Supplementary File 1 (Mathematical Model) and
# Supplementary File 2 (Solver Design).
MODEL_SCHEMA = "mathematical-model-coefficient-heterogeneous-v1"
SOLVER_SCHEMA = "solver-design-etdrk4-dealiased-v1"
RESULT_SCHEMA = "coefficient-disease-study-v1"
PARENT_REFERENCE_SCHEMA = "parent-reference-audit-v3"


@dataclass(frozen=True)
class ModelConstants:
    Lg: float = 4*np.pi
    b_ref: float = 1.0
    g_ref: float = 0.005
    Cg: float = 0.1
    a0: float = 1.0
    c0: float = 0.0
    A1: float = 1.0
    A2_ratio: float = 0.3
    A3_ratio: float = 0.1
    phases: Tuple[float,float,float] = (0.0, np.pi/4, np.pi/2)
    k_cutoff_ratio: float = 1.5
    heterogeneity_limit: float = 0.3


CONSTS = ModelConstants()


@dataclass(frozen=True)
class Lesion:
    """One term in the normalized multiple-lesion morphology Psi_M."""
    amplitude: float
    xi_c: float
    w: float
    p: int = 1

    def __post_init__(self):
        if not np.isfinite(self.amplitude) or self.amplitude <= 0:
            raise ValueError("Lesion.amplitude must be finite and > 0.")
        if not np.isfinite(self.xi_c):
            raise ValueError("Lesion.xi_c must be finite.")
        if not np.isfinite(self.w) or self.w <= 0:
            raise ValueError("Lesion.w must be finite and > 0.")
        if int(self.p) != self.p or self.p < 1:
            raise ValueError("Lesion.p must be an integer >= 1.")


@dataclass(frozen=True)
class DistributedMode:
    """One Fourier component in the distributed irregular morphology r_D."""
    amplitude: float
    q: float
    phase: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.amplitude):
            raise ValueError("DistributedMode.amplitude must be finite.")
        if not np.isfinite(self.q) or self.q == 0:
            raise ValueError("DistributedMode.q must be finite and nonzero.")
        if not np.isfinite(self.phase):
            raise ValueError("DistributedMode.phase must be finite.")


@dataclass(frozen=True)
class CaseSpec:
    """Complete mathematical/numerical specification for one deterministic case.

    Disease coordinates are coefficient-space coordinates from the Mathematical
    Model: morphology class/parameters and signed sensitivities chi_b, chi_g.
    No local-radius or local-Womersley constitutive closure is represented here.
    """

    case_class: str
    Wo0: float
    N: int = 512
    Lg: float = 4*np.pi
    dt: float = 2e-4
    T_final: float = 60.0
    k0: float = 1.0
    A1: float = 1.0
    A2_ratio: float = 0.3
    A3_ratio: float = 0.1
    phases: Tuple[float,float,float] = (0.0, np.pi/4, np.pi/2)

    # Parent sinusoidal coefficient modulation.
    eps_b: float = 0.0
    eps_g: float = 0.0
    q: float = 1.0

    # Disease-to-coefficient sensitivities from Supplementary File 1.
    chi_b: float = 0.0
    chi_g: float = 0.0

    # DL: single localized morphology Psi_L.
    xi_c: float = 2*np.pi
    w: float = 1.5
    p: int = 1

    # DM: normalized sum of localized terms Psi_M.
    lesions: Tuple[Lesion, ...] = ()

    # DR: either analytical distributed modes Psi_R or an externally supplied
    # normalized geometry-derived morphology sampled on the collocation grid.
    distributed_modes: Tuple[DistributedMode, ...] = ()
    sampled_psi: Tuple[float, ...] = ()
    morphology_provenance: str = ""
    morphology_scale: Optional[float] = None  # ell_D / L0 when not inferable from w.

    # MM is generated programmatically and carries the exact parent means.
    b_mean_override: Optional[float] = None
    g_mean_override: Optional[float] = None

    output_every_steps: int = 100
    checkpoint_every_steps: int = 5000
    mechanism: bool = False

    # Declared reduced-order consistency inputs. The Mathematical Model states
    # R0/ell_D << 1; the numerical study supplies a conservative acceptance limit.
    R0_over_L0: Optional[float] = None
    slow_variation_limit: Optional[float] = None
    morphology_projection_limit: float = 1e-8
    coeff_projection_limit: float = 1e-8

    notes: str = ""
    model_schema: str = MODEL_SCHEMA
    solver_schema: str = SOLVER_SCHEMA
    result_schema: str = RESULT_SCHEMA

    def __post_init__(self):
        allowed={"H0","P0","P1","DL","DM","DR","MM"}
        if self.case_class not in allowed:
            raise ValueError(f"case_class must be one of {sorted(allowed)}.")
        if not np.isfinite(self.Wo0) or self.Wo0 <= 0:
            raise ValueError("Wo0 must be finite and positive.")
        if self.N < 16 or self.N % 2:
            raise ValueError("N must be an even integer >= 16.")
        if self.dt <= 0 or self.T_final <= 0:
            raise ValueError("dt and T_final must be positive.")
        if self.k0 <= 0:
            raise ValueError("k0 must be positive.")
        if self.output_every_steps < 1 or self.checkpoint_every_steps < 1:
            raise ValueError("Output/checkpoint cadence must be positive integers.")
        if self.morphology_projection_limit <= 0 or self.coeff_projection_limit <= 0:
            raise ValueError("Projection-error limits must be positive.")
        if not np.isfinite(self.eps_b) or not np.isfinite(self.eps_g):
            raise ValueError("eps_b and eps_g must be finite.")
        if not np.isfinite(self.chi_b) or not np.isfinite(self.chi_g):
            raise ValueError("chi_b and chi_g must be finite.")
        if abs(self.eps_b) > 0.3 + 1e-14 or abs(self.eps_g) > 0.3 + 1e-14:
            raise ValueError("Parent sinusoidal modulation must remain within |eps_b|, |eps_g| <= 0.3.")
        if (self.eps_b != 0 or self.eps_g != 0):
            cycles=self.q*self.Lg/(2*np.pi)
            if not np.isfinite(self.q) or abs(cycles-round(cycles)) > 1e-12:
                raise ValueError("Parent q must be periodic on [0,Lg].")

        if self.case_class in {"H0","P0"}:
            if any(abs(v)>0 for v in (self.eps_b,self.eps_g,self.chi_b,self.chi_g)):
                raise ValueError("H0/P0 are constant-coefficient cases.")
            if self.lesions or self.distributed_modes or self.sampled_psi:
                raise ValueError("H0/P0 do not carry a disease morphology.")
        elif self.case_class == "P1":
            if self.chi_b != 0 or self.chi_g != 0:
                raise ValueError("P1 is the parent sinusoidal coefficient class and requires chi_b=chi_g=0.")
            if self.lesions or self.distributed_modes or self.sampled_psi:
                raise ValueError("P1 has Psi_D identically zero.")
        elif self.case_class == "DL":
            if not np.isfinite(self.w) or self.w <= 0 or int(self.p)!=self.p or self.p<1:
                raise ValueError("DL requires w>0 and integer p>=1.")
        elif self.case_class == "DM":
            if not self.lesions:
                raise ValueError("DM requires a non-empty lesions tuple.")
        elif self.case_class == "DR":
            if bool(self.distributed_modes) == bool(self.sampled_psi):
                raise ValueError("DR requires exactly one of distributed_modes or sampled_psi.")
            for mode in self.distributed_modes:
                cycles=mode.q*self.Lg/(2*np.pi)
                if abs(cycles-round(cycles)) > 1e-12:
                    raise ValueError("Every distributed q_j must be periodic on [0,Lg].")
            if self.sampled_psi and not self.morphology_provenance.strip():
                raise ValueError("Sampled geometry-derived DR morphology requires provenance.")
        elif self.case_class == "MM":
            if self.b_mean_override is None or self.g_mean_override is None:
                raise ValueError("MM requires exact b_mean_override and g_mean_override.")
            if self.b_mean_override <= 0 or self.g_mean_override <= 0:
                raise ValueError("MM mean coefficients must be positive.")
            if any(abs(v)>0 for v in (self.eps_b,self.eps_g,self.chi_b,self.chi_g)):
                raise ValueError("MM contains no coefficient heterogeneity.")


@dataclass
class SpectralGrid:
    N: int
    Lg: float
    xi: np.ndarray
    dx: float
    nu: np.ndarray
    k: np.ndarray
    mask: np.ndarray
    k_ret: float


@dataclass
class PreparedCase:
    spec: CaseSpec
    case_id: str
    grid: SpectralGrid
    psi_D_raw: np.ndarray
    psi_D: np.ndarray
    b: np.ndarray
    g: np.ndarray
    b_bar: float
    g_bar: float
    b_tilde: np.ndarray
    g_tilde: np.ndarray
    morphology_error: float
    coeff_error: float
    admissibility: Dict[str,Any]
    morphology_provenance: str = ""
    parent_case_id: Optional[str] = None


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, tuple): return [_jsonable(v) for v in x]
    if isinstance(x, list): return [_jsonable(v) for v in x]
    if isinstance(x, dict): return {str(k): _jsonable(v) for k,v in x.items()}
    if hasattr(x, "__dataclass_fields__"): return _jsonable(asdict(x))
    return x


def canonical_json(payload: Any) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",",":"), allow_nan=False)


def stable_hash(payload: Any, n: int=16) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:n]


def array_sha256(a: np.ndarray) -> str:
    arr=np.ascontiguousarray(np.asarray(a))
    h=hashlib.sha256()
    h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode()); h.update(arr.view(np.uint8))
    return h.hexdigest()


def file_sha256(path: os.PathLike) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def make_grid(N: int, Lg: float=4*np.pi) -> SpectralGrid:
    xi=np.linspace(0.0,Lg,N,endpoint=False)
    dx=Lg/N
    nu=np.fft.fftfreq(N,d=1.0/N)
    k=-(2*np.pi/Lg)*nu
    cutoff=int(np.floor(N/3))
    mask=np.abs(nu)<=cutoff
    k_ret=(2*np.pi/Lg)*cutoff
    return SpectralGrid(N,Lg,xi,dx,nu,k,mask,k_ret)


def project_hat(fhat: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    out=np.array(fhat,dtype=np.complex128,copy=True)
    out[~grid.mask]=0.0
    return out


def project_real(f: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    return np.fft.ifft(project_hat(np.fft.fft(f),grid)).real


def normalized_hat(f: np.ndarray) -> np.ndarray:
    return np.fft.fft(f)/f.size


def derivative_from_hat(fhat: np.ndarray, grid: SpectralGrid, order: int=1) -> np.ndarray:
    if order==1: mult=-1j*grid.k
    elif order==3: mult=1j*grid.k**3
    else: mult=(-1j*grid.k)**order
    return np.fft.ifft(mult*fhat)


def lambda_from_hat(fhat: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    return np.fft.ifft(np.abs(grid.k)*fhat)


def localized_morphology(xi: np.ndarray, xi_c: float, w: float, p: int, Lg: float) -> np.ndarray:
    """Canonical smooth periodic localized morphology Psi_L."""
    if w<=0 or int(p)!=p or p<1:
        raise ValueError("w>0 and integer p>=1 are required.")
    z=(Lg/(np.pi*w))*np.sin(np.pi*(xi-xi_c)/Lg)
    return np.exp(-(z**(2*int(p))))


# Backward name retained only as a morphology-kernel alias; it no longer encodes radius.
lesion_kernel = localized_morphology


def multiple_morphology(xi: np.ndarray, lesions: Sequence[Lesion], Lg: float) -> np.ndarray:
    if not lesions:
        raise ValueError("At least one lesion is required for Psi_M.")
    raw=np.zeros_like(xi,dtype=float)
    for lesion in lesions:
        raw += lesion.amplitude*localized_morphology(xi,lesion.xi_c,lesion.w,lesion.p,Lg)
    M=float(np.max(raw))
    if not np.isfinite(M) or M<=0:
        raise ValueError("Multiple-lesion normalization is undefined.")
    return raw/M


def distributed_morphology(xi: np.ndarray, modes: Sequence[DistributedMode]) -> np.ndarray:
    if not modes:
        raise ValueError("At least one distributed mode is required for Psi_R.")
    raw=np.zeros_like(xi,dtype=float)
    for mode in modes:
        raw += mode.amplitude*np.cos(mode.q*xi+mode.phase)
    lo=float(np.min(raw)); hi=float(np.max(raw))
    span=hi-lo
    if not np.isfinite(span) or span<=100*np.finfo(float).eps*max(1.0,abs(lo),abs(hi)):
        raise ValueError("Distributed morphology must have nonzero spatial range before normalization.")
    return (raw-lo)/span


def morphology_field(spec: CaseSpec, grid: SpectralGrid) -> np.ndarray:
    """Evaluate the Mathematical Model morphology Psi_D without any radius closure."""
    if spec.case_class in {"H0","P0","P1","MM"}:
        psi=np.zeros(grid.N,dtype=float)
    elif spec.case_class=="DL":
        psi=localized_morphology(grid.xi,spec.xi_c,spec.w,spec.p,spec.Lg)
    elif spec.case_class=="DM":
        psi=multiple_morphology(grid.xi,spec.lesions,spec.Lg)
    elif spec.case_class=="DR":
        if spec.sampled_psi:
            if len(spec.sampled_psi)!=grid.N:
                raise ValueError("sampled_psi must contain exactly N collocation values; no hidden interpolation is performed.")
            psi=np.asarray(spec.sampled_psi,dtype=float)
        else:
            psi=distributed_morphology(grid.xi,spec.distributed_modes)
    else:
        raise RuntimeError("Unhandled case class")
    if not np.all(np.isfinite(psi)):
        raise ValueError("Psi_D contains non-finite values.")
    tol=2e-13
    if float(np.min(psi)) < -tol or float(np.max(psi)) > 1+tol:
        raise ValueError("Psi_D must satisfy 0 <= Psi_D <= 1.")
    return np.clip(psi,0.0,1.0)


def morphology_projection_error(raw: np.ndarray, grid: SpectralGrid) -> float:
    fhat=np.fft.fft(raw)
    den=np.linalg.norm(fhat)
    if den==0: return 0.0
    return float(np.linalg.norm(fhat[~grid.mask])/den)


def coeff_projection_error(raw: np.ndarray, grid: SpectralGrid) -> float:
    fhat=np.fft.fft(raw)
    den=np.linalg.norm(fhat)
    if den==0: return 0.0
    return float(np.linalg.norm(fhat[~grid.mask])/den)


def coefficient_fields(spec: CaseSpec, psi_D: np.ndarray, grid: SpectralGrid,
                       consts: ModelConstants=CONSTS) -> Tuple[np.ndarray,np.ndarray]:
    """Evaluate Eqs. b=b0[1+eps_b cos(q xi)+chi_b Psi_D] and g analogously."""
    b0=consts.b_ref*spec.Wo0**-2
    g0=consts.g_ref*(1.0+consts.Cg/spec.Wo0)
    if spec.case_class=="MM":
        return np.full(grid.N,float(spec.b_mean_override)), np.full(grid.N,float(spec.g_mean_override))
    b=b0*(1.0+spec.eps_b*np.cos(spec.q*grid.xi)+spec.chi_b*psi_D)
    g=g0*(1.0+spec.eps_g*np.cos(spec.q*grid.xi)+spec.chi_g*psi_D)
    if spec.case_class in {"H0","P0"}:
        b.fill(b0); g.fill(g0)
    return b,g


def _characteristic_morphology_scale(spec: CaseSpec) -> Optional[float]:
    if spec.case_class=="DL": return float(spec.w)
    if spec.case_class=="DM": return float(min(x.w for x in spec.lesions))
    if spec.case_class=="DR": return None if spec.morphology_scale is None else float(spec.morphology_scale)
    return None


def _morphology_provenance(spec: CaseSpec) -> str:
    if spec.case_class=="DL": return "analytic: Psi_L"
    if spec.case_class=="DM": return "analytic: normalized Psi_M"
    if spec.case_class=="DR" and spec.sampled_psi: return spec.morphology_provenance.strip()
    if spec.case_class=="DR": return "analytic: normalized Psi_R"
    if spec.case_class=="P1": return "parent sinusoidal coefficient modulation; Psi_D=0"
    if spec.case_class=="MM": return "matched mean generated from heterogeneous parent"
    return "Psi_D=0"


def admissibility_report(spec: CaseSpec, grid: SpectralGrid, psi_D: np.ndarray,
                         b: np.ndarray, g: np.ndarray,
                         consts: ModelConstants=CONSTS) -> Dict[str,Any]:
    bbar=float(np.mean(b)); gbar=float(np.mean(g))
    hb=float(np.max(np.abs(b-bbar))/bbar) if bbar>0 else np.inf
    hg=float(np.max(np.abs(g-gbar))/gbar) if gbar>0 else np.inf
    disease=spec.case_class in {"DL","DM","DR"}
    ellD_over_L0=_characteristic_morphology_scale(spec)
    lw={"required":disease,"R0_over_L0":spec.R0_over_L0,"ellD_over_L0":ellD_over_L0,
        "limit":spec.slow_variation_limit,"R0_over_ellD":None,"verified":not disease}
    if disease and spec.R0_over_L0 is not None and ellD_over_L0 is not None and ellD_over_L0>0:
        lw["R0_over_ellD"]=float(spec.R0_over_L0/ellD_over_L0)
        if spec.slow_variation_limit is not None:
            lw["verified"]=bool(lw["R0_over_ellD"] < spec.slow_variation_limit)

    psi_ok=(np.all(np.isfinite(psi_D)) and float(np.min(psi_D))>=-2e-13 and float(np.max(psi_D))<=1+2e-13)
    positives=(float(np.min(b))>0 and float(np.min(g))>0)
    hetero=max(hb,hg)<=consts.heterogeneity_limit+1e-14
    bg_ok=(abs(spec.eps_b)<=0.3+1e-14 and abs(spec.eps_g)<=0.3+1e-14)
    sufficient_positive=(abs(spec.eps_b)+abs(spec.chi_b)<1 and abs(spec.eps_g)+abs(spec.chi_g)<1)

    if not psi_ok or not positives or not hetero or not bg_ok:
        status="OUTSIDE_MODEL_RANGE"
    elif disease and not lw["verified"]:
        missing=(spec.R0_over_L0 is None or spec.slow_variation_limit is None or ellD_over_L0 is None)
        status="ASSUMPTION_INPUT_REQUIRED" if missing else "OUTSIDE_MODEL_RANGE"
    else:
        status="ADMISSIBLE"
    return {
        "status":status,
        "psi_min":float(np.min(psi_D)),"psi_max":float(np.max(psi_D)),"psi_mean":float(np.mean(psi_D)),
        "b_min":float(np.min(b)),"g_min":float(np.min(g)),
        "b_heterogeneity":hb,"g_heterogeneity":hg,"heterogeneity_limit":consts.heterogeneity_limit,
        "background_modulation_ok":bool(bg_ok),"sufficient_positivity_bound":bool(sufficient_positive),
        "long_wave":lw,
    }


def prepare_case(spec: CaseSpec, consts: ModelConstants=CONSTS) -> PreparedCase:
    grid=make_grid(spec.N,spec.Lg)
    psi_raw=morphology_field(spec,grid)
    morphology_error=morphology_projection_error(psi_raw,grid)
    psi=project_real(psi_raw,grid)
    b_raw,g_raw=coefficient_fields(spec,psi_raw,grid,consts)
    coeff_error=max(coeff_projection_error(b_raw,grid),coeff_projection_error(g_raw,grid))
    b=project_real(b_raw,grid); g=project_real(g_raw,grid)

    raw_report=admissibility_report(spec,grid,psi_raw,b_raw,g_raw,consts)
    resolved_report=admissibility_report(spec,grid,psi,b,g,consts)
    report=dict(raw_report)
    report["raw_status"]=raw_report["status"]
    report["resolved_status"]=resolved_report["status"]
    report["resolved_psi_min"]=resolved_report["psi_min"]
    report["resolved_psi_max"]=resolved_report["psi_max"]
    report["resolved_b_min"]=resolved_report["b_min"]
    report["resolved_g_min"]=resolved_report["g_min"]
    report["resolved_b_heterogeneity"]=resolved_report["b_heterogeneity"]
    report["resolved_g_heterogeneity"]=resolved_report["g_heterogeneity"]
    report["morphology_projection_error"]=morphology_error
    report["morphology_projection_limit"]=spec.morphology_projection_limit
    report["coefficient_projection_error"]=coeff_error
    report["coefficient_projection_limit"]=spec.coeff_projection_limit

    if raw_report["status"]=="OUTSIDE_MODEL_RANGE" or resolved_report["status"]=="OUTSIDE_MODEL_RANGE":
        report["status"]="OUTSIDE_MODEL_RANGE"
    elif raw_report["status"]=="ASSUMPTION_INPUT_REQUIRED" or resolved_report["status"]=="ASSUMPTION_INPUT_REQUIRED":
        report["status"]="ASSUMPTION_INPUT_REQUIRED"
    elif morphology_error>spec.morphology_projection_limit or coeff_error>spec.coeff_projection_limit:
        report["status"]="UNDER_RESOLVED"
    else:
        report["status"]="ADMISSIBLE"

    payload={"spec":_jsonable(spec),"model_schema":MODEL_SCHEMA,"solver_schema":SOLVER_SCHEMA,"result_schema":RESULT_SCHEMA}
    cid=f"{spec.case_class}-{stable_hash(payload)}"
    bb=float(np.mean(b)); gg=float(np.mean(g))
    return PreparedCase(
        spec=spec,case_id=cid,grid=grid,psi_D_raw=psi_raw,psi_D=psi,b=b,g=g,
        b_bar=bb,g_bar=gg,b_tilde=b-bb,g_tilde=g-gg,
        morphology_error=morphology_error,coeff_error=coeff_error,admissibility=report,
        morphology_provenance=_morphology_provenance(spec),parent_case_id=None,
    )


def make_matched_mean(parent: PreparedCase) -> PreparedCase:
    if parent.spec.case_class not in {"P1","DL","DM","DR"}:
        raise ValueError("Matched-mean controls require a heterogeneous P1/DL/DM/DR parent case.")
    spec=replace(
        parent.spec,case_class="MM",eps_b=0.0,eps_g=0.0,chi_b=0.0,chi_g=0.0,
        lesions=(),distributed_modes=(),sampled_psi=(),morphology_provenance="",
        b_mean_override=float(parent.b_bar),g_mean_override=float(parent.g_bar),
        notes=f"Exact matched-mean control generated from {parent.case_id}",
    )
    mm=prepare_case(spec)
    mm.parent_case_id=parent.case_id
    mm.admissibility["matched_mean_parent"]=parent.case_id
    return mm


def initial_condition(spec: CaseSpec, grid: SpectralGrid) -> np.ndarray:
    A1=spec.A1; A2=spec.A2_ratio*A1; A3=spec.A3_ratio*A1
    p1,p2,p3=spec.phases
    x=grid.xi
    a=A1*np.sin(spec.k0*x+p1)+A2*np.sin(2*spec.k0*x+p2)+A3*np.sin(3*spec.k0*x+p3)
    return project_real(a,grid)


def phi_functions(z: np.ndarray, small: float=1e-6) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    z=np.asarray(z,dtype=np.complex128)
    p1=np.empty_like(z); p2=np.empty_like(z); p3=np.empty_like(z)
    m=np.abs(z)<small
    zz=z[m]
    # Taylor expansions through z^5.
    p1[m]=1+zz/2+zz**2/6+zz**3/24+zz**4/120+zz**5/720
    p2[m]=0.5+zz/6+zz**2/24+zz**3/120+zz**4/720+zz**5/5040
    p3[m]=1/6+zz/24+zz**2/120+zz**3/720+zz**4/5040+zz**5/40320
    z2=z[~m]; em1=np.expm1(z2)
    p1[~m]=em1/z2
    p2[~m]=(em1-z2)/(z2**2)
    p3[~m]=(em1-z2-z2**2/2)/(z2**3)
    return p1,p2,p3

@dataclass
class ETDCoefficients:
    L0: np.ndarray; E: np.ndarray; E2: np.ndarray; Q: np.ndarray; f1: np.ndarray; f2: np.ndarray; f3: np.ndarray


def etd_coefficients(prep: PreparedCase, dt: Optional[float]=None) -> ETDCoefficients:
    h=prep.spec.dt if dt is None else dt; k=prep.grid.k
    L0=-1j*prep.b_bar*k**3-prep.g_bar*np.abs(k)
    z=h*L0; E=np.exp(z); E2=np.exp(z/2)
    p1h,_,_=phi_functions(z/2)
    p1,p2,p3=phi_functions(z)
    Q=(h/2)*p1h
    f1=h*(p1-3*p2+4*p3)
    f2=h*(p2-2*p3)
    f3=h*(-p2+4*p3)
    return ETDCoefficients(L0,E,E2,Q,f1,f2,f3)


def residual_hat(ahat: np.ndarray, prep: PreparedCase, include_nonlinearity: bool=True, include_heterogeneity: bool=True) -> np.ndarray:
    grid=prep.grid
    ah=project_hat(ahat,grid)
    a=np.fft.ifft(ah)
    out=np.zeros(grid.N,dtype=np.complex128)
    if include_nonlinearity:
        out += project_hat((1j*grid.k/2)*np.fft.fft(a*a),grid)
    if include_heterogeneity:
        axxx=np.fft.ifft(1j*grid.k**3*ah)
        lam=np.fft.ifft(np.abs(grid.k)*ah)
        out += project_hat(np.fft.fft(-prep.b_tilde*axxx-prep.g_tilde*lam),grid)
    return project_hat(out,grid)


def full_rhs_hat(ahat: np.ndarray, prep: PreparedCase, include_nonlinearity: bool=True, include_heterogeneity: bool=True) -> np.ndarray:
    etd=etd_coefficients(prep)
    return project_hat(etd.L0*project_hat(ahat,prep.grid)+residual_hat(ahat,prep,include_nonlinearity,include_heterogeneity),prep.grid)


def etdrk4_step(ahat: np.ndarray, prep: PreparedCase, etd: ETDCoefficients, include_nonlinearity: bool=True, include_heterogeneity: bool=True) -> np.ndarray:
    grid=prep.grid; v=project_hat(ahat,grid)
    Nv=residual_hat(v,prep,include_nonlinearity,include_heterogeneity)
    a1=project_hat(etd.E2*v+etd.Q*Nv,grid)
    Na=residual_hat(a1,prep,include_nonlinearity,include_heterogeneity)
    a2=project_hat(etd.E2*v+etd.Q*Na,grid)
    Nb=residual_hat(a2,prep,include_nonlinearity,include_heterogeneity)
    a3=project_hat(etd.E2*a1+etd.Q*(2*Nb-Nv),grid)
    Nc=residual_hat(a3,prep,include_nonlinearity,include_heterogeneity)
    return project_hat(etd.E*v+etd.f1*Nv+2*etd.f2*(Na+Nb)+etd.f3*Nc,grid)


def integrals_and_balance(ahat: np.ndarray, prep: PreparedCase, rhs_hat: Optional[np.ndarray]=None) -> Dict[str,float]:
    grid=prep.grid
    a=np.fft.ifft(project_hat(ahat,grid)).real
    if rhs_hat is None: rhs_hat=full_rhs_hat(ahat,prep)
    rhs=np.fft.ifft(rhs_hat).real
    I1=grid.dx*np.sum(a); I2=grid.dx*np.sum(a*a); E=0.5*I2
    bh=np.fft.fft(prep.b); gh=np.fft.fft(prep.g)
    b1=np.fft.ifft((-1j*grid.k)*bh).real
    b3=np.fft.ifft((1j*grid.k**3)*bh).real
    lamg=np.fft.ifft(np.abs(grid.k)*gh).real
    ax=np.fft.ifft((-1j*grid.k)*np.fft.fft(a)).real
    lama=np.fft.ifft(np.abs(grid.k)*np.fft.fft(a)).real
    B1=grid.dx*np.sum(a*b3-a*lamg)
    BE=grid.dx*np.sum(0.5*b3*a*a-1.5*b1*ax*ax-prep.g*a*lama)
    dI1_num=grid.dx*np.sum(rhs); dE_num=grid.dx*np.sum(a*rhs)
    G_bal=2*BE/I2 if I2>0 else np.nan
    return {"I1":float(I1),"I2":float(I2),"E":float(E),"B1":float(B1),"BE":float(BE),
            "dI1_num":float(dI1_num),"dE_num":float(dE_num),"R_I1_inst":float(dI1_num-B1),
            "R_E_inst":float(dE_num-BE),"G_bal":float(G_bal)}


def spectral_broadening(ahat: np.ndarray, prep: PreparedCase) -> Dict[str,float]:
    # Use normalized coefficients so energies have a direct Parseval interpretation.
    hn=project_hat(ahat,prep.grid)/prep.grid.N
    kabs=np.abs(prep.grid.k); kc=CONSTS.k_cutoff_ratio*(3*prep.spec.k0)
    low=np.sum(np.abs(hn[kabs<=kc])**2)
    high=np.sum(np.abs(hn[(kabs>kc)&prep.grid.mask])**2)
    R=float(high/low) if low>0 else np.inf
    tail_sel=(kabs>0.8*prep.grid.k_ret)&prep.grid.mask
    tot=np.sum(np.abs(hn[prep.grid.mask])**2)
    tail=float(np.sum(np.abs(hn[tail_sel])**2)/tot) if tot>0 else 0.0
    return {"R":R,"E_low_spec":float(low),"E_high_spec":float(high),"eta_tail":tail,"k_c":float(kc)}


def modal_energy_budget(ahat: np.ndarray, prep: PreparedCase) -> Dict[str,np.ndarray]:
    grid=prep.grid; ah=project_hat(ahat,grid); ahn=ah/grid.N
    a=np.fft.ifft(ah)
    # RHS components in raw FFT normalization.
    F_N=project_hat((1j*grid.k/2)*np.fft.fft(a*a),grid)
    axxx=np.fft.ifft(1j*grid.k**3*ah); lam=np.fft.ifft(np.abs(grid.k)*ah)
    F_tb=project_hat(np.fft.fft(-prep.b_tilde*axxx),grid)
    F_tg=project_hat(np.fft.fft(-prep.g_tilde*lam),grid)
    F_bg=project_hat(-prep.g_bar*np.abs(grid.k)*ah,grid)
    F_bb=project_hat(-1j*prep.b_bar*grid.k**3*ah,grid)
    comps={"N":F_N,"b_tilde":F_tb,"g_tilde":F_tg,"g_bar":F_bg,"b_bar":F_bb}
    T={name:2*grid.Lg*np.real(np.conj(ahn)*(F/grid.N)) for name,F in comps.items()}
    Tsum=T["N"]+T["b_tilde"]+T["g_tilde"]+T["g_bar"]+T["b_bar"]
    full=sum(comps.values()); dEk=2*grid.Lg*np.real(np.conj(ahn)*(full/grid.N))
    high=(np.abs(grid.k)>CONSTS.k_cutoff_ratio*3*prep.spec.k0)&grid.mask
    out={"dEk":dEk,"T_sum":Tsum,"closure":dEk-Tsum,"high_mask":high}
    out.update({f"T_{k}":v for k,v in T.items()})
    out.update({f"PiH_{k}":np.array(float(np.sum(v[high]))) for k,v in T.items()})
    out["dEhigh"]=np.array(float(np.sum(dEk[high])))
    out["PiH_sum"]=np.array(float(np.sum(Tsum[high])))
    return out


def topology_from_curve(Wo: Sequence[float], y: Sequence[float], flat_rel_tol: float=1e-3) -> Dict[str,Any]:
    x=np.asarray(Wo,float); v=np.asarray(y,float)
    if x.ndim!=1 or len(x)<3 or len(x)!=len(v): raise ValueError("Need matching 1D Wo and response arrays with at least 3 points.")
    idx=np.argsort(x); x=x[idx]; v=v[idx]
    span=float(np.max(v)-np.min(v)); scale=max(float(np.max(np.abs(v))),1e-15)
    if span/scale < flat_rel_tol:
        return {"class":"flat/unresolved","peaks":[],"global_index":int(np.argmax(v))}
    peaks,_=find_peaks(v)
    if len(peaks)==0:
        dv=np.diff(v)
        if np.all(dv>=0) or np.all(dv<=0): cls="monotone"
        else: cls="boundary maximum"
        return {"class":cls,"peaks":[],"global_index":int(np.argmax(v))}
    proms=peak_prominences(v,peaks)[0]
    pinfo=[{"index":int(i),"Wo":float(x[i]),"value":float(v[i]),"prominence":float(p)} for i,p in zip(peaks,proms)]
    cls="single interior peak" if len(peaks)==1 else "multiple peaks"
    return {"class":cls,"peaks":pinfo,"global_index":int(np.argmax(v))}


def half_prominence_width(Wo: Sequence[float], y: Sequence[float], peak_index: int) -> Dict[str,Any]:
    x=np.asarray(Wo,float); v=np.asarray(y,float); i=int(peak_index)
    if i<=0 or i>=len(v)-1: return {"resolved":False,"reason":"peak at boundary"}
    # conservative local baseline = larger of minima to left/right; half-prominence level.
    left_base=float(np.min(v[:i+1])); right_base=float(np.min(v[i:])); base=max(left_base,right_base)
    level=base+0.5*(float(v[i])-base)
    def cross_left():
        for j in range(i-1,-1,-1):
            if (v[j]-level)*(v[j+1]-level)<=0 and v[j]!=v[j+1]:
                return float(x[j]+(level-v[j])*(x[j+1]-x[j])/(v[j+1]-v[j]))
        return None
    def cross_right():
        for j in range(i,len(v)-1):
            if (v[j]-level)*(v[j+1]-level)<=0 and v[j]!=v[j+1]:
                return float(x[j]+(level-v[j])*(x[j+1]-x[j])/(v[j+1]-v[j]))
        return None
    l=cross_left(); r=cross_right()
    if l is None or r is None: return {"resolved":False,"reason":"two half-prominence crossings not bracketed","level":level}
    return {"resolved":True,"left":l,"right":r,"width":r-l,"level":level}


def direct_modal_rhs_normalized(a: np.ndarray, b: np.ndarray, g: np.ndarray, grid: SpectralGrid, mask_output: bool=True) -> np.ndarray:
    ah=np.fft.fft(a)/grid.N; bh=np.fft.fft(b)/grid.N; gh=np.fft.fft(g)/grid.N
    N=grid.N; out=np.zeros(N,complex)
    for l in range(N):
        kn=grid.k
        conv=0j
        for p in range(N): conv += ah[p]*ah[(l-p)%N]
        term=(1j*grid.k[l]/2)*conv
        lin=0j
        for n in range(N):
            m=(l-n)%N
            lin += (-1j*bh[m]*grid.k[n]**3-gh[m]*abs(grid.k[n]))*ah[n]
        out[l]=term+lin
    if mask_output: out[~grid.mask]=0
    return out


def dense_linear_matrix(prep: PreparedCase) -> Tuple[np.ndarray,np.ndarray]:
    grid=prep.grid; inds=np.flatnonzero(grid.mask)
    bh=np.fft.fft(prep.b)/grid.N; gh=np.fft.fft(prep.g)/grid.N
    A=np.zeros((len(inds),len(inds)),complex)
    for il,l in enumerate(inds):
        for jn,n in enumerate(inds):
            m=(l-n)%grid.N
            A[il,jn]=-1j*bh[m]*grid.k[n]**3-gh[m]*abs(grid.k[n])
    return A,inds


def verify_fft_sign() -> Dict[str,Any]:
    g=make_grid(96,4*np.pi); x=g.xi
    f=np.sin(2.0*x)+0.4*np.cos(3.0*x)
    exact=2*np.cos(2*x)-1.2*np.sin(3*x)
    num=derivative_from_hat(np.fft.fft(f),g,1).real
    err=float(np.max(np.abs(num-exact)))
    return {"name":"fft_sign","error":err,"pass":err<1e-11}


def verify_three_form() -> Dict[str,Any]:
    N=24; g=make_grid(N,4*np.pi); x=g.xi
    # Low-frequency deterministic state and coefficients prevent truncation ambiguity.
    a=0.7*np.sin(0.5*x)+0.2*np.cos(x)-0.1*np.sin(1.5*x)
    b=0.02*(1+0.1*np.cos(x)); gg=0.006*(1+0.08*np.sin(x))
    # Physical form.
    ah=np.fft.fft(a); axxx=derivative_from_hat(ah,g,3); lam=lambda_from_hat(ah,g)
    rhs_phys=-0.5*derivative_from_hat(np.fft.fft(a*a),g,1)-b*axxx-gg*lam
    rh=np.fft.fft(rhs_phys)/N; rh[~g.mask]=0
    direct=direct_modal_rhs_normalized(a,b,gg,g,True)
    # Split form via a temporary PreparedCase.
    spec=CaseSpec("H0",Wo0=10,N=N,Lg=4*np.pi,dt=1e-3,T_final=0.1)
    z=np.zeros(N)
    prep=PreparedCase(spec=spec,case_id="audit",grid=g,psi_D_raw=z.copy(),psi_D=z.copy(),b=b,g=gg,
                      b_bar=float(np.mean(b)),g_bar=float(np.mean(gg)),b_tilde=b-np.mean(b),g_tilde=gg-np.mean(gg),
                      morphology_error=0.0,coeff_error=0.0,admissibility={"status":"ADMISSIBLE"})
    split=full_rhs_hat(ah,prep)/N
    e1=float(np.max(np.abs(rh-direct))); e2=float(np.max(np.abs(rh-split))); e3=float(np.max(np.abs(direct-split)))
    return {"name":"three_form","physical_vs_modal":e1,"physical_vs_split":e2,"modal_vs_split":e3,"pass":max(e1,e2,e3)<1e-11}


def verify_constant_linear() -> Dict[str,Any]:
    spec=CaseSpec("H0",Wo0=10,N=64,dt=0.01,T_final=0.2,k0=1.0)
    prep=prepare_case(spec); ah=np.fft.fft(initial_condition(spec,prep.grid)); etd=etd_coefficients(prep)
    n=int(round(spec.T_final/spec.dt)); a0=ah.copy()
    for _ in range(n): ah=etdrk4_step(ah,prep,etd,False,False)
    exact=np.exp(spec.T_final*etd.L0)*a0; exact=project_hat(exact,prep.grid)
    err=float(np.linalg.norm(ah-exact)/np.linalg.norm(exact))
    return {"name":"constant_linear","relative_error":err,"pass":err<1e-12}


def verify_heterogeneous_linear_order() -> Dict[str,Any]:
    N=24; T=1.0; base=CaseSpec("P1",Wo0=2,N=N,dt=0.1,T_final=T,eps_b=0.2,eps_g=0.2,q=1.0,k0=0.5)
    prep0=prepare_case(base); A,inds=dense_linear_matrix(prep0); ah0=np.fft.fft(initial_condition(base,prep0.grid)); exact=expm(T*A)@ah0[inds]
    dts=[0.1,0.05,0.025,0.0125]; errs=[]
    for dt in dts:
        spec=replace(base,dt=dt,T_final=T); prep=prepare_case(spec); ah=ah0.copy(); etd=etd_coefficients(prep,dt)
        steps=int(round(T/dt)); actual=steps*dt
        if abs(actual-T)>1e-12: raise AssertionError("T must be divisible by dt in benchmark")
        for _ in range(steps): ah=etdrk4_step(ah,prep,etd,False,True)
        errs.append(float(np.linalg.norm(ah[inds]-exact)/np.linalg.norm(exact)))
    orders=[float(np.log(errs[i]/errs[i+1])/np.log(2)) for i in range(len(errs)-1)]
    passed=(orders[-1]>3.5 and errs[-1]<errs[0])
    return {"name":"heterogeneous_linear_order","dt":dts,"errors":errs,"orders":orders,"pass":passed}


def verify_balance_identities() -> Dict[str,Any]:
    N=96; g=make_grid(N,4*np.pi); x=g.xi
    spec=CaseSpec("P1",Wo0=10,N=N,dt=1e-3,T_final=.1,eps_b=.12,eps_g=.1,q=1.0,k0=.5)
    prep=prepare_case(spec)
    a=0.8*np.sin(.5*x)+0.2*np.cos(x)+0.08*np.sin(1.5*x)
    ah=project_hat(np.fft.fft(a),g); rhs=full_rhs_hat(ah,prep)
    bal=integrals_and_balance(ah,prep,rhs)
    scale1=max(abs(bal["B1"]),1.0); scaleE=max(abs(bal["BE"]),1.0)
    e1=abs(bal["R_I1_inst"])/scale1; eE=abs(bal["R_E_inst"])/scaleE
    return {"name":"balance_identities","I1_residual":bal["R_I1_inst"],"E_residual":bal["R_E_inst"],"relative_I1":e1,"relative_E":eE,"pass":max(e1,eE)<1e-10}


def verify_modal_budget() -> Dict[str,Any]:
    spec=CaseSpec("P1",Wo0=10,N=96,dt=1e-3,T_final=.1,eps_b=.12,eps_g=.1,q=1.0,k0=.5)
    prep=prepare_case(spec); ah=np.fft.fft(initial_condition(spec,prep.grid)); B=modal_energy_budget(ah,prep)
    e=float(np.max(np.abs(B["closure"]))); eb=abs(float(B["dEhigh"]-B["PiH_sum"])); phase=float(np.max(np.abs(B["T_b_bar"])))
    scale=max(float(np.max(np.abs(B["dEk"]))),1.0)
    return {"name":"modal_budget","closure_max":e,"highband_closure":eb,"mean_dispersion_energy_contribution":phase,"pass":max(e,eb,phase)/scale<1e-11}


def verify_kdv_refinement() -> Dict[str,Any]:
    base=CaseSpec("H0",Wo0=10,N=128,dt=0.002,T_final=0.12,k0=1.0)
    # Build a custom KdV PreparedCase with constant b and zero g.
    g=make_grid(base.N,base.Lg); bval=0.02
    def run(dt):
        spec=replace(base,dt=dt); b=np.full(g.N,bval); gg=np.zeros(g.N)
        z=np.zeros(g.N)
        prep=PreparedCase(spec=spec,case_id="kdv",grid=g,psi_D_raw=z.copy(),psi_D=z.copy(),b=b,g=gg,
                          b_bar=bval,g_bar=0.0,b_tilde=z.copy(),g_tilde=z.copy(),morphology_error=0.0,
                          coeff_error=0.0,admissibility={"status":"ADMISSIBLE"})
        a0=initial_condition(spec,g); ah=np.fft.fft(a0); etd=etd_coefficients(prep,dt)
        def inv(ah):
            a=np.fft.ifft(ah).real; ax=np.fft.ifft((-1j*g.k)*ah).real
            return np.array([g.dx*np.sum(a),g.dx*np.sum(a*a),g.dx*np.sum(a**3/6-bval*ax*ax/2)])
        q0=inv(ah)
        for _ in range(int(round(spec.T_final/dt))): ah=etdrk4_step(ah,prep,etd,True,False)
        q1=inv(ah); return float(np.linalg.norm(q1-q0)/max(np.linalg.norm(q0),1e-15))
    d1=run(0.002); d2=run(0.001)
    return {"name":"kdv_refinement","drift_dt":d1,"drift_dt_over_2":d2,"pass":d2 < d1*1.05}


def quick_numerical_check() -> Dict[str,Any]:
    tests=[verify_fft_sign(),verify_three_form(),verify_constant_linear(),verify_balance_identities(),verify_modal_budget()]
    return {"tests":tests,"pass":all(t["pass"] for t in tests)}


def core_verification_suite(include_kdv: bool=True) -> Dict[str,Any]:
    tests=[verify_fft_sign(),verify_three_form(),verify_constant_linear(),verify_heterogeneous_linear_order(),verify_balance_identities(),verify_modal_budget()]
    if include_kdv: tests.append(verify_kdv_refinement())
    return {"schema":SOLVER_SCHEMA,"tests":tests,"pass":all(t["pass"] for t in tests),"timestamp":time.time()}



__all__ = [name for name in globals() if not name.startswith("_")]
