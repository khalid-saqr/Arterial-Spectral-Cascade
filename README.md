# Arterial Spectral Cascade

[![Research software CI](https://github.com/khalid-saqr/Arterial-Spectral-Cascade/actions/workflows/ci.yml/badge.svg)](https://github.com/khalid-saqr/Arterial-Spectral-Cascade/actions/workflows/ci.yml)

Research software for spatially heterogeneous extensions of the fractional-KdV model introduced in:

**K. M. Saqr, “Resonant spectral cascade in Womersley flow triggered by arterial geometry,” _Physics of Fluids_ 38, 041901 (2026), DOI `10.1063/5.0319995`.**

The package retains the pointwise dispersion and fractional-damping coefficient fields that were averaged in the published numerical calculation. It is intended for controlled coefficient-space studies of how spatial arterial morphology can alter internal spectral coupling within the same reduced-order Womersley-based model.

> **Permission is required before use.** Public availability of this repository does not grant permission to run, execute, reproduce, modify, distribute, incorporate, or use this software for research, commercial, educational, or other purposes. Before using the package, obtain prior written permission from **khalid.saqr@knowdyn.co.uk**. See [LICENSE](LICENSE). Citation does not substitute for permission.

## Contents

- [Scientific scope](#scientific-scope)
- [Mathematical model](#mathematical-model)
- [Disease morphology input](#disease-morphology-input)
- [Case hierarchy and matched-mean controls](#case-hierarchy-and-matched-mean-controls)
- [Admissibility and reduced-order validity](#admissibility-and-reduced-order-validity)
- [Numerical solver](#numerical-solver)
- [Diagnostics and scientific outputs](#diagnostics-and-scientific-outputs)
- [Verification and numerical acceptance](#verification-and-numerical-acceptance)
- [Installation](#installation)
- [Researcher quick start](#researcher-quick-start)
- [Defining disease cases](#defining-disease-cases)
- [Geometry-derived morphology](#geometry-derived-morphology)
- [Running a complete study](#running-a-complete-study)
- [Persistent results and restart](#persistent-results-and-restart)
- [Performance backend](#performance-backend)
- [Interpreting results](#interpreting-results)
- [Package organization](#package-organization)
- [Reproducibility, citation, and license](#reproducibility-citation-and-license)

## Scientific scope

The solved system is a **reduced-order, one-dimensional, periodic, weakly nonlinear model**. The extension is deliberately narrow.

The package:

- preserves the parent Womersley-dependent baseline scaling;
- preserves the quadratic nonlinearity and fractional order;
- retains pointwise spatial variation in the effective dispersion and fractional-damping coefficients;
- represents disease through a bounded morphology field and signed coefficient sensitivities;
- resolves the resulting heterogeneous Fourier coupling;
- pairs every heterogeneous disease calculation with an exact matched-mean control;
- provides exact integral-balance diagnostics, convergence tests, restart integrity checks, and modal-energy budgets.

The package does **not** add or infer:

- a local radius-dependent Womersley number;
- a local stenosis-to-coefficient or aneurysm-to-coefficient law;
- wall compliance or fluid-structure interaction;
- plaque or thrombus mechanics;
- separated-flow or secondary-flow closure;
- arterial branching;
- patient-specific boundary conditions;
- direct clinical-outcome prediction.

Anatomical information can enter only through an admissible normalized morphology field $\Psi_D$ and externally justified or parametrically varied signed sensitivities $\chi_b,\chi_g$.

## Mathematical model

The governing equation is

$$
a_s+a\,a_\xi+b(\xi)\,a_{\xi\xi\xi}+g(\xi)\Lambda a=0,
\qquad
\Lambda=(-\partial_\xi^2)^{1/2},
$$

on the periodic domain

$$
\xi\in[0,L_g],\qquad L_g=4\pi.
$$

The Riesz operator is positive and first order:

$$
\widehat{\Lambda f}(k)=|k|\widehat f(k).
$$

The Womersley-dependent baseline coefficients are inherited from the published model:

$$
b_0(\mathrm{Wo})=b_{\mathrm{ref}}\mathrm{Wo}^{-2},
$$

$$
g_0(\mathrm{Wo})
=g_{\mathrm{ref}}\left(1+\frac{C_g}{\mathrm{Wo}}\right),
\qquad
g_{\mathrm{ref}}=0.005,
\qquad
C_g=0.1.
$$

The full coefficient fields are

$$
b(\xi)
=b_0(\mathrm{Wo})
\left[1+\varepsilon_b\cos(q\xi)+\chi_b\Psi_D(\xi)\right],
$$

$$
g(\xi)
=g_0(\mathrm{Wo})
\left[1+\varepsilon_g\cos(q\xi)+\chi_g\Psi_D(\xi)\right].
$$

Here:

- $\Psi_D$ is a smooth periodic morphology field satisfying $0\leq\Psi_D\leq1$;
- $\chi_b$ is the signed effective sensitivity of dispersion to the morphology field;
- $\chi_g$ is the signed effective sensitivity of fractional damping to the morphology field;
- $\varepsilon_b,\varepsilon_g,q$ describe the parent sinusoidal coefficient modulation.

For the primary disease-only study,

$$
\varepsilon_b=\varepsilon_g=0,
$$

so that the disease morphology is the only source of coefficient heterogeneity. A combined disease-plus-parent-background case is permitted only when it is identified explicitly in the study record.

### Initial condition

The inherited multi-harmonic form is

$$
a(\xi,0)
=\sum_{n=1}^{3}A_n\sin(nk_0\xi+\phi_n),
$$

with

$$
A_2/A_1=0.3,\qquad A_3/A_1=0.1.
$$

`CaseSpec` exposes $A_1$, the amplitude ratios, phases, and $k_0$. The implementation defaults are intended as reproducible package defaults; researchers should use values justified by the study they are reproducing or extending. The published detailed and Womersley-sweep calculations used $k_0=0.5$ and $k_0=1.0$, respectively, as recovery references.

## Disease morphology input

The package provides three disease morphology classes. They are alternative choices for the same $\Psi_D$; they are not different governing equations.

### `DL`: single localized morphology

$$
\Psi_L(\xi;\xi_c,w,p)
=
\exp\left\{
-\left[
\frac{L_g}{\pi w}
\sin\left(\frac{\pi(\xi-\xi_c)}{L_g}\right)
\right]^{2p}
\right\},
\qquad
w>0,\quad p\in\mathbb N,\ p\ge1.
$$

- `xi_c` controls lesion location;
- `w` controls local axial extent;
- `p` controls edge sharpness while preserving smooth periodicity.

### `DM`: normalized multiple-lesion morphology

For lesions with amplitudes $A_j$, locations $\xi_j$, widths $w_j$, and smoothness exponents $p_j$,

$$
\Psi_M(\xi)
=
\frac{1}{M}
\sum_{j=1}^{N_D}A_j\Psi_{L,j}(\xi),
$$

where $M$ is the maximum of the unnormalized sum, giving $0\leq\Psi_M\leq1$.

### `DR`: distributed irregular or geometry-derived morphology

An analytical distributed morphology begins with

$$
r_D(\xi)=\sum_{j=1}^{J}A_j\cos(q_j\xi+\varphi_j)
$$

and is normalized as

$$
\Psi_R(\xi)
=
\frac{r_D(\xi)-r_{\min}}{r_{\max}-r_{\min}}.
$$

A `DR` case may instead use a sampled geometry-derived $\Psi_D$. The sampled field must be supplied on the solver collocation grid, normalized consistently with $0\leq\Psi_D\leq1$, sufficiently smooth and periodic at the resolved scale, and accompanied by morphology provenance and a declared characteristic morphology scale.

The package performs **no hidden interpolation** and constructs **no local Womersley field** from geometry.

## Case hierarchy and matched-mean controls

| Class | Coefficients / morphology | Purpose |
| --- | --- | --- |
| `H0` | Constant $b_0,g_0$, no disease morphology | Homogeneous baseline and exact checks |
| `P0` | Spatial means only | Recovery of the published constant-coefficient numerical model |
| `P1` | Parent sinusoidal $b(\xi),g(\xi)$, $\Psi_D=0$ | Pointwise parent coefficient effect omitted by averaging |
| `DL` | Single localized $\Psi_L$ | Localized disease heterogeneity |
| `DM` | Normalized multiple-lesion $\Psi_M$ | Multiple localized disease heterogeneity |
| `DR` | Distributed $\Psi_R$ or sampled $\Psi_D$ | Distributed or geometry-derived heterogeneity |
| `MM` | Constant exact $\bar b,\bar g$ of a paired heterogeneous case | Isolate spatial arrangement from mean renormalization |

For any heterogeneous case,

$$
b=\bar b+\widetilde b,\qquad
g=\bar g+\widetilde g.
$$

For a parent sinusoid compatible with the periodic domain,

$$
\bar b=b_0(1+\chi_b\overline{\Psi}_D),
\qquad
\bar g=g_0(1+\chi_g\overline{\Psi}_D).
$$

The heterogeneous equation can therefore be written exactly as

$$
a_s+a a_\xi+\bar b a_{\xi\xi\xi}+\bar g\Lambda a
=
-\widetilde b\,a_{\xi\xi\xi}
-\widetilde g\,\Lambda a.
$$

The matched-mean control evolves

$$
a_s^{(m)}
+a^{(m)}a_\xi^{(m)}
+\bar b\,a_{\xi\xi\xi}^{(m)}
+\bar g\,\Lambda a^{(m)}=0
$$

with the **same initial condition, grid, timestep, output cadence, and final time**.

This comparison is central to the research design. It distinguishes:

1. changes caused by disease-induced shifts in the mean coefficient levels; from
2. changes caused by the spatial arrangement of coefficient heterogeneity itself.

## Admissibility and reduced-order validity

A disease case is not accepted merely because its numerical parameters can be constructed. The package checks the reduced-order model regime before integration.

### Required model conditions

1. **Smooth periodic morphology and coefficients.**
2. **Positive effective coefficients:**
   $$
   \inf_\xi b(\xi)>0,\qquad \inf_\xi g(\xi)>0.
   $$
3. **Weak-to-moderate heterogeneity:**
   $$
   \max\left(
   \frac{\|b-\bar b\|_\infty}{\bar b},
   \frac{\|g-\bar g\|_\infty}{\bar g}
   \right)\le0.3.
   $$
4. **Long-wave consistency.** For a localized morphology scale $\ell_D$,
   $$
   R_0/\ell_D\ll1.
   $$
   The package requires explicit reduced-order consistency inputs and operationalizes this asymptotic requirement with a declared `slow_variation_limit`.
5. **Resolved morphology and coefficient fields.** The two-thirds projection errors for $\Psi_D$, $b$, and $g$ must remain below configured limits.

The default study configuration uses:

```text
R0_OVER_L0                  = 0.05
SLOW_VARIATION_LIMIT        = 0.10
MORPHOLOGY_PROJECTION_LIMIT = 1e-8
COEFF_PROJECTION_LIMIT      = 1e-8
```

These are computational study settings, not universal physiological thresholds. Researchers must confirm that they are appropriate for the intended application.

Possible pre-run statuses include:

- `ADMISSIBLE`
- `OUTSIDE_MODEL_RANGE`
- `ASSUMPTION_INPUT_REQUIRED`
- `UNDER_RESOLVED`

Model admissibility and numerical validity are recorded separately.

## Numerical solver

### Fourier convention

The continuum convention is

$$
f(\xi)=\sum_{\ell\in\mathbb Z}\widehat f_\ell e^{-ik_\ell\xi},
\qquad
k_\ell=\frac{2\pi\ell}{L_g},
$$

so that

$$
\partial_\xi\mapsto-ik,\qquad
\partial_\xi^3\mapsto ik^3,\qquad
\Lambda\mapsto|k|.
$$

NumPy's FFT ordering is reconciled with this convention by using the signed code wavenumber

$$
k_m=-\frac{2\pi}{L_g}\nu_m.
$$

### Exact mean-heterogeneity split

The diagonal mean operator is

$$
L_0(k)=-i\bar b k^3-\bar g|k|.
$$

The residual is

$$
\mathcal F(a)
=
-\frac12\partial_\xi(a^2)
-\widetilde b\,a_{\xi\xi\xi}
-\widetilde g\,\Lambda a.
$$

This split is exact. No dense heterogeneous matrix exponential is used in production calculations.

The operator ordering is fixed:

- $b(\xi)a_{\xi\xi\xi}$, not $\partial_\xi^3[ba]$;
- $g(\xi)\Lambda a$, not $\Lambda[ga]$.

### Two-thirds projection

With $N$ even, the package retains modes satisfying

$$
|\nu_m|\le\lfloor N/3\rfloor
$$

and sets the remaining modes to zero. The state and coefficient spectra are projected before multiplicative evaluation, and transformed products are projected before return to the time integrator.

The zero Fourier mode is retained and is **not reset**. Heterogeneous coefficient fields can generate a nonzero mean through the exact $I_1$ balance.

### ETDRK4

The mean operator is advanced analytically with fourth-order exponential time differencing Runge-Kutta (ETDRK4). The implementation uses cancellation-safe $\varphi$-functions and their analytic zero-mode limits,

$$
\varphi_1(0)=1,\qquad
\varphi_2(0)=\frac12,\qquad
\varphi_3(0)=\frac16.
$$

Every ETDRK4 intermediate state and final state is projected.

### Explicit-scale screening

The explicitly treated nonlinear/heterogeneous scale is monitored with

$$
\chi_h
=
h\left[
\|a\|_\infty k_{\mathrm{ret}}
+\|\widetilde b\|_\infty k_{\mathrm{ret}}^3
+\|\widetilde g\|_\infty k_{\mathrm{ret}}
\right].
$$

`chi_h` is a screening diagnostic, not a stability theorem. Timestep acceptance is determined by refinement.

## Diagnostics and scientific outputs

### Exact integral balances

The package records

$$
I_1(s)=\int_0^{L_g}a\,d\xi,
\qquad
I_2(s)=\int_0^{L_g}a^2\,d\xi,
\qquad
E=\frac12 I_2.
$$

For the heterogeneous system,

$$
\frac{dI_1}{ds}
=
\int a\,b'''(\xi)\,d\xi
-
\int a\,\Lambda g(\xi)\,d\xi,
$$

and

$$
\frac{dE}{ds}
=
\frac12\int b'''a^2\,d\xi
-\frac32\int b'(a_\xi)^2\,d\xi
-\int g\,a\,\Lambda a\,d\xi.
$$

Instantaneous and integrated balance residuals are stored and used in numerical acceptance.

### Logarithmic wave-energy growth

$$
G(s)=\frac{d}{ds}\ln I_2(s),
\qquad
G_{\mathrm{bal}}=\frac{2\mathcal B_E}{I_2}.
$$

An independent finite-difference estimate is generated from the saved $I_2$ history. A negative $G$ is **not** imposed as a universal validity condition for heterogeneous cases.

### Spectral broadening

The inherited cutoff is

$$
k_c=1.5(3k_0).
$$

Using normalized Fourier coefficients,

$$
R(s)=\frac{E_{\mathrm{high}}}{E_{\mathrm{low}}},
$$

where the low band includes the zero mode and the high band extends to the retained two-thirds cutoff.

The historical $R>1.5$ criterion is retained only as a parent-reference diagnostic. Heterogeneous disease conclusions should be based primarily on converged heterogeneous-versus-matched-mean comparisons.

### Matched-mean observables

For every paired heterogeneous calculation,

$$
\Delta R(s)=R_{\mathrm{het}}(s)-R_{\mathrm{mm}}(s),
$$

$$
D_2(s)
=
\frac{\|a_{\mathrm{het}}-a_{\mathrm{mm}}\|_{L^2}}
{\max(\|a_{\mathrm{mm}}\|_{L^2},\epsilon_{\mathrm{mach}})}.
$$

### Tail-resolution diagnostic

$$
\eta_{\mathrm{tail}}
=
\frac{
\sum_{0.8k_{\mathrm{ret}}<|k|\le k_{\mathrm{ret}}}
|\widehat a_k|^2
}{
\sum_{|k|\le k_{\mathrm{ret}}}|\widehat a_k|^2
}.
$$

Persistent accumulation near the retained cutoff or material change under refinement indicates inadequate spatial resolution.

### Modal-energy mechanism budget

For selected mechanism cases,

$$
\frac{dE_k}{ds}
=
T_N(k)
+T_{\widetilde b}(k)
+T_{\widetilde g}(k)
+T_{\bar g}(k)
+T_{\bar b}(k).
$$

The decomposition is checked for modal and high-band closure. The mean dispersive contribution $T_{\bar b}$ must vanish to numerical roundoff because it is purely phase rotating.

## Verification and numerical acceptance

A full scientific study should not begin from an arbitrary $N$ and timestep. The package separates:

1. **analytical/numerical verification of the solver**, and
2. **parameter selection for the configured morphology classes**.

### Independent verification suite

`full_verification_suite()` covers the Solver Design requirements, including:

- Fourier-sign verification;
- physical/modal/split three-form operator identity;
- exact constant-coefficient linear propagation;
- fourth-order heterogeneous linear convergence against a dense matrix exponential;
- classical KdV invariant refinement;
- heterogeneous fractional Burgers-type limit;
- pointwise parent-coefficient recovery;
- published constant-coefficient recovery;
- exact heterogeneous balance identities and refinement;
- morphology/coefficient field-resolution refinement;
- aliasing/tail control;
- modal-energy budget closure;
- exact matched-mean construction;
- single-case restart equivalence;
- paired heterogeneous/matched-mean restart equivalence;
- optimized/reference backend equivalence when the optimized backend is active.

The permanent GitHub Actions workflow runs the Python test suite across supported Python versions, executes scientific verification on both reference and optimized backends, tests restart behavior, performs the notebook same-kernel installation preflight, and builds/installs release artifacts.

### Spatial refinement

For each configured morphology class, a demanding admissible coefficient-sensitivity case is refined in space. At common saved times,

$$
\epsilon_{I_2}^{(N)}
=
\frac{\max_s|I_2^{(2N)}-I_2^{(N)}|}
{\max_s I_2^{(2N)}}.
$$

The inherited minimum requirement is

$$
\epsilon_{I_2}^{(N)}<10^{-5}.
$$

Acceptance also requires stability of the full relevant histories and observables, including $R$, matched-mean quantities, $R_{\max}$, its occurrence time, projection errors, tail behavior, and integrated balance residuals.

### Temporal refinement

Temporal refinement is performed **after a converged spatial resolution has been established**. The full histories of $I_2$, $R$, $\Delta R$, $D_2$, and the balance residuals are compared. The heterogeneous linear benchmark must show fourth-order convergence until another error floor dominates.

The published timesteps are recovery references, not automatic full-study choices.

### Default parameter-selection grids

The current default selection sets are

```text
SELECTION_N_VALUES  = (256, 512, 1024)
SELECTION_DT_VALUES = (8e-4, 4e-4, 2e-4)

I2_CONVERGENCE_TOL         = 1e-5
OBSERVABLE_CONVERGENCE_TOL = 1e-3
```

When parameter selection succeeds, the main disease calculations use the accepted $N$ and timestep for their own morphology class. Explicit values supplied for verification or convergence calculations always override those class-specific settings.

## Installation

**Do not install or use the package until prior written permission has been obtained from `khalid.saqr@knowdyn.co.uk`.**

After permission is granted, clone and install in an isolated Python environment:

```bash
git clone https://github.com/khalid-saqr/Arterial-Spectral-Cascade.git
cd Arterial-Spectral-Cascade

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Supported Python versions are defined in `pyproject.toml` and checked by CI.

For development and verification:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Run the lightweight package check:

```bash
python examples/quick_check.py
```

Run the researcher-facing heterogeneous/matched-mean example:

```bash
python examples/researcher_quickstart.py
```

The coefficient sensitivities in the example are **synthetic interface-demonstration values only**. They are not clinical mappings.

## Researcher quick start

The minimal paired workflow is:

```python
from arterial_spectral_cascade import (
    CaseSpec,
    prepare_case,
    run_paired_case,
)

spec = CaseSpec(
    "DL",
    Wo0=10.0,
    N=128,
    dt=2.0e-3,
    T_final=4.0e-2,
    k0=0.5,
    chi_b=0.05,          # example coefficient-space value only
    chi_g=-0.03,         # example coefficient-space value only
    xi_c=2.0 * 3.141592653589793,
    w=3.0,
    p=1,
    R0_over_L0=0.01,
    slow_variation_limit=0.1,
    output_every_steps=2,
    checkpoint_every_steps=10,
)

parent = prepare_case(spec)

if parent.admissibility["status"] != "ADMISSIBLE":
    raise RuntimeError(parent.admissibility)

result = run_paired_case(
    parent,
    paths=None,
    resume=False,
    progress=False,
)

print(result["summary"])
```

Important research rule: do not convert a stenosis percentage, dilation ratio, clinical label, or local radius into `chi_b` or `chi_g` unless that mapping has been established independently of this package.

## Defining disease cases

### Localized `DL`

```python
from arterial_spectral_cascade import CaseSpec

spec = CaseSpec(
    "DL",
    Wo0=10.0,
    chi_b=...,
    chi_g=...,
    xi_c=...,
    w=...,
    p=1,
    R0_over_L0=...,
    slow_variation_limit=...,
)
```

### Multiple-lesion `DM`

```python
from arterial_spectral_cascade import CaseSpec, Lesion

spec = CaseSpec(
    "DM",
    Wo0=10.0,
    chi_b=...,
    chi_g=...,
    lesions=(
        Lesion(amplitude=1.0, xi_c=2.0, w=2.5, p=1),
        Lesion(amplitude=0.7, xi_c=7.0, w=1.8, p=2),
    ),
    R0_over_L0=...,
    slow_variation_limit=...,
)
```

Every lesion amplitude must be finite and positive; widths must be positive; and `p` must be an integer $\ge1$. The combined field is normalized automatically.

### Analytical distributed `DR`

```python
from arterial_spectral_cascade import CaseSpec, DistributedMode

spec = CaseSpec(
    "DR",
    Wo0=10.0,
    chi_b=...,
    chi_g=...,
    distributed_modes=(
        DistributedMode(amplitude=1.0, q=1.0, phase=0.0),
        DistributedMode(amplitude=0.3, q=2.0, phase=0.2),
    ),
    morphology_scale=...,
    R0_over_L0=...,
    slow_variation_limit=...,
)
```

The distributed wavenumbers must be compatible with periodicity on $[0,L_g]$.

## Geometry-derived morphology

A measured or externally generated geometry may be used only after it has been converted into the Mathematical Model input $\Psi_D$.

```python
import numpy as np
from arterial_spectral_cascade import geometry_derived_spec, prepare_case

psi_D = np.asarray(...)  # one normalized value at each solver collocation point

spec = geometry_derived_spec(
    psi_D=psi_D,
    provenance="Describe the geometry source and the conversion to normalized Psi_D",
    Wo=10.0,
    chi_b=...,
    chi_g=...,
    morphology_scale=...,
    N=len(psi_D),
    dt=...,
    T_final=...,
)

prepared = prepare_case(spec)
print(prepared.admissibility)
```

Requirements:

- exactly one morphology value per collocation point;
- $0\leq\Psi_D\leq1$;
- periodicity and sufficient smoothness at the resolved scale;
- declared provenance;
- declared characteristic morphology scale;
- independently justified or explicitly parametric `chi_b` and `chi_g`.

If the supplied field is too sharp for the current $N$, `prepare_case` reports an excessive morphology/coefficient projection error rather than silently smoothing or interpolating the input.

## Running a complete study

`notebooks/Full_Study.ipynb` is a thin Colab-compatible orchestration interface. The scientific implementation lives in the package.

The public run modes are:

- `QUICK_CHECK` — lightweight numerical sanity checks;
- `VERIFICATION` — independent solver verification and parent-reference audit;
- `PARAMETER_SELECTION` — model preflight plus spatial and temporal convergence for the configured disease classes;
- `FULL_STUDY` — verified, converged disease/matched-mean calculations and study outputs;
- `FIGURES` — regenerate available publication figures from stored results.

`FULL_STUDY` is the default mode but intentionally refuses to start when `DISEASE_CASES` is empty.

### Configure a study in Python

```python
from arterial_spectral_cascade.config import default_study_config
from arterial_spectral_cascade.study import configured_root, run_study_mode
from arterial_spectral_cascade.storage import init_project_paths

cfg = default_study_config()

cfg["RUN_MODE"] = "FULL_STUDY"

cfg["DISEASE_CASES"] = (
    {
        "case_id": "localized_case_1",
        "case_class": "DL",
        "xi_c": 2.0 * 3.141592653589793,
        "w": ...,
        "p": 1,
        "chi_b": ...,
        "chi_g": ...,
        "notes": "Scientific provenance or parameter-scan rationale",
    },
)

paths = init_project_paths(configured_root(cfg))
report = run_study_mode(paths, cfg, progress=True)
```

For a disease-only record, `eps_b` and `eps_g` default to zero. Nonzero parent-background modulation requires

```python
"combined_parent_background": True
```

together with explicit `eps_b`, `eps_g`, and `q`.

### Recommended execution sequence

For new research, use the same logical order enforced by the package:

1. define scientifically justified coefficient-space cases;
2. inspect model preflight/admissibility;
3. run `VERIFICATION`;
4. run `PARAMETER_SELECTION`;
5. allow the package to use the accepted class-specific $N$ and timestep;
6. run `FULL_STUDY`;
7. inspect heterogeneous-versus-matched-mean observables and balance/convergence evidence;
8. regenerate figures from persisted scientific results.

A compatible completed verification or parameter-selection stage is reused from persistent storage. Incompatible schema/version evidence is not silently accepted.

## Persistent results and restart

Persistent study storage is organized into:

```text
metadata/
checkpoints/
verification/
results/
tables/
figures/
logs/
```

Result subdirectories separate parent, localized, multiple, distributed, matched-mean, mechanism, and optional calculations.

Each case is identified from a canonical serialization of its scientific and numerical specification. Metadata records include, as applicable:

- morphology representation and provenance;
- Womersley number;
- `eps_b`, `eps_g`, `q`, `chi_b`, `chi_g`;
- grid, timestep, final time, and output cadence;
- Fourier/dealiasing convention;
- software versions and schema identifiers;
- hashes of morphology, coefficients, grid, retained-mode mask, and initial condition.

Restart checkpoints are transactional and integrity checked. A restart is rejected when the case specification or morphology/coefficient/grid/initial-condition hashes do not match.

Numerically invalid trajectories terminate early, receive an explicit failure status, and are excluded from convergence calculations. Non-finite metadata are rejected by strict JSON serialization rather than being persisted as `NaN` or infinity.

## Performance backend

The default CPU backend is `optimized`, but it is activated only after an automatic deterministic comparison with the preserved reference implementation.

The equivalence check compares:

- the heterogeneous residual;
- one ETDRK4 step;
- a short double-precision trajectory.

If the comparison fails, the reference backend is retained.

Inspect the active backend:

```python
import arterial_spectral_cascade as asc

print(asc.PERFORMANCE_BACKEND_STATUS)
```

Force the reference implementation **before importing the package**:

```bash
ASC_BACKEND=reference python examples/researcher_quickstart.py
```

The optimized backend changes transform/caching/checkpoint implementation details only. It does not change the Mathematical Model, projector, floating-point precision, ETDRK4 coefficients, convergence tolerances, or scientific acceptance criteria.

`benchmark_performance_backend(...)` is informational only. Wall-clock timing is never a scientific acceptance criterion.

## Interpreting results

The package is designed to test a specific mechanism:

$$
\text{morphology}
\rightarrow
\Psi_D(\xi)
\rightarrow
\{b(\xi),g(\xi)\}
\rightarrow
\text{off-diagonal spectral coupling}
\rightarrow
\text{modified cascade dynamics}.
$$

Only the first three arrows are imposed by model construction. The final dynamical effect must be established from converged simulations.

For research interpretation:

- do not equate a positive or negative `chi_b`/`chi_g` with a clinical disease type without independent calibration;
- do not interpret the heterogeneity ceiling `0.3` as a clinical severity threshold;
- do not infer a local Womersley number from a geometry-derived field;
- do not use an under-resolved morphology even if the coefficients remain positive;
- do not compare a heterogeneous disease case only with the baseline: use its exact matched-mean control;
- do not treat a historical parent resonance location as a numerical acceptance criterion;
- do not rely on `R_max` alone: inspect full histories, matched-mean differences, tail behavior, and balance/convergence evidence.

A scientifically defensible heterogeneous result should therefore be accompanied by:

1. morphology definition and provenance;
2. explicit `chi_b`, `chi_g` rationale or parameter-scan statement;
3. admissibility report;
4. accepted spatial and temporal refinement evidence;
5. matched-mean comparison;
6. balance and tail diagnostics;
7. software version/backend/schema information.

## Package organization

- `core_base.py` — foundational Mathematical Model implementation and reference numerical operators.
- `core.py` — current Solver Design layer, additional verification, and explicit-scale diagnostic.
- `geometry.py` — public morphology-field interface.
- `coefficients.py` — coefficient-space closure and matched-mean interface.
- `spectral.py` — Fourier grid, projection, differential operators, and residual interface.
- `etdrk4.py` — ETDRK4 interface.
- `diagnostics.py` — integral and spectral-broadening diagnostics.
- `budgets.py` — modal/high-wavenumber energy-rate decomposition.
- `admissibility.py` — model-admissibility interface.
- `storage.py` / `storage_base.py` — metadata, transactional checkpoints, restart, failure handling, and result archives.
- `paired_runtime.py` — synchronized heterogeneous/matched-mean integration.
- `parent.py` / `parent_base.py` — parent recovery and reference audit.
- `convergence.py` / `planning.py` — spatial/temporal refinement and resonance planning.
- `solver_verification.py` / `verification.py` — independent numerical verification.
- `performance.py` — verified optimized CPU backend and reference equivalence.
- `study.py` / `study_base.py` — study configuration, parameter selection, orchestration, and result tables.
- `plotting.py` — Physics of Fluids/AIP-oriented figure generation.
- `notebooks/Full_Study.ipynb` — thin Colab-compatible orchestration interface.
- `examples/quick_check.py` — minimal installation/numerical check.
- `examples/researcher_quickstart.py` — minimal heterogeneous/matched-mean research example.
- `tests/` — model, solver, numerical-screening, restart, notebook, performance, plotting, terminology, and release tests.

## Reproducibility, citation, and license

### Citation

If written permission to use the software has been granted and the software contributes to research, cite the software release and the associated article:

K. M. Saqr, “Resonant spectral cascade in Womersley flow triggered by arterial geometry,” _Physics of Fluids_ **38**, 041901 (2026), DOI `10.1063/5.0319995`.

Repository citation metadata are provided in `CITATION.cff`.

### License and permission

This repository is **not open-source software** and no general-use license is granted.

Before running, executing, reproducing, modifying, distributing, incorporating, or otherwise using any part of the software, obtain prior written permission from:

**Khalid M. Saqr**  
**khalid.saqr@knowdyn.co.uk**

The complete terms are in [LICENSE](LICENSE). Any permission granted by the copyright holder may contain additional terms and must be retained as part of the research record.
