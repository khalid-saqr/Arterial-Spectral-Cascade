# Arterial spectral-cascade computational study

This repository implements the spatially heterogeneous extension of the fractional-KdV model introduced in the 2026 *Physics of Fluids* article **“Resonant spectral cascade in Womersley flow triggered by arterial geometry.”**

The authoritative sources for the present package are:

1. **Supplementary File 1 — Mathematical Model**: defines the coefficient-space disease extension.
2. **Supplementary File 2 — Solver Design**: defines the Fourier pseudospectral/ETDRK4 numerical method.

The package does not introduce a local radius-dependent Womersley law.

## Mathematical Model

The solved equation is

$$
a_s+a a_\xi+b(\xi)a_{\xi\xi\xi}+g(\xi)\Lambda a=0,
\qquad
\Lambda=(-\partial_\xi^2)^{1/2}.
$$

The Womersley-dependent baselines are inherited from the published work,

$$
b_0(\mathrm{Wo})=b_{\mathrm{ref}}\mathrm{Wo}^{-2},
$$

$$
g_0(\mathrm{Wo})=g_{\mathrm{ref}}
\left(1+\frac{C_g}{\mathrm{Wo}}\right),
\qquad g_{\mathrm{ref}}=0.005,
\qquad C_g=0.1.
$$

Disease is admitted only through a bounded periodic morphology field

$$
0\leq\Psi_D(\xi)\leq1
$$

and signed effective coefficient sensitivities $\chi_b,\chi_g$:

$$
b(\xi)=b_0(\mathrm{Wo})
\left[1+\varepsilon_b\cos(q\xi)+\chi_b\Psi_D(\xi)\right],
$$

$$
g(\xi)=g_0(\mathrm{Wo})
\left[1+\varepsilon_g\cos(q\xi)+\chi_g\Psi_D(\xi)\right].
$$

For primary disease-only calculations, $\varepsilon_b=\varepsilon_g=0$. The package never infers $\chi_b$ or $\chi_g$ from a clinical label, stenosis percentage, dilation ratio, local radius, or local Womersley number. Those sensitivities must be externally justified or explicitly treated as coefficient-space parameters.

The canonical morphology classes are:

- `DL`: single localized $\Psi_L$;
- `DM`: normalized multiple-lesion $\Psi_M$;
- `DR`: distributed irregular $\Psi_R$ or a supplied geometry-derived sampled $\Psi_D$;
- `MM`: exact matched-mean control generated from a heterogeneous case.

The parent recovery classes remain `H0`, `P0`, and `P1`.

## Solver Design

The exact coefficient decomposition is

$$
b=\bar b+\widetilde b,
\qquad
g=\bar g+\widetilde g,
$$

with diagonal mean operator

$$
L_0(k)=-i\bar b k^3-\bar g|k|,
$$

and residual

$$
\mathcal F(a)
=-\frac12\partial_\xi(a^2)
-\widetilde b\,a_{\xi\xi\xi}
-\widetilde g\,\Lambda a.
$$

The numerical implementation uses the signed Fourier convention of the Solver Design, a symmetric two-thirds projector, ETDRK4 with cancellation-safe $\varphi$ functions, and the dynamically generated zero mode. Every saved state records the exact heterogeneous balance quantities, $\eta_{\mathrm{tail}}$, and the explicit-scale screening quantity $\chi_h$. Disease calculations are advanced with their exact matched-mean controls. Spatial and temporal acceptance uses the complete common-time histories of $I_2$, $R$, $\Delta R$, and $D_2$, together with $R_{\max}$, its occurrence time, morphology/coefficient projection errors, tail behavior, and integrated balance residuals. Temporal refinement is performed only after a converged spatial resolution has been established.

The independent verification suite includes Fourier-sign and three-form identities, exact constant-coefficient propagation, fourth-order heterogeneous linear convergence, KdV refinement, a heterogeneous fractional Burgers-type limit, pointwise parent-coefficient recovery, published constant-coefficient recovery, field-resolution refinement, heterogeneous balance refinement, aliasing control, modal-energy closure, exact matched-mean construction, and single/paired restart equivalence. Numerically invalid trajectories terminate early and are excluded from convergence; strict JSON persistence rejects non-finite metadata.

## Geometry-derived morphology input

A real anatomical or disease-model geometry can be admitted only after it has been converted into the Mathematical Model input $\Psi_D$. The helper `geometry_derived_spec(...)` accepts a sampled normalized morphology field and requires provenance and a declared characteristic morphology scale. It performs no hidden interpolation and does not construct a radius-dependent Womersley field.

## No default clinical disease library

The repository intentionally does **not** ship predefined clinical severity-to-coefficient mappings. `STUDY_CONFIG["DISEASE_CASES"]` is empty by default. A `FULL_STUDY` therefore requires the user to supply explicit coefficient-space cases containing morphology parameters and $\chi_b,\chi_g$.

This prevents the software from silently asserting a constitutive mapping that is not contained in the published PoF derivation or the Mathematical Model.

## Package organization

- `core.py` — Mathematical Model objects, morphology construction, coefficient closure, Fourier operators, ETDRK4, balances, resonance diagnostics, and numerical checks.
- `geometry.py` — public morphology-field interface.
- `coefficients.py` — public coefficient-space closure and matched-mean interface.
- `spectral.py` — Fourier grid, projection, differential operators, and pseudospectral residual.
- `etdrk4.py` — ETDRK4 interface.
- `diagnostics.py` — integral and spectral-broadening diagnostics.
- `budgets.py` — modal/high-wavenumber energy-rate decomposition.
- `admissibility.py` — Mathematical Model admissibility interface.
- `storage.py` / `persistence.py` — metadata, transactional checkpoints, restart, and result archives.
- `parent.py` — parent-reference audit and heterogeneous/matched-mean paired integration.
- `planning.py` / `resonance.py` — convergence and resonance-topology planning.
- `study.py` — model-neutral coefficient-space study configuration and orchestration.
- `plotting.py` — Physics of Fluids/AIP publication plotting and figure regeneration.
- `notebooks/Full_Study.ipynb` — thin Colab-compatible orchestration interface.

## Run modes

The public interface uses:

- `QUICK_CHECK`
- `VERIFICATION`
- `PARAMETER_SELECTION`
- `FULL_STUDY`
- `FIGURES`

`FULL_STUDY` remains the default mode, but it will refuse to start without explicit configured disease cases.

## Local installation and tests

```bash
python -m pip install -e .[test]
pytest -q
```

## Scope

The package solves only the reduced heterogeneous fractional-KdV system defined by the Mathematical Model and Solver Design. It does not add wall compliance, plaque mechanics, separated-flow closure, branching, thrombosis, patient-specific boundary conditions, or direct clinical-outcome prediction.
