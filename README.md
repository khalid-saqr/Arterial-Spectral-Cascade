# Arterial spectral-cascade computational study

This package contains the numerical implementation and the thin Google Colab/Jupyter orchestration notebook for the disease-resolved extension of the heterogeneous fractional KdV framework.

The authoritative reduced-order model is

$$
a_s+a a_\xi+b(\xi)a_{\xi\xi\xi}+g(\xi)\Lambda a=0,
\qquad
\Lambda=(-\partial_\xi^2)^{1/2}.
$$

Disease geometry is introduced through the Stage-1 radius field

$$
R(x)=R_0r(\xi),
\qquad
\mathrm{Wo}_R(\xi)=\mathrm{Wo}_0r(\xi),
$$

with

$$
b(\xi)=b_0(\mathrm{Wo}_0)r(\xi)^{-2}B_G(\xi),
$$

and

$$
g(\xi)=g_{\mathrm{ref}}
\left(1+\frac{C_g}{\mathrm{Wo}_0r(\xi)}\right)G_G(\xi).
$$

For the disease-only comparisons, $\varepsilon_b=\varepsilon_g=0$.

## Package organization

- `core.py` — Stage-1 model objects, Fourier operators, ETDRK4, balances, resonance diagnostics, and independent numerical checks.
- `geometry.py` — radius-field construction.
- `coefficients.py` — radius-to-coefficient closure and matched-mean construction.
- `spectral.py` — Fourier grid, projection, differential operators, and pseudospectral residual.
- `etdrk4.py` — stable fourth-order exponential time differencing.
- `diagnostics.py` — integral and spectral-broadening diagnostics.
- `budgets.py` — modal and high-wavenumber energy-rate decomposition.
- `admissibility.py` — Stage-1 model-range checks.
- `storage.py` / `persistence.py` — case metadata, transactional checkpoints, restart, and result archives.
- `parent.py` — parent-reference audit, heterogeneous/matched-mean paired integration, and complete verification suite.
- `planning.py` / `resonance.py` — convergence and resonance-topology refinement.
- `study.py` — evidence-referenced disease representations and full-study orchestration.
- `plotting.py` — Physics of Fluids/AIP publication plotting template and paper-figure generation.
- `notebooks/Full_Study.ipynb` — thin Colab-compatible orchestration and plotting interface.

## Evidence-referenced disease representations

The full study uses six fixed amplitude representations. They are study identifiers rather than universal clinical-risk categories.

| ID | Reduced geometry | Evidence-referenced amplitude |
|---|---|---|
| S10 | smooth distributed narrowing | 10% diameter reduction |
| S20 | smooth distributed narrowing | 20% diameter reduction |
| S30 | smooth distributed narrowing | 30% diameter reduction |
| D20 | smooth distributed dilation | $D_{\max}/D_0=1.20$ |
| D50 | smooth distributed dilation | $D_{\max}/D_0=1.50$ |
| D60 | smooth distributed dilation | $D_{\max}/D_0=1.60$ |

The source registry is encoded in `study.py` and written to the study tables. The diameter-reduction levels are anchored to Chen et al. (2024); the $1.5$ dilation ratio is used as an abdominal-aortic-aneurysm definition context from Ullery et al. (2018); the idealized dilation-ratio benchmarks include Abdelhamid and Rahma (2025). These publications supply severity-amplitude context only. The Stage-1 reduced radius field remains the governing representation used by the solver.

## Run modes

The public interface uses five modes:

- `QUICK_CHECK`
- `VERIFICATION`
- `PARAMETER_SELECTION`
- `FULL_STUDY`
- `FIGURES`

`FULL_STUDY` is the default.

## Parent-reference audit

The numerically verified Stage-2 parent baseline is the hard acceptance criterion. The legacy published resonance topology is retained as a non-coercive diagnostic. A mismatch in the legacy peak location is recorded but does not invalidate a Stage-2 trajectory that passes the independent numerical verification requirements.

## Google Colab

1. Upload and extract this archive so that the directory `arterial_spectral_cascade_package` is available in `/content` or the current working directory.
2. Open `notebooks/Full_Study.ipynb` in Colab.
3. Use **Run all**. The notebook installs the local package, mounts Google Drive, initializes persistent study storage, runs the complete verification and parameter-selection sequence, performs R1–R5 calculations, and writes publication figures.

By default, persistent outputs are written to:

`/content/drive/MyDrive/PoF_ArterialSpectralCascade`

The location can be overridden in the notebook configuration cell.

## Local installation and tests

```bash
python -m pip install -e .[test]
pytest -q
```

## Publication graphics

`plotting.py` uses the AIP/Physics of Fluids figure template included in this package. It enforces final-size figures, 8-pt minimum text, a 0.5-pt minimum line width, one-column and two-column width limits, 600-dpi raster output, vector PDF/SVG output, panel labels for multipart figures, and alt-text sidecars. Curves are distinguished by line style and/or marker as well as by color.

See `PLOTTING_STANDARD.md` for the locked graphics specification.

## Scope

The package solves the reduced heterogeneous fractional-KdV system defined by Stages 1 and 2. It does not add wall mechanics, plaque mechanics, separated-flow closure, branching, thrombosis, or direct clinical-outcome prediction.
