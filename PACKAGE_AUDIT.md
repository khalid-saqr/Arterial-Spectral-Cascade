# Package audit

The package is checked against the current Mathematical Model and Solver Design.

- Python source compilation: required.
- Notebook JSON/Markdown validation: required.
- Mathematical Model correspondence: required, including $\Psi_D$ normalization, coefficient closure, pointwise parent recovery, and exact matched-mean construction.
- No local-radius/local-Womersley disease closure: required.
- Two-thirds de-aliasing and cancellation-safe ETDRK4: required.
- $\epsilon_{\Psi}$ and $\epsilon_{\mathrm{coeff}}$ field-resolution checks: required.
- $\chi_h$ and $\eta_{\mathrm{tail}}$ recording: required.
- Full-history spatial/temporal convergence of $I_2$, $R$, $\Delta R$, and $D_2$, plus $R_{\max}$, its occurrence time, projection errors, and integrated balances: required.
- Complete independent verification suite: required, including FFT sign, three-form equality, exact constant-linear propagation, fourth-order heterogeneous-linear benchmark, KdV refinement, fractional Burgers-type refinement, published-model recovery, parent pointwise coefficients, field-resolution refinement, heterogeneous balance refinement, aliasing control, modal-budget closure, matched-mean control, and single/paired restart equivalence.
- Numerical-instability screening, explicit failed-case markers, and strict JSON serialization: required.
- Physics of Fluids/AIP plotting-template export: required.

The expensive full-resolution computational study is not part of the unit-test suite. `FULL_STUDY` first requires compatible Solver Design verification and morphology-class-specific parameter selection.
