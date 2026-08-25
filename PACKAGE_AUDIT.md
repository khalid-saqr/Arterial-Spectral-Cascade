# Package audit

The package is checked against the current Mathematical Model and Solver Design.

- Python source compilation: required.
- Notebook JSON/Markdown validation: required.
- Mathematical Model correspondence tests: required, including $\Psi_D$ normalization, coefficient closure, parent pointwise recovery, and exact matched-mean construction.
- No local-radius/local-Womersley disease closure: required.
- Quick numerical verification: required.
- Complete independent numerical verification suite: required, including FFT sign, three-form operator equality, exact constant-coefficient propagation, fourth-order heterogeneous linear benchmark, exact heterogeneous balances, modal-budget closure, KdV refinement, single-trajectory restart, and paired heterogeneous/matched-mean restart.
- Numerical-instability screening and strict JSON serialization: required.
- Physics of Fluids/AIP plotting-template export test: required.

The complete full-resolution computational study is not part of the unit-test suite. It is executed from `FULL_STUDY` after the user supplies explicit admissible coefficient-space disease cases.
