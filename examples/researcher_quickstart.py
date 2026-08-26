"""Minimal coefficient-space heterogeneous/matched-mean research example.

The coefficient sensitivities below are synthetic demonstration values only.
They are not clinical mappings and must not be interpreted as disease severity.
"""

from arterial_spectral_cascade import CaseSpec, prepare_case, run_paired_case


spec = CaseSpec(
    "DL",
    Wo0=10.0,
    N=128,
    dt=2.0e-3,
    T_final=4.0e-2,
    k0=0.5,
    chi_b=0.05,
    chi_g=-0.03,
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

result = run_paired_case(parent, paths=None, resume=False, progress=False)
summary = result["summary"]

print("case_id:", parent.case_id)
print("Psi_D range:", float(parent.psi_D.min()), float(parent.psi_D.max()))
print("mean b:", parent.b_bar)
print("mean g:", parent.g_bar)
print("numerical_status:", summary["numerical_status"])
print("Delta_R_maxima:", summary["Delta_R_maxima"])
print("D2_max:", summary["D2_max"])

if not summary["runtime_valid"]:
    raise SystemExit(1)
