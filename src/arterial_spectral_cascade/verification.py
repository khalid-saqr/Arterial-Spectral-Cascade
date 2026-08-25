"""Independent solver verification and parent-reference audit."""
from .core import verify_fft_sign, verify_three_form, verify_constant_linear, verify_heterogeneous_linear_order, verify_balance_identities, verify_modal_budget, verify_kdv_refinement, quick_numerical_check, core_verification_suite
from .storage import verify_restart_equivalence
from .parent import verify_paired_restart_equivalence, full_verification_suite, run_parent_reference_audit
__all__=[name for name in globals() if name.startswith("verify_") or name in {"quick_numerical_check","core_verification_suite","full_verification_suite","run_parent_reference_audit"}]
