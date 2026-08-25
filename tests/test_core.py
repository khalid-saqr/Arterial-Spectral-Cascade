from arterial_spectral_cascade.verification import quick_numerical_check, full_verification_suite


def test_quick_numerical_check():
    report = quick_numerical_check()
    assert report["pass"]


def test_full_verification_suite():
    report = full_verification_suite()
    assert report["pass"]
    assert all(item["pass"] for item in report["tests"])
    names={item["name"] for item in report["tests"]}
    required={"heterogeneous_linear_order","kdv_refinement","fractional_burgers_refinement",
              "published_recovery_convergence","parent_pointwise_coefficients","field_resolution_refinement",
              "heterogeneous_balance_refinement","aliasing_control","modal_budget","matched_mean_control",
              "restart_equivalence","paired_restart_equivalence"}
    assert required.issubset(names)
