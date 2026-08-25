from arterial_spectral_cascade.verification import quick_numerical_check, full_verification_suite


def test_quick_numerical_check():
    report = quick_numerical_check()
    assert report["pass"]


def test_full_verification_suite():
    report = full_verification_suite()
    assert report["pass"]
    assert all(item["pass"] for item in report["tests"])
