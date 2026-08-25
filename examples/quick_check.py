from arterial_spectral_cascade.verification import quick_numerical_check

report = quick_numerical_check()
print(report)
if not report["pass"]:
    raise SystemExit(1)
