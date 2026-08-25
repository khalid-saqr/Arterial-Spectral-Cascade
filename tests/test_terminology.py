from pathlib import Path
import re
import arterial_spectral_cascade


RETIRED = ("campaign", "production", "pilot", "smoke", "gate", "manifest")


def test_retired_software_terms_absent_from_package_source():
    package_dir = Path(arterial_spectral_cascade.__file__).parent
    for path in package_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for term in RETIRED:
            assert re.search(rf"\\b{term}\\b", text, flags=re.IGNORECASE) is None, (path.name, term)
