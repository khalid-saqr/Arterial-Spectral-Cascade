from pathlib import Path
import re
import arterial_spectral_cascade


RETIRED=("campaign","production","pilot","smoke","gate","gating","manifest")
FORBIDDEN_MODEL_TOKENS=("Wo_R","radius_field","Stage 1","Stage-1","Stage 2","Stage-2")


def test_retired_software_terms_absent_from_package_source():
    package_dir=Path(arterial_spectral_cascade.__file__).parent
    for path in package_dir.glob("*.py"):
        text=path.read_text(encoding="utf-8")
        for term in RETIRED:
            assert re.search(rf"\b{re.escape(term)}\b",text,flags=re.IGNORECASE) is None,(path.name,term)
        for token in FORBIDDEN_MODEL_TOKENS:
            assert token.lower() not in text.lower(),(path.name,token)
