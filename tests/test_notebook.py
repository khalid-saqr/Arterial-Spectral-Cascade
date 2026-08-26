from pathlib import Path
import json
import re


def _notebook():
    notebook=Path(__file__).parents[1]/"notebooks"/"Full_Study.ipynb"
    return json.loads(notebook.read_text(encoding="utf-8"))


def test_notebook_math_and_terminology():
    nb=_notebook()
    markdown="\n".join("".join(c.get("source",[])) for c in nb["cells"] if c["cell_type"]=="markdown")
    assert "\\(" not in markdown
    assert "\\[" not in markdown
    for term in ("campaign","production","pilot","smoke","gate","gating","manifest"):
        assert re.search(rf"\b{term}\b",markdown,flags=re.IGNORECASE) is None
    for token in ("Stage 1","Stage-1","Stage 2","Stage-2","Wo_R"):
        assert token.lower() not in markdown.lower()
    assert "$" in markdown


def test_notebook_install_cell_exposes_src_to_current_kernel():
    nb=_notebook()
    install=next(c for c in nb["cells"] if c.get("id")=="8719de88")
    source="".join(install["source"])
    assert 'PACKAGE_ROOT / "src"' in source
    assert "sys.path.insert(0, PACKAGE_SRC)" in source
    assert "importlib.invalidate_caches()" in source
    assert "import arterial_spectral_cascade as _asc_import_check" in source
