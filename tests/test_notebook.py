from pathlib import Path
import json
import re


def test_notebook_math_and_terminology():
    notebook = Path(__file__).parents[1] / "notebooks" / "Full_Study.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    markdown = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "\\(" not in markdown
    assert "\\[" not in markdown
    for term in ("campaign", "production", "pilot", "smoke", "gate", "manifest"):
        assert re.search(rf"\\b{term}\\b", markdown, flags=re.IGNORECASE) is None
    assert "$" in markdown
