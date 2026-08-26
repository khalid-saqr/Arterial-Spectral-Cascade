from pathlib import Path
import re

import arterial_spectral_cascade as asc


ROOT = Path(__file__).parents[1]


def test_release_version_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE).group(1)
    citation_version = re.search(r'^version:\s*([^\s]+)', citation, flags=re.MULTILINE).group(1)
    assert asc.__version__ == project_version == citation_version == "0.4.1"


def test_research_citation_metadata_tracks_published_model():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    doi = "10.1063/5.0319995"
    title = "Resonant spectral cascade in Womersley flow triggered by arterial geometry"
    assert doi in citation
    assert title in citation
    assert doi in readme


def test_permission_required_license_is_shipped_and_documented():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    email = "khalid.saqr@knowdyn.co.uk"
    assert email in license_text
    assert email in readme
    assert "prior written permission" in license_text.lower()
    assert "prior written permission" in readme.lower()
    assert "prior written permission" in citation.lower()


def test_internal_build_markdown_is_not_shipped_at_repository_root():
    for name in ("PACKAGE_AUDIT.md", "PLOTTING_STANDARD.md", "TERMINOLOGY_AND_NAMING.md"):
        assert not (ROOT / name).exists()
