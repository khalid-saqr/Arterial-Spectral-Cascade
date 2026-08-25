from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
from arterial_spectral_cascade.plotting import (
    SINGLE_COLUMN_IN, DOUBLE_COLUMN_IN, MAX_DEPTH_IN,
    new_figure, new_panels, validate_figure, save_pof_figure, series_style,
)


def test_publication_dimensions_and_export(tmp_path):
    fig, ax = new_figure("single", height_in=2.5)
    x = np.linspace(0.0, 1.0, 20)
    ax.plot(x, x, **series_style(0))
    assert abs(fig.get_size_inches()[0] - SINGLE_COLUMN_IN) < 1e-12
    assert fig.get_size_inches()[1] <= MAX_DEPTH_IN
    assert validate_figure(fig) == []
    files = save_pof_figure(fig, tmp_path / "figure", alt_text="Test publication figure")
    suffixes = {Path(p).suffix for p in files}
    assert {".pdf", ".svg", ".png", ".txt"}.issubset(suffixes)


def test_two_column_width_and_panel_shape():
    fig, axes = new_panels(2, 2, width="double", height_in=5.0)
    assert abs(fig.get_size_inches()[0] - DOUBLE_COLUMN_IN) < 1e-12
    assert axes.shape == (2, 2)
