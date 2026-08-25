"""
PoF/AIP publication plotting template for the disease-resolved spectral-cascade study.

This module implements figure sizing and typography consistent with the current
AIP Publishing graphics requirements used by Physics of Fluids:

- maximum one-column width: 3.37 in (8.5 cm)
- maximum two-column width: 6.69 in (17 cm)
- maximum depth: 8.25 in (21.1 cm)
- minimum label/legend/tick type size: 8 pt
- minimum reproduced line width: 0.5 pt
- figure parts labelled (a), (b), ...
- fonts embedded in vector output
- figures prepared at final publication size
- 600 dpi raster export for line/combination art
- alt-text sidecar supported for accessibility

The template deliberately does not encode scientific conclusions, colors, or
case-specific plotting choices. Distinct curves should remain identifiable by
line style and/or marker as well as by color.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt


# AIP/Physics of Fluids publication dimensions
SINGLE_COLUMN_IN = 3.37
DOUBLE_COLUMN_IN = 6.69
MAX_DEPTH_IN = 8.25

MIN_FONT_PT = 8.0
DEFAULT_FONT_PT = 8.5
AXIS_LABEL_PT = 9.0
PANEL_LABEL_PT = 9.0
MIN_LINE_PT = 0.5
DEFAULT_LINE_PT = 1.0
DEFAULT_RASTER_DPI = 600


@dataclass(frozen=True)
class FigureSpec:
    width_in: float
    height_in: float
    raster_dpi: int = DEFAULT_RASTER_DPI


def apply_pof_style() -> None:
    """Apply a conservative AIP/PoF-compatible Matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",

            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": DEFAULT_FONT_PT,

            "axes.labelsize": AXIS_LABEL_PT,
            "axes.titlesize": AXIS_LABEL_PT,
            "axes.linewidth": 0.75,
            "axes.grid": False,

            "xtick.labelsize": MIN_FONT_PT,
            "ytick.labelsize": MIN_FONT_PT,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,

            "legend.fontsize": MIN_FONT_PT,
            "legend.frameon": False,

            "lines.linewidth": DEFAULT_LINE_PT,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.75,

            # Embed TrueType fonts in PDF/PS and retain text in SVG.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",

            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",

            "figure.constrained_layout.use": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def figure_spec(
    width: str = "single",
    *,
    height_in: float | None = None,
    aspect: float = 0.72,
) -> FigureSpec:
    """
    Return a final-size figure specification.

    Parameters
    ----------
    width:
        "single" for 3.37 in or "double" for 6.69 in.
    height_in:
        Explicit figure height. If omitted, height = width * aspect.
    aspect:
        Height/width ratio when height_in is omitted.
    """
    if width == "single":
        w = SINGLE_COLUMN_IN
    elif width == "double":
        w = DOUBLE_COLUMN_IN
    else:
        raise ValueError("width must be 'single' or 'double'")

    h = float(height_in) if height_in is not None else float(w * aspect)
    if h > MAX_DEPTH_IN:
        raise ValueError(
            f"Requested height {h:.3f} in exceeds AIP maximum depth "
            f"{MAX_DEPTH_IN:.2f} in."
        )
    return FigureSpec(w, h)


def new_figure(
    width: str = "single",
    *,
    height_in: float | None = None,
    aspect: float = 0.72,
):
    """Create one final-size PoF figure and axis."""
    apply_pof_style()
    spec = figure_spec(width, height_in=height_in, aspect=aspect)
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))
    return fig, ax


def new_panels(
    nrows: int,
    ncols: int,
    *,
    width: str = "double",
    height_in: float | None = None,
    panel_aspect: float = 0.72,
    sharex: bool = False,
    sharey: bool = False,
):
    """
    Create a multi-panel figure at final publication size.

    The default height is estimated from the width allocated to each panel.
    """
    apply_pof_style()
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive integers")

    base_width = SINGLE_COLUMN_IN if width == "single" else DOUBLE_COLUMN_IN
    if height_in is None:
        panel_width = base_width / ncols
        height_in = min(MAX_DEPTH_IN, panel_aspect * panel_width * nrows)

    spec = figure_spec(width, height_in=height_in)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(spec.width_in, spec.height_in),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return fig, axes


def format_axis(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim=None,
    ylim=None,
    minor_ticks: bool = True,
) -> None:
    """Apply the common axis treatment used in the paper figures."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(which="both", direction="in", top=True, right=True)
    if minor_ticks:
        ax.minorticks_on()


def panel_label(ax, label: str, *, x: float = 0.02, y: float = 0.98) -> None:
    """Place a panel identifier such as '(a)' in a consistent location."""
    if not (label.startswith("(") and label.endswith(")")):
        label = f"({label})"
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_PT,
        fontweight="normal",
    )


def series_style(index: int) -> dict:
    """
    Return a color-independent line-style/marker combination.

    Curves should not rely on color alone for identification.
    """
    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    return {
        "linestyle": line_styles[index % len(line_styles)],
        "marker": markers[index % len(markers)],
        "linewidth": DEFAULT_LINE_PT,
        "markevery": 12,
    }


def validate_figure(fig) -> list[str]:
    """
    Audit the figure against the mechanical AIP requirements implemented here.

    Returns a list of warnings. Size violations raise ValueError because they
    directly contradict the publication dimensions.
    """
    warnings_list: list[str] = []
    width, height = fig.get_size_inches()

    if width > DOUBLE_COLUMN_IN + 1e-9:
        raise ValueError(
            f"Figure width {width:.3f} in exceeds AIP two-column maximum "
            f"{DOUBLE_COLUMN_IN:.2f} in."
        )
    if height > MAX_DEPTH_IN + 1e-9:
        raise ValueError(
            f"Figure height {height:.3f} in exceeds AIP maximum depth "
            f"{MAX_DEPTH_IN:.2f} in."
        )

    for text in fig.findobj(match=mpl.text.Text):
        if not text.get_text().strip():
            continue
        size = float(text.get_fontsize())
        if size < MIN_FONT_PT - 1e-9:
            warnings_list.append(
                f"Text '{text.get_text()[:40]}' is {size:.2f} pt; "
                f"AIP minimum is {MIN_FONT_PT:.1f} pt."
            )

    for line in fig.findobj(match=mpl.lines.Line2D):
        lw = float(line.get_linewidth())
        if lw > 0 and lw < MIN_LINE_PT - 1e-9:
            warnings_list.append(
                f"A plotted line has width {lw:.2f} pt; "
                f"AIP minimum reproduced width is {MIN_LINE_PT:.1f} pt."
            )

    return warnings_list


def save_pof_figure(
    fig,
    output_stem: str | Path,
    *,
    alt_text: str | None = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    raster_dpi: int = DEFAULT_RASTER_DPI,
) -> list[Path]:
    """
    Validate and export one complete figure.

    PDF/SVG retain vector line art. PNG is written at 600 dpi by default.
    A plain-text alt-text sidecar is written when alt_text is supplied.
    """
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    issues = validate_figure(fig)
    for issue in issues:
        warnings.warn(issue, stacklevel=2)

    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = output_stem.with_suffix(f".{fmt}")
        kwargs = {}
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = raster_dpi

        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={"Creator": "arterial_spectral_cascade plotting module"},
            **kwargs,
        )
        written.append(path)

    if alt_text is not None:
        alt_path = output_stem.with_suffix(".alt.txt")
        alt_path.write_text(alt_text.strip() + "\n", encoding="utf-8")
        written.append(alt_path)

    return written


# Recommended scientific symbol labels for this study.
# Use math mode so physical quantities are italicized consistently.
LABELS = {
    "xi": r"$\xi$",
    "s": r"$s$",
    "Wo": r"$\mathrm{Wo}$",
    "psi_D": r"$\Psi_D(\xi)$",
    "b": r"$b(\xi)$",
    "g": r"$g(\xi)$",
    "R": r"$R$",
    "Rmax": r"$R_{\max}$",
    "deltaR": r"$\Delta R$",
    "D2": r"$D_2$",
    "I1": r"$I_1$",
    "I2": r"$I_2$",
    "G": r"$G$",
    "k": r"$k$",
}


if __name__ == "__main__":
    # Minimal self-check. This is a template demonstration, not a study figure.
    import numpy as np

    x = np.linspace(0.0, 2.0 * np.pi, 200)
    fig, ax = new_figure("single")
    ax.plot(x, np.sin(x), **series_style(0), label="Series 1")
    ax.plot(x, np.cos(x), **series_style(1), label="Series 2")
    format_axis(ax, xlabel=LABELS["xi"], ylabel=r"$a$")
    ax.legend()
    panel_label(ax, "a")

    issues = validate_figure(fig)
    if issues:
        for item in issues:
            print("WARNING:", item)
    else:
        print("PoF plotting template self-check: PASS")


# ---------------------------------------------------------------------------
# Study-specific publication figures
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Study-specific publication figures
# ---------------------------------------------------------------------------
import json, time
import numpy as np
import pandas as pd
from .core import make_grid, prepare_case, RESULT_SCHEMA
from .storage import atomic_write_json


def _result_dir_for_case(paths, case_class, case_id):
    base={"DL":"localized","DM":"multiple","DR":"distributed","MM":"matched_mean",
          "P0":"parent","P1":"parent","H0":"parent"}.get(case_class,"optional")
    return paths.results/base/case_id


def figure_morphology_atlas(cfg=None):
    from .study import STUDY_CONFIG, case_record_to_spec
    if cfg is None: cfg=STUDY_CONFIG
    records=list(cfg.get("DISEASE_CASES",()))
    if not records: raise ValueError("No configured disease cases are available for a morphology atlas.")
    apply_pof_style(); fig,axs2=new_panels(1,2,width="double",height_in=3.4); axs=axs2[0]
    wo=15.0 if 15.0 in cfg.get("COARSE_WO",()) else float(cfg["COARSE_WO"][0])
    for idx,record in enumerate(records):
        prep=prepare_case(case_record_to_spec(record,wo,N=max(128,min(512,int(cfg["STUDY_N"]))),T_final=1.0,cfg=cfg))
        label=str(record["case_id"]); sty=series_style(idx)
        axs[0].plot(prep.grid.xi,prep.psi_D,label=label,**sty)
        axs[1].plot(prep.grid.xi,prep.b/prep.b_bar,label=label,**sty)
    axs[0].set(xlabel=r"$\xi$",ylabel=r"$\Psi_D(\xi)$"); axs[0].legend(frameon=False)
    axs[1].set(xlabel=r"$\xi$",ylabel=r"$b/\bar b$"); axs[1].legend(frameon=False)
    panel_label(axs[0],"a"); panel_label(axs[1],"b"); return fig


def figure_parent_reference_audit(paths):
    f=paths.verification/"PARENT_REFERENCE_AUDIT.json"
    if not f.exists(): raise FileNotFoundError(f)
    d=json.loads(f.read_text()); rows=d["solver_design_parent_baseline"]["rows"]; legacy=d["legacy_reference_audit"]
    apply_pof_style(); fig,ax=new_figure("single",height_in=3.0)
    ax.plot([r["Wo"] for r in rows],[r["R_max"] for r in rows],label="Solver Design parent baseline",**series_style(0))
    ax.axvline(float(legacy["reported_peak_Wo"]),linestyle="--",linewidth=1.0,label="Legacy reported peak")
    ax.set(xlabel=r"$\mathrm{Wo}$",ylabel=r"$R_{\max}$"); ax.legend(frameon=False); return fig


def coupling_matrix_from_archive(archive_path, mode_limit=10):
    with np.load(archive_path,allow_pickle=False) as z: xi=z["xi"]; b=z["b"]; g=z["g"]
    N=len(xi); Lg=(xi[1]-xi[0])*N; grid=make_grid(N,Lg); bt=b-np.mean(b); gt=g-np.mean(g)
    bh=np.fft.fft(bt)/N; gh=np.fft.fft(gt)/N; inds=np.flatnonzero(grid.mask); modes=-grid.nu[inds]; sel=inds[np.abs(modes)<=mode_limit]; sm=-grid.nu[sel]
    H=np.zeros((len(sel),len(sel)),complex)
    for il,l in enumerate(sel):
        for jn,n in enumerate(sel):
            m=(l-n)%N; H[il,jn]=-1j*bh[m]*grid.k[n]**3-gh[m]*abs(grid.k[n])
    order=np.argsort(sm); return sm[order],np.abs(H[np.ix_(order,order)])


def figure_R1_from_archive(archive_path):
    apply_pof_style()
    with np.load(archive_path,allow_pickle=False) as z: xi=z["xi"]; psi=z["psi_D"]; b=z["b"]; g=z["g"]
    N=len(xi); Lg=(xi[1]-xi[0])*N; grid=make_grid(N,Lg); bb=np.mean(b); gg=np.mean(g); bh=np.fft.fft(b-bb)/N; gh=np.fft.fft(g-gg)/N; modes,H=coupling_matrix_from_archive(archive_path)
    fig,axs=new_panels(2,2,width="double",height_in=5.8)
    axs[0,0].plot(xi,psi,linestyle="-",linewidth=DEFAULT_LINE_PT); axs[0,0].set(xlabel=r"$\xi$",ylabel=r"$\Psi_D(\xi)$")
    axs[0,1].plot(xi,b/bb,label=r"$b/\bar b$",**series_style(0)); axs[0,1].plot(xi,g/gg,label=r"$g/\bar g$",**series_style(1)); axs[0,1].legend(frameon=False); axs[0,1].set(xlabel=r"$\xi$")
    kk=np.fft.fftshift(grid.k); axs[1,0].semilogy(kk,np.fft.fftshift(np.abs(bh))+1e-30,label=r"$|\widehat{\tilde b}|$",linestyle="-"); axs[1,0].semilogy(kk,np.fft.fftshift(np.abs(gh))+1e-30,label=r"$|\widehat{\tilde g}|$",linestyle="--"); axs[1,0].legend(frameon=False); axs[1,0].set(xlabel=r"$k$")
    im=axs[1,1].imshow(H,origin="lower",aspect="auto",extent=[modes.min(),modes.max(),modes.min(),modes.max()]); axs[1,1].set(xlabel=r"$k_n$",ylabel=r"$k_\ell$"); fig.colorbar(im,ax=axs[1,1],shrink=.8)
    for ax,label in zip(axs.ravel(),["a","b","c","d"]): panel_label(ax,label)
    return fig


def figure_R2_resonance_landscape(df,case_class):
    apply_pof_style(); sub=df[df.case_class==case_class].copy(); fig,ax=new_figure("double",height_in=4.0)
    for idx,(cid,gp) in enumerate(sub.groupby("study_case_id")):
        gp=gp.sort_values("Wo"); ax.plot(gp.Wo,gp.R_max_het,label=str(cid),**series_style(idx))
    ax.set(xlabel=r"$\mathrm{Wo}$",ylabel=r"$R_{\max}$"); ax.legend(frameon=False); return fig


def figure_R3_from_archive(archive_path):
    with np.load(archive_path,allow_pickle=False) as z: Hs=z["het_s"]; HR=z["het_R"]; Ms=z["mm_s"]; MR=z["mm_R"]; Cs=z["cmp_s"]; dR=z["cmp_DeltaR"]; D2=z["cmp_D2"]
    apply_pof_style(); fig,axs2=new_panels(2,1,width="single",height_in=5.4,sharex=True); axs=axs2[:,0]
    axs[0].plot(Hs,HR,label="heterogeneous",**series_style(0)); axs[0].plot(Ms,MR,label="matched mean",**series_style(1)); axs[0].set_ylabel(r"$R$"); axs[0].legend(frameon=False)
    axs[1].plot(Cs,dR,label=r"$\Delta R$",linestyle="-"); ax2=axs[1].twinx(); ax2.plot(Cs,D2,linestyle="--",label=r"$D_2$"); axs[1].set(xlabel=r"$s$",ylabel=r"$\Delta R$"); ax2.set_ylabel(r"$D_2$")
    panel_label(axs[0],"a"); panel_label(axs[1],"b"); return fig


def figure_R4_from_archive(archive_path):
    with np.load(archive_path,allow_pickle=False) as z:
        req=["budget_peak_T_N","budget_peak_T_b_tilde","budget_peak_T_g_tilde","budget_peak_T_g_bar","budget_peak_PiH_N","budget_peak_PiH_b_tilde","budget_peak_PiH_g_tilde","budget_peak_PiH_g_bar"]
        if not all(k in z.files for k in req): raise ValueError("Archive contains no mechanism budget.")
        N=len(z["xi"]); Lg=(z["xi"][1]-z["xi"][0])*N; grid=make_grid(N,Lg); modes=-grid.nu; vals={k:z[k] for k in req}
    apply_pof_style(); fig,axs2=new_panels(1,2,width="double",height_in=3.5); axs=axs2[0]
    for idx,(key,label) in enumerate([("budget_peak_T_N","nonlinear"),("budget_peak_T_b_tilde",r"$\tilde b$"),("budget_peak_T_g_tilde",r"$\tilde g$"),("budget_peak_T_g_bar",r"$\bar g$")]): axs[0].plot(modes,vals[key],label=label,linestyle=("-","--","-.",":")[idx])
    axs[0].set(xlabel=r"$k$",ylabel=r"$T_j(k)$"); axs[0].legend(frameon=False,ncol=2)
    hv=[float(vals[k]) for k in ["budget_peak_PiH_N","budget_peak_PiH_b_tilde","budget_peak_PiH_g_tilde","budget_peak_PiH_g_bar"]]; axs[1].bar([r"$N$",r"$\tilde b$",r"$\tilde g$",r"$\bar g$"],hv); axs[1].axhline(0,lw=.8); axs[1].set(ylabel=r"$\Pi_j^H$")
    panel_label(axs[0],"a"); panel_label(axs[1],"b"); return fig


def figure_morphology_scale(df):
    apply_pof_style(); good=df[df.status=="ADMISSIBLE"].sort_values("factor"); fig,ax=new_figure("single",height_in=3.2)
    ax.plot(good.factor,good.Delta_R_maxima,**series_style(0)); ax.set(xlabel="morphology scale factor",ylabel=r"$R_{\max,het}-R_{\max,mm}$"); return fig


def regenerate_available_figures(paths,cfg=None):
    from .study import STUDY_CONFIG
    if cfg is None: cfg=STUDY_CONFIG
    paths.figures.mkdir(parents=True,exist_ok=True); made=[]
    if cfg.get("DISEASE_CASES"):
        fig=figure_morphology_atlas(cfg); made += [str(p) for p in save_pof_figure(fig,paths.figures/"R1_morphology_atlas",alt_text="Configured disease morphology fields and their dispersion-coefficient imprints.")]; plt.close(fig)
    if (paths.verification/"PARENT_REFERENCE_AUDIT.json").exists():
        fig=figure_parent_reference_audit(paths); made += [str(p) for p in save_pof_figure(fig,paths.figures/"V_parent_reference_audit",alt_text="Solver Design parent baseline spectral-broadening response across Womersley number with the legacy reported peak indicated for reference.")]; plt.close(fig)
    primary=paths.tables/"primary_resonance.csv"
    if primary.exists():
        df=pd.read_csv(primary)
        for cls in ["DL","DM","DR"]:
            if (df.case_class==cls).any():
                fig=figure_R2_resonance_landscape(df,cls); made += [str(p) for p in save_pof_figure(fig,paths.figures/f"R2_{cls}_resonance",alt_text=f"Spectral-broadening response across Womersley number for configured {cls} coefficient-space morphology cases.")]; plt.close(fig)
        sf=paths.tables/"mechanism_selection.csv"
        if sf.exists():
            sel=pd.read_csv(sf)
            for _,row in sel.iterrows():
                arc=_result_dir_for_case(paths,row.case_class,row.source_case_id)/"paired_result.npz"
                if arc.exists():
                    cid=str(row.study_case_id)
                    fig=figure_R1_from_archive(arc); made += [str(p) for p in save_pof_figure(fig,paths.figures/f"R1_{cid}_morphology_coupling",alt_text=f"Morphology-to-coefficient-to-modal-coupling representation for {cid}.")]; plt.close(fig)
                    fig=figure_R3_from_archive(arc); made += [str(p) for p in save_pof_figure(fig,paths.figures/f"R3_{cid}_matched_mean",alt_text=f"Heterogeneous and matched-mean spectral-broadening comparison for {cid}.")]; plt.close(fig)
    for branch in ["localized","multiple","distributed"]:
        for arc in (paths.results/branch).glob("*/paired_result.npz"):
            try:
                with np.load(arc,allow_pickle=False) as z: has="budget_peak_T_N" in z.files
                if has:
                    fig=figure_R4_from_archive(arc); made += [str(p) for p in save_pof_figure(fig,paths.figures/f"R4_{branch}_{arc.parent.name}",alt_text=f"Modal and high-wavenumber energy-rate decomposition for a selected {branch} coefficient-space morphology case.")]; plt.close(fig)
            except Exception: continue
    for wf in paths.tables.glob("morphology_scale_*.csv"):
        dfw=pd.read_csv(wf)
        if not dfw.empty:
            fig=figure_morphology_scale(dfw); made += [str(p) for p in save_pof_figure(fig,paths.figures/f"R5_{wf.stem}",alt_text="Morphology-scale dependence of the heterogeneous minus matched-mean peak spectral-broadening response.")]; plt.close(fig)
    atomic_write_json(paths.figures/"FIGURE_INDEX.json",{"files":made,"result_schema":RESULT_SCHEMA,"generated_unix":time.time()}); return made
