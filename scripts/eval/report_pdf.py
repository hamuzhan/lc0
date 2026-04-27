"""LaTeX-based PDF report generator for eval-reports/ artifacts.

Produces a paper-style PDF (article class, booktabs tables, embedded vector
figures) by:

1. rendering each chart as a separate vector PDF with matplotlib,
2. emitting a `.tex` source assembled from string templates,
3. compiling with `pdflatex -interaction=nonstopmode` (run twice for
   cross-references and the page count in the title block).

This module imports matplotlib at load time, so callers should lazy-import
it (only when --pdf is requested) to keep eval.py's default execution path
stdlib-only.

Public entry points:
    render_run_pdf(report_dir)       -> Path to report.pdf
    render_compare_pdf(baseline, current) -> Path to compare.pdf
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

# ---------------------------------------------------------------------------
# constants & style
# ---------------------------------------------------------------------------

COLOR_BASELINE = "#888888"
COLOR_CURRENT = "#1f77b4"
COLOR_GOOD = "#2ca02c"
COLOR_BAD = "#d62728"
COLOR_NEUTRAL = "#7f7f7f"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 100,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_build_info_md(path: Path) -> Dict[str, str]:
    """Parse the build_info.md key/value table back into a dict."""
    if not path.is_file():
        return {}
    out: Dict[str, str] = {}
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("|") and "key" in s and "value" in s:
            in_table = True
            continue
        if in_table:
            if not s.startswith("|"):
                if not s or s.startswith("#"):
                    in_table = False
                continue
            if "---" in s:
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 2:
                k = cells[0]
                v = cells[1].replace("<br>", " ").replace("`", "").strip()
                out[k] = v
    return out


def _read_scores(report_dir: Path) -> Dict[str, str]:
    rows = _read_csv(report_dir / "scores.csv")
    return rows[0] if rows else {}


def _to_float(s: Any) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s: Any) -> Optional[int]:
    f = _to_float(s)
    return int(f) if f is not None else None


def _movetime_from_logs(report_dir: Path, log_name: str) -> Optional[int]:
    log = report_dir / "logs" / log_name
    if not log.is_file():
        return None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# command:") and "--movetime=" in line:
            for tok in line.split():
                if tok.startswith("--movetime="):
                    return _to_int(tok.split("=", 1)[1])
    return None


# ---------------------------------------------------------------------------
# matplotlib chart figures (each returns a Figure; LaTeX embeds via
# \includegraphics, so no titles/footers here — captions live in the .tex).
# ---------------------------------------------------------------------------

def _save_fig(fig: Figure, path: Path) -> None:
    fig.savefig(str(path), format="pdf")
    plt.close(fig)


def _fig_radar(scores: Dict[str, str]) -> Optional[Figure]:
    metrics = []
    for label, key in [("correctness", "correctness_score"),
                       ("stability",   "stability_score"),
                       ("tactics",     "tactics_pct"),
                       ("unit tests",  "unit_tests_pct")]:
        v = _to_float(scores.get(key))
        if v is not None:
            metrics.append((label, max(0.0, min(1.0, v / 100.0))))
    if len(metrics) < 3:
        return None

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed_v = values + values[:1]
    closed_a = angles + angles[:1]

    fig = plt.figure(figsize=(4.5, 3.8))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(closed_a, closed_v, color=COLOR_CURRENT, linewidth=1.6)
    ax.fill(closed_a, closed_v, color=COLOR_CURRENT, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#777")
    ax.set_ylim(0, 1.0)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    return fig


def _fig_backend_nps(rows: List[Dict[str, str]]) -> Optional[Figure]:
    if not rows:
        return None
    batch = np.array([_to_float(r["batch_size"]) or 0 for r in rows])
    mean_nps = np.array([_to_float(r["mean_nps"]) or 0 for r in rows])
    median_nps = np.array([_to_float(r.get("median_nps")) or 0 for r in rows])
    max_nps = np.array([_to_float(r.get("max_nps")) or 0 for r in rows])
    min_nps = np.array([_to_float(r.get("min_nps")) or 0 for r in rows])

    fig = plt.figure(figsize=(6.5, 3.4))
    ax = fig.add_subplot(111)
    ax.fill_between(batch, min_nps, max_nps, color=COLOR_CURRENT,
                    alpha=0.12, linewidth=0, label=r"min..max")
    ax.plot(batch, mean_nps, color=COLOR_CURRENT, marker="o", markersize=4,
            linewidth=1.6, label="mean")
    if np.any(median_nps != mean_nps):
        ax.plot(batch, median_nps, color=COLOR_CURRENT, linestyle="--",
                linewidth=1, alpha=0.6, label="median")
    peak_idx = int(np.argmax(mean_nps))
    ax.scatter([batch[peak_idx]], [mean_nps[peak_idx]],
               color=COLOR_GOOD, s=70, zorder=5, edgecolor="white",
               linewidth=1.0,
               label=f"peak {int(mean_nps[peak_idx])} @ batch {int(batch[peak_idx])}")
    ax.set_xlabel("batch size")
    ax.set_ylabel("NPS")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_ylim(bottom=0)
    return fig


def _fig_backend_latency(rows: List[Dict[str, str]]) -> Optional[Figure]:
    if not rows:
        return None
    batch = np.array([_to_float(r["batch_size"]) or 0 for r in rows])
    mean_ms = np.array([_to_float(r["mean_ms"]) or 0 for r in rows])
    sdev_ms = np.array([_to_float(r.get("sdev_ms")) or 0 for r in rows])
    first_max_ms = np.array([_to_float(r.get("first_max_ms")) or np.nan
                             for r in rows])

    fig = plt.figure(figsize=(6.5, 3.4))
    ax = fig.add_subplot(111)
    ax.errorbar(batch, mean_ms, yerr=sdev_ms, color=COLOR_CURRENT,
                marker="s", markersize=4, linewidth=1.4, capsize=2,
                label=r"mean $\pm$ stdev")
    if np.any(~np.isnan(first_max_ms)):
        ax.plot(batch, first_max_ms, color=COLOR_BAD, linestyle=":",
                marker="^", markersize=4, linewidth=1, alpha=0.85,
                label="first-batch max (cold)")
    ax.set_xlabel("batch size")
    ax.set_ylabel("latency (ms)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_ylim(bottom=0)
    return fig


def _fig_search_per_position(rows: List[Dict[str, str]]) -> Optional[Figure]:
    if not rows:
        return None
    idx = np.array([_to_int(r["position_idx"]) or 0 for r in rows])
    nps = np.array([_to_float(r["nps"]) or 0 for r in rows])
    bestmoves = [r.get("bestmove", "") for r in rows]

    fig = plt.figure(figsize=(6.5, 3.4))
    ax = fig.add_subplot(111)
    ax.bar(idx, nps, color=COLOR_CURRENT, edgecolor="white",
           linewidth=0.6, width=0.7)
    mean_nps = float(np.mean(nps))
    median_nps = float(np.median(nps))
    ax.axhline(mean_nps, color=COLOR_NEUTRAL, linestyle="--", linewidth=1,
               label=f"mean {mean_nps:.0f}")
    ax.axhline(median_nps, color=COLOR_NEUTRAL, linestyle=":", linewidth=1,
               label=f"median {median_nps:.0f}")
    ax.set_xticks(idx)
    ax.set_xlabel("position")
    ax.set_ylabel("NPS")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ymax = float(np.max(nps)) if len(nps) else 1.0
    ax.set_ylim(0, ymax * 1.18)
    for b, n, bm in zip(idx, nps, bestmoves):
        ax.text(b, n + ymax * 0.02, bm, ha="center", fontsize=7, color="#444")
    return fig


def _fig_tactics_solved(rows: List[Dict[str, str]]) -> Optional[Figure]:
    if not rows:
        return None
    ids = [r.get("id", f"#{i}") for i, r in enumerate(rows)]
    sol_flags = [(r.get("solved") or "0") == "1" for r in rows]
    colors = [COLOR_GOOD if s else COLOR_BAD for s in sol_flags]
    y_pos = np.arange(len(rows))[::-1]

    fig = plt.figure(figsize=(6.5, max(2.8, 0.35 * len(rows))))
    ax = fig.add_subplot(111)
    ax.barh(y_pos, [1.0] * len(rows), color=colors, edgecolor="white",
            linewidth=0.7, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    ax.tick_params(left=False)
    return fig


def _fig_compare_score_deltas(b_scores: Dict[str, str],
                              c_scores: Dict[str, str]) -> Optional[Figure]:
    metrics = [
        ("correctness_score", "correctness_score"),
        ("stability_score",   "stability_score"),
        ("unit_tests_pct",    "unit_tests_pct"),
        ("tactics_pct",       "tactics_pct"),
        ("backend_peak_nps",  "backend_peak_nps"),
        ("search_total_nps",  "search_total_nps"),
    ]
    labels: List[str] = []
    pcts: List[float] = []
    for key, label in metrics:
        bv = _to_float(b_scores.get(key))
        cv = _to_float(c_scores.get(key))
        if bv is None or cv is None or not bv:
            continue
        labels.append(label)
        pcts.append(100.0 * (cv - bv) / bv)
    if not labels:
        return None
    fig = plt.figure(figsize=(6.5, 3.0))
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))[::-1]
    colors = [COLOR_GOOD if v >= 0 else COLOR_BAD for v in pcts]
    ax.barh(y, pcts, color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#444", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$\Delta\,\%$ (current vs baseline)")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
    ax.grid(False, axis="y")
    # Headroom for value labels — extend x-range to ~1.5x the largest abs delta.
    extent = max(map(abs, pcts)) if pcts else 1.0
    extent = max(extent, 0.5)
    ax.set_xlim(-extent * 1.5, extent * 1.5)
    pad = extent * 0.05
    for yi, vi in zip(y, pcts):
        if vi >= 0:
            ax.text(vi + pad, yi, f"{vi:+.2f}%",
                    va="center", ha="left", fontsize=8, color="#222")
        else:
            ax.text(vi - pad, yi, f"{vi:+.2f}%",
                    va="center", ha="right", fontsize=8, color="#222")
    return fig


def _fig_compare_backend_overlay(b_rows, c_rows) -> Optional[Figure]:
    if not (b_rows or c_rows):
        return None
    fig = plt.figure(figsize=(6.5, 3.4))
    ax = fig.add_subplot(111)
    if b_rows:
        bx = [_to_float(r["batch_size"]) or 0 for r in b_rows]
        by = [_to_float(r["mean_nps"]) or 0 for r in b_rows]
        ax.plot(bx, by, color=COLOR_BASELINE, marker="o", markersize=3,
                linewidth=1.2, label="baseline")
    if c_rows:
        cx = [_to_float(r["batch_size"]) or 0 for r in c_rows]
        cy = [_to_float(r["mean_nps"]) or 0 for r in c_rows]
        ax.plot(cx, cy, color=COLOR_CURRENT, marker="o", markersize=4,
                linewidth=1.6, label="current")
    ax.set_xlabel("batch size")
    ax.set_ylabel("mean NPS")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_ylim(bottom=0)
    return fig


def _fig_compare_backend_delta(b_rows, c_rows) -> Optional[Figure]:
    by_b = {int(r["batch_size"]): _to_float(r["mean_nps"]) for r in b_rows
            if _to_int(r.get("batch_size")) is not None}
    by_c = {int(r["batch_size"]): _to_float(r["mean_nps"]) for r in c_rows
            if _to_int(r.get("batch_size")) is not None}
    common = sorted(set(by_b) & set(by_c))
    deltas = [(bs, 100.0 * (by_c[bs] - by_b[bs]) / by_b[bs])
              for bs in common if by_b[bs]]
    if not deltas:
        return None
    fig = plt.figure(figsize=(6.5, 2.6))
    ax = fig.add_subplot(111)
    xs = [d[0] for d in deltas]
    ys = [d[1] for d in deltas]
    colors = [COLOR_GOOD if y >= 0 else COLOR_BAD for y in ys]
    ax.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.6,
           width=max(min(xs) * 0.6, 1) if xs else 1)
    ax.axhline(0, color="#444", linewidth=0.6)
    ax.set_xlabel("batch size")
    ax.set_ylabel(r"$\Delta\,\%$")
    return fig


def _fig_compare_search_overlay(b_rows, c_rows) -> Optional[Figure]:
    by_b = {int(r["position_idx"]): _to_float(r["nps"]) for r in b_rows
            if _to_int(r.get("position_idx")) is not None}
    by_c = {int(r["position_idx"]): _to_float(r["nps"]) for r in c_rows
            if _to_int(r.get("position_idx")) is not None}
    common = sorted(set(by_b) | set(by_c))
    if not common:
        return None
    bvals = [by_b.get(p) or 0 for p in common]
    cvals = [by_c.get(p) or 0 for p in common]
    width = 0.4
    x = np.arange(len(common))
    fig = plt.figure(figsize=(6.5, 3.0))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, bvals, width=width, color=COLOR_BASELINE,
           label="baseline", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, cvals, width=width, color=COLOR_CURRENT,
           label="current", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in common], fontsize=8)
    ax.set_xlabel("position")
    ax.set_ylabel("NPS")
    ax.legend(fontsize=9, frameon=False)
    return fig


# ---------------------------------------------------------------------------
# LaTeX assembly
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
    "<":  r"\textless{}",
    ">":  r"\textgreater{}",
}


def _esc(s: Any) -> str:
    if s is None:
        return ""
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(s))


def _esc_path(s: Any) -> str:
    """Escape a path (or other long string) and insert non-breaking
    hyphenation hints so LaTeX can wrap on / characters."""
    s = _esc(s)
    return s.replace("/", r"/\allowbreak{}")


def _esc_tt(s: Any) -> str:
    return r"\texttt{" + _esc_path(s) + "}"


_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[a4paper,margin=2.3cm,top=2.5cm,headheight=14pt]{geometry}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage{microtype}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{titling}
\usepackage{enumitem}
\usepackage{seqsplit}

\definecolor{good}{HTML}{2CA02C}
\definecolor{bad}{HTML}{D62728}
\definecolor{neutral}{HTML}{7F7F7F}

\hypersetup{
  pdftitle={%TITLE%},
  pdfauthor={lc0 eval harness},
  pdfsubject={lc0 evaluation report}
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small lc0 eval}
\fancyhead[R]{\small\texttt{%SHA%}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}

\setlength{\droptitle}{-2.0em}

\title{\bfseries %TITLE%}
\author{Automated harness \,\textemdash\, \texttt{scripts/eval/eval.py}}
\date{%DATE%}

\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\begin{document}
\maketitle
"""


def _preamble(title: str, sha: str, date: str) -> str:
    return (_PREAMBLE
            .replace("%TITLE%", _esc(title))
            .replace("%SHA%", _esc(sha))
            .replace("%DATE%", _esc(date)))


def _kv_table_tex(rows: List[Tuple[str, str]],
                  key_w: str = "0.32",
                  val_w: str = "0.62") -> str:
    """Render a 2-column key/value table using booktabs + tabularx."""
    if not rows:
        return r"\textit{(no data)}"
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}>{\bfseries}L{" + key_w
        + r"\linewidth}X@{}}",
        r"\toprule",
    ]
    for k, v in rows:
        lines.append(f"{_esc(k)} & {v} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def _scores_table_tex(scores: Dict[str, str]) -> str:
    rows: List[Tuple[str, str]] = []
    cs = _to_float(scores.get("correctness_score"))
    ss = _to_float(scores.get("stability_score"))
    if cs is not None:
        rows.append(("correctness\\_score", f"{cs:.1f} / 100"))
    if ss is not None:
        rows.append(("stability\\_score", f"{ss:.1f} / 100"))
    if scores.get("unit_tests_total"):
        pct = _to_float(scores.get("unit_tests_pct")) or 0.0
        rows.append((
            "unit tests passed",
            f"{scores.get('unit_tests_passed','')}/{scores.get('unit_tests_total','')} "
            f"({pct:.1f}\\%)",
        ))
    if scores.get("tactics_total"):
        pct = _to_float(scores.get("tactics_pct")) or 0.0
        rows.append((
            "tactics solved",
            f"{scores.get('tactics_solved','')}/{scores.get('tactics_total','')} "
            f"({pct:.1f}\\%)",
        ))
    if scores.get("backend_peak_nps"):
        rows.append((
            "backend peak NPS",
            f"{scores['backend_peak_nps']} @ batch {scores.get('backend_peak_batch','?')}",
        ))
    if scores.get("search_total_nps"):
        rows.append(("search total NPS", _esc(scores["search_total_nps"])))
    return _kv_table_tex_pretyped(rows)


def _kv_table_tex_pretyped(rows: List[Tuple[str, str]]) -> str:
    """Like _kv_table_tex but values are already LaTeX-safe."""
    if not rows:
        return r"\textit{(no data)}"
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}>{\bfseries}L{0.32\linewidth}X@{}}",
        r"\toprule",
    ]
    for k, v in rows:
        # k is plain text (already LaTeX-safe in our callers); v is pre-escaped
        lines.append(f"{k} & {v} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def _build_metadata_rows(build_info: Dict[str, str]) -> List[Tuple[str, str]]:
    keys = [
        ("git_branch", "branch"),
        ("git_short", "git sha"),
        ("dirty", "dirty"),
        ("host", "host"),
        ("cpu", "cpu"),
        ("gpu", "gpu"),
        ("os", "os"),
        ("python", "python"),
        ("lc0_version", "lc0 version"),
        ("lc0_path", "lc0 path"),
        ("net_path", "net"),
        ("net_sha256_prefix", "net sha"),
        ("backend", "backend"),
        ("backend_opts", "backend opts"),
        ("quick", "quick mode"),
        ("started_utc", "started"),
        ("finished_utc", "finished"),
        ("total_seconds", "duration (s)"),
    ]
    rows: List[Tuple[str, str]] = []
    for k, label in keys:
        v = build_info.get(k, "").strip()
        if not v:
            continue
        if k in ("lc0_path", "net_path"):
            rows.append((label, _esc_tt(v)))
        elif k in ("git_short", "lc0_version", "net_sha256_prefix"):
            rows.append((label, _esc_tt(v)))
        else:
            rows.append((label, _esc(v)))
    return rows


def _abstract_run(scores: Dict[str, str], build_info: Dict[str, str]) -> str:
    bits: List[str] = []
    cs = _to_float(scores.get("correctness_score"))
    ss = _to_float(scores.get("stability_score"))
    if cs is not None:
        bits.append(f"correctness {cs:.1f}/100")
    if ss is not None:
        bits.append(f"stability {ss:.1f}/100")
    if scores.get("backend_peak_nps"):
        bits.append(
            f"backend peak {scores['backend_peak_nps']} NPS at "
            f"batch {scores.get('backend_peak_batch','?')}"
        )
    if scores.get("search_total_nps"):
        bits.append(f"search total {scores['search_total_nps']} NPS")
    score_blurb = ", ".join(_esc(b) for b in bits) if bits else "no scores recorded"
    gpu = _esc(build_info.get("gpu", "(unknown GPU)"))
    net = _esc(Path(build_info.get("net_path", "")).name or "(no net)")
    backend = _esc(build_info.get("backend") or "default")
    duration = _esc(build_info.get("total_seconds", "?"))
    quick = build_info.get("quick", "False").lower() == "true"
    mode = "quick" if quick else "full"
    return (
        f"Automated {mode}-mode evaluation pass on {gpu}. "
        f"Net: \\texttt{{{net}}}, backend: \\texttt{{{backend}}}. "
        f"Total duration {duration}\\,s. Headline scores: {score_blurb}."
    )


def _phase_status_table(build_info: Dict[str, str], phase_keys: List[str]) -> str:
    """Phase status block (status, duration) parsed from build_info."""
    rows: List[Tuple[str, str]] = []
    for k in phase_keys:
        st = build_info.get(f"phase_{k}_status", "")
        dur = build_info.get(f"phase_{k}_duration_s", "")
        if not st:
            continue
        color = "good" if st == "OK" else ("bad" if st == "FAILED" else "neutral")
        st_tex = f"\\textcolor{{{color}}}{{\\bfseries {_esc(st)}}}"
        dur_tex = _esc(dur) + "\\,s" if dur else ""
        rows.append((_esc(k), f"{st_tex} \\hfill {dur_tex}"))
    return _kv_table_tex_pretyped(rows)


def _delta_color(value: Optional[float]) -> str:
    if value is None:
        return "neutral"
    if value > 0:
        return "good"
    if value < 0:
        return "bad"
    return "neutral"


def _fmt_delta_tex(cur: Optional[float], base: Optional[float],
                   precision: int = 1, unit: str = "") -> str:
    if cur is None or base is None:
        return "---"
    delta = cur - base
    sign = "+" if delta >= 0 else ""
    color = _delta_color(delta)
    if precision == 0:
        body = f"{sign}{delta:.0f}{unit}"
    else:
        body = f"{sign}{delta:.{precision}f}{unit}"
    if base:
        pct = 100.0 * delta / base
        body = f"{body} ({sign}{pct:.2f}\\%)"
    return f"\\textcolor{{{color}}}{{{_esc(body) if False else body}}}"


def _score_delta_table_tex(b_scores: Dict[str, str],
                           c_scores: Dict[str, str]) -> str:
    rows = []
    metrics = [
        ("correctness_score", "correctness\\_score", "", 1),
        ("stability_score",   "stability\\_score",   "", 1),
        ("unit_tests_pct",    "unit\\_tests\\_pct",  "\\%", 2),
        ("tactics_pct",       "tactics\\_pct",       "\\%", 2),
        ("backend_peak_nps",  "backend\\_peak\\_nps", "", 0),
        ("search_total_nps",  "search\\_total\\_nps", "", 0),
    ]
    for key, label, unit, prec in metrics:
        bv = _to_float(b_scores.get(key))
        cv = _to_float(c_scores.get(key))
        if bv is None and cv is None:
            continue
        bv_s = "---" if bv is None else (f"{bv:.{prec}f}{unit}" if prec
                                          else f"{int(bv)}{unit}")
        cv_s = "---" if cv is None else (f"{cv:.{prec}f}{unit}" if prec
                                          else f"{int(cv)}{unit}")
        delta_s = _fmt_delta_tex(cv, bv, precision=prec, unit=unit)
        rows.append((label, bv_s, cv_s, delta_s))
    if not rows:
        return r"\textit{(no comparable metrics)}"
    out = [
        r"\begin{tabularx}{\linewidth}{@{}>{\bfseries}lL{0.20\linewidth}L{0.20\linewidth}X@{}}",
        r"\toprule",
        r"\textbf{metric} & \textbf{baseline} & \textbf{current} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]
    for label, b, c, d in rows:
        out.append(f"{label} & {b} & {c} & {d} \\\\")
    out += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(out)


def _search_table_tex(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return r"\textit{(no data)}"
    out = [
        r"\begin{tabularx}{\linewidth}{@{}rrrX@{}}",
        r"\toprule",
        r"\textbf{pos} & \textbf{NPS} & \textbf{nodes} & \textbf{bestmove} \\",
        r"\midrule",
    ]
    for r in rows:
        out.append(
            f"{_esc(r.get('position_idx',''))} & "
            f"{_esc(r.get('nps',''))} & "
            f"{_esc(r.get('nodes',''))} & "
            f"\\texttt{{{_esc(r.get('bestmove',''))}}} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(out)


def _tactics_table_tex(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return r"\textit{(no data)}"
    out = [
        r"\begin{tabularx}{\linewidth}{@{}lcL{0.18\linewidth}L{0.16\linewidth}rr@{}}",
        r"\toprule",
        r"\textbf{id} & \textbf{ok} & \textbf{expected} & \textbf{engine} & "
        r"\textbf{eval (P)} & \textbf{nodes} \\",
        r"\midrule",
    ]
    for r in rows:
        solved = (r.get("solved") or "0") == "1"
        ok = (r"\textcolor{good}{\ensuremath{\checkmark}}" if solved
              else r"\textcolor{bad}{\ensuremath{\times}}")
        cp = _to_int(r.get("score_cp"))
        if cp is None:
            eval_s = "---"
        elif abs(cp) >= 50000:
            eval_s = "mate"
        else:
            eval_s = f"{cp/100:+.2f}"
        # collapse multi-bm
        exp = r.get("expected_bm", "")
        exp_list = exp.split()
        if len(exp_list) > 1:
            exp_disp = (r"\texttt{" + _esc(exp_list[0]) + "}" +
                        f" \\,\\textcolor{{neutral}}{{(+{len(exp_list) - 1})}}")
        else:
            exp_disp = r"\texttt{" + _esc(exp) + "}"
        out.append(
            f"{_esc(r.get('id',''))} & {ok} & {exp_disp} & "
            f"\\texttt{{{_esc(r.get('engine_bestmove',''))}}} & "
            f"{eval_s} & {_esc(r.get('nodes',''))} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(out)


def _unit_failures_tex(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return r"All %d unit tests passed.\par" % 0
    out = [
        r"\begin{tabularx}{\linewidth}{@{}lL{0.30\linewidth}lX@{}}",
        r"\toprule",
        r"\textbf{suite} & \textbf{name} & \textbf{status} & \textbf{message} \\",
        r"\midrule",
    ]
    for r in rows:
        out.append(
            f"\\texttt{{{_esc(r.get('suite',''))}}} & "
            f"\\texttt{{{_esc(r.get('name',''))}}} & "
            f"\\textcolor{{bad}}{{{_esc(r.get('status',''))}}} & "
            f"\\small {_esc(r.get('message','')[:200])} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(out)


def _build_info_table_tex(build_info: Dict[str, str]) -> str:
    """Description-list dump of every key/value in build_info. We use
    enumitem's description with multiline style so long values wrap
    cleanly under the key without overflowing tabularx columns."""
    if not build_info:
        return r"\textit{(no build info)}"
    out = [
        r"\begin{description}[font=\ttfamily\bfseries, "
        r"leftmargin=4cm, labelindent=0pt, labelwidth=3.6cm, "
        r"style=multiline, itemsep=2pt, parsep=0pt, topsep=4pt]",
    ]
    phase_keys = sorted(k for k in build_info if k.startswith("phase_"))
    other_keys = sorted(k for k in build_info if not k.startswith("phase_"))
    for k in other_keys + phase_keys:
        v = build_info[k]
        if not v:
            v_tex = r"\textit{(empty)}"
        elif k.endswith("_status"):
            color = "good" if v == "OK" else ("bad" if v == "FAILED" else "neutral")
            v_tex = f"\\textcolor{{{color}}}{{{_esc(v)}}}"
        else:
            # Apply \seqsplit per-token so we can wrap inside long
            # unbreakable runs (paths, hashes) without eating the spaces
            # that separate tokens (e.g. GPU descriptions).
            parts_v: List[str] = []
            for tok in v.split(" "):
                inner = _esc(tok)
                if len(tok) > 30:
                    parts_v.append(r"\seqsplit{" + inner + "}")
                else:
                    parts_v.append(inner)
            v_tex = r"{\ttfamily\small " + " ".join(parts_v) + "}"
        out.append(f"\\item[{_esc(k)}] {v_tex}")
    out.append(r"\end{description}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def _compile_pdf(tex_source: str, out_pdf: Path) -> Path:
    """Compile a .tex string to a PDF at out_pdf. Uses pdflatex twice for
    cross-references. Auxiliary files are kept in a temp dir and removed."""
    if shutil.which("pdflatex") is None:
        raise RuntimeError(
            "pdflatex not found on PATH; install texlive-latex-recommended "
            "(see scripts/eval/README.md)"
        )
    with TempBuildDir(out_pdf.stem) as build_dir:
        tex_file = build_dir / "doc.tex"
        tex_file.write_text(tex_source, encoding="utf-8")
        # PDF figures are referenced via relative paths from build_dir;
        # we expect the caller to drop them into <build_dir>/figs/.
        for _ in range(2):  # run twice for refs / fancyhdr
            cp = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-halt-on-error", "doc.tex"],
                cwd=str(build_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
            if cp.returncode != 0:
                log_path = build_dir / "doc.log"
                tail = ""
                if log_path.is_file():
                    tail = log_path.read_text(errors="replace").splitlines()[-40:]
                    tail = "\n".join(tail)
                raise RuntimeError(
                    f"pdflatex failed (rc={cp.returncode}). "
                    f"Last 40 log lines:\n{tail}"
                )
        produced = build_dir / "doc.pdf"
        if not produced.is_file():
            raise RuntimeError("pdflatex completed but no PDF was produced")
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, out_pdf)
    return out_pdf


class TempBuildDir:
    """Temp dir context manager. Caller can mkdir 'figs' inside before
    invoking pdflatex; we expose this dir so figure generation can drop
    PDFs into <dir>/figs/ alongside the .tex source."""
    def __init__(self, label: str = "lc0eval"):
        self.label = label
        self._tmp: Optional[Path] = None

    def __enter__(self) -> Path:
        import tempfile
        self._tmp = Path(tempfile.mkdtemp(prefix=f"lc0eval-{self.label}-"))
        (self._tmp / "figs").mkdir()
        return self._tmp

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmp and self._tmp.is_dir():
            shutil.rmtree(self._tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public: render run report
# ---------------------------------------------------------------------------

def render_run_pdf(report_dir: Path) -> Path:
    report_dir = Path(report_dir).resolve()
    if not report_dir.is_dir():
        raise FileNotFoundError(report_dir)

    scores = _read_scores(report_dir)
    backend_rows = _read_csv(report_dir / "backend_bench.csv")
    search_rows = _read_csv(report_dir / "search_bench.csv")
    tactics_rows = _read_csv(report_dir / "tactics.csv")
    unit_rows = _read_csv(report_dir / "unit_tests.csv")
    failed_unit = [r for r in unit_rows
                   if (r.get("status") or "").lower()
                   in ("failed", "errored", "error")]
    build_info = _read_build_info_md(report_dir / "build_info.md")

    sha_short = build_info.get("git_short") or build_info.get("git_sha", "")[:7]
    started = build_info.get("started_utc", "") or "(unknown)"

    out_pdf = report_dir / "report.pdf"
    with TempBuildDir("run") as build_dir:
        figs_dir = build_dir / "figs"
        # render figures
        if (f := _fig_radar(scores)) is not None:
            _save_fig(f, figs_dir / "radar.pdf")
        if backend_rows:
            if (f := _fig_backend_nps(backend_rows)) is not None:
                _save_fig(f, figs_dir / "backend_nps.pdf")
            if (f := _fig_backend_latency(backend_rows)) is not None:
                _save_fig(f, figs_dir / "backend_latency.pdf")
        if search_rows:
            if (f := _fig_search_per_position(search_rows)) is not None:
                _save_fig(f, figs_dir / "search_per_position.pdf")
        if tactics_rows:
            if (f := _fig_tactics_solved(tactics_rows)) is not None:
                _save_fig(f, figs_dir / "tactics.pdf")

        tex = _build_run_tex(
            report_dir=report_dir,
            scores=scores,
            backend_rows=backend_rows,
            search_rows=search_rows,
            tactics_rows=tactics_rows,
            unit_rows=unit_rows,
            failed_unit=failed_unit,
            build_info=build_info,
            sha=sha_short,
            started=started,
            figs_present=(figs_dir.iterdir().__next__() is not None
                          if any(figs_dir.iterdir()) else False),
        )

        tex_file = build_dir / "doc.tex"
        tex_file.write_text(tex, encoding="utf-8")
        _run_pdflatex_twice(build_dir)
        produced = build_dir / "doc.pdf"
        if not produced.is_file():
            raise RuntimeError("pdflatex completed but no PDF was produced")
        shutil.copyfile(produced, out_pdf)
    return out_pdf


def render_compare_pdf(baseline_dir: Path, current_dir: Path) -> Path:
    baseline_dir = Path(baseline_dir).resolve()
    current_dir = Path(current_dir).resolve()
    if not baseline_dir.is_dir() or not current_dir.is_dir():
        raise FileNotFoundError(f"{baseline_dir} or {current_dir}")

    b_scores = _read_scores(baseline_dir)
    c_scores = _read_scores(current_dir)
    b_info = _read_build_info_md(baseline_dir / "build_info.md")
    c_info = _read_build_info_md(current_dir / "build_info.md")
    b_backend = _read_csv(baseline_dir / "backend_bench.csv")
    c_backend = _read_csv(current_dir / "backend_bench.csv")
    b_search = _read_csv(baseline_dir / "search_bench.csv")
    c_search = _read_csv(current_dir / "search_bench.csv")
    b_tactics = _read_csv(baseline_dir / "tactics.csv")
    c_tactics = _read_csv(current_dir / "tactics.csv")

    sha_short = c_info.get("git_short") or c_info.get("git_sha", "")[:7]
    started = c_info.get("started_utc", "") or "(unknown)"

    out_pdf = current_dir / "compare.pdf"
    with TempBuildDir("compare") as build_dir:
        figs_dir = build_dir / "figs"
        if (f := _fig_compare_score_deltas(b_scores, c_scores)) is not None:
            _save_fig(f, figs_dir / "score_deltas.pdf")
        if (f := _fig_compare_backend_overlay(b_backend, c_backend)) is not None:
            _save_fig(f, figs_dir / "backend_overlay.pdf")
        if (f := _fig_compare_backend_delta(b_backend, c_backend)) is not None:
            _save_fig(f, figs_dir / "backend_delta.pdf")
        if (f := _fig_compare_search_overlay(b_search, c_search)) is not None:
            _save_fig(f, figs_dir / "search_overlay.pdf")

        tex = _build_compare_tex(
            baseline_dir=baseline_dir, current_dir=current_dir,
            b_scores=b_scores, c_scores=c_scores,
            b_info=b_info, c_info=c_info,
            b_backend=b_backend, c_backend=c_backend,
            b_search=b_search, c_search=c_search,
            b_tactics=b_tactics, c_tactics=c_tactics,
            sha=sha_short, started=started,
        )
        (build_dir / "doc.tex").write_text(tex, encoding="utf-8")
        _run_pdflatex_twice(build_dir)
        shutil.copyfile(build_dir / "doc.pdf", out_pdf)
    return out_pdf


def _run_pdflatex_twice(build_dir: Path) -> None:
    if shutil.which("pdflatex") is None:
        raise RuntimeError(
            "pdflatex not found on PATH; install texlive-latex-recommended"
        )
    for _ in range(2):
        cp = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
            cwd=str(build_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace",
        )
        if cp.returncode != 0:
            log_path = build_dir / "doc.log"
            tail = ""
            if log_path.is_file():
                tail = "\n".join(
                    log_path.read_text(errors="replace").splitlines()[-50:]
                )
            raise RuntimeError(
                f"pdflatex failed (rc={cp.returncode}). Last log lines:\n{tail}"
            )


# ---------------------------------------------------------------------------
# .tex assemblers
# ---------------------------------------------------------------------------

def _build_run_tex(*, report_dir: Path,
                   scores: Dict[str, str],
                   backend_rows: List[Dict[str, str]],
                   search_rows: List[Dict[str, str]],
                   tactics_rows: List[Dict[str, str]],
                   unit_rows: List[Dict[str, str]],
                   failed_unit: List[Dict[str, str]],
                   build_info: Dict[str, str],
                   sha: str, started: str,
                   figs_present: bool = True) -> str:

    parts: List[str] = []
    parts.append(_preamble(
        title="lc0 Evaluation Report",
        sha=sha,
        date=started,
    ))

    # ---- abstract ----
    parts.append(r"\begin{abstract}")
    parts.append(r"\noindent " + _abstract_run(scores, build_info))
    parts.append(r"\end{abstract}")

    # ---- 1. Run metadata ----
    parts.append(r"\section{Run Metadata}")
    parts.append(_kv_table_tex_pretyped(_build_metadata_rows(build_info)))

    # ---- 2. Headline scores ----
    parts.append(r"\section{Headline Scores}")
    parts.append(
        r"This section reports the composite scores computed by "
        r"\texttt{compute\_scores()}: \emph{correctness} averages the "
        r"per-phase correctness rates that were measured (unit-test pass "
        r"rate, tactics solved rate); \emph{stability} reports the "
        r"fraction of non-skipped phases that completed successfully. "
        r"Performance numbers are kept in raw units because absolute "
        r"NPS is hardware-dependent.\par\bigskip"
    )
    parts.append(_scores_table_tex(scores))
    if (Path("figs") / "radar.pdf").as_posix() and (build_info_has_radar := True):
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.6\linewidth]{figs/radar.pdf}")
        parts.append(r"\caption{Score radar (each axis normalised to 0--100).}")
        parts.append(r"\label{fig:radar}")
        parts.append(r"\end{figure}")

    # ---- 3. Backend bench ----
    if backend_rows:
        parts.append(r"\section{Backend Microbenchmark}")
        parts.append(
            r"\texttt{lc0 backendbench} measures the neural network's "
            r"forward-pass throughput at a sweep of batch sizes. NPS "
            r"normally rises with batch size until the GPU saturates; "
            r"latency rises monotonically. The first batch is "
            r"systematically slower because of CUDA-graph capture, "
            r"cuBLAS auto-tune, and JIT compilation; we report the "
            r"\emph{first-batch max} as a separate trace."
        )
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.95\linewidth]{figs/backend_nps.pdf}")
        parts.append(r"\caption{Backend throughput as a function of batch size. "
                     r"Shaded band shows the per-batch \texttt{min..max} envelope; "
                     r"the green marker is the peak mean.}")
        parts.append(r"\label{fig:backend-nps}")
        parts.append(r"\end{figure}")
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.95\linewidth]{figs/backend_latency.pdf}")
        parts.append(r"\caption{Per-batch latency. Error bars show one standard "
                     r"deviation over the warm-batch population; the dotted line "
                     r"shows the cold-batch maximum (first invocation).}")
        parts.append(r"\label{fig:backend-latency}")
        parts.append(r"\end{figure}")

    # ---- 4. Search bench ----
    if search_rows:
        parts.append(r"\section{Search Benchmark}")
        movetime = _movetime_from_logs(report_dir, "search_bench.log")
        mt_text = (f" (\\texttt{{--movetime={movetime}}}\\,ms per position)"
                   if movetime else "")
        parts.append(
            r"\texttt{lc0 bench} runs a fixed-time search on each position "
            r"and reports total nodes / NPS" + mt_text + r". "
            r"The bars below show per-position NPS; horizontal lines "
            r"are the mean and median across the suite. NPS is sensitive "
            r"to the working batch size that the search assembles, so "
            r"per-position variance is expected."
        )
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.95\linewidth]{figs/search_per_position.pdf}")
        parts.append(r"\caption{Per-position NPS in the search benchmark, with "
                     r"the engine's chosen \texttt{bestmove} annotated above each bar.}")
        parts.append(r"\label{fig:search}")
        parts.append(r"\end{figure}")
        parts.append(r"\par\bigskip")
        parts.append(_search_table_tex(search_rows))

    # ---- 5. Tactics ----
    if tactics_rows:
        parts.append(r"\section{Tactics Suite}")
        solved = sum(1 for r in tactics_rows
                     if (r.get("solved") or "0") == "1")
        total = len(tactics_rows)
        pct = 100.0 * solved / total if total else 0.0
        parts.append(
            f"The tactics suite asks the engine to find the best move on "
            f"each of {total} hand-picked positions within a short time "
            f"budget, and counts a position as solved if the engine's "
            f"\\texttt{{bestmove}} matches one of the accepted moves "
            f"(\\texttt{{bm}} operands). The engine solved \\textbf{{{solved}/{total}}} "
            f"({pct:.1f}\\%)."
        )
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.85\linewidth]{figs/tactics.pdf}")
        parts.append(r"\caption{Tactics outcomes by position id. "
                     r"Green: solved; red: failed.}")
        parts.append(r"\label{fig:tactics}")
        parts.append(r"\end{figure}")
        parts.append(r"\par\bigskip")
        parts.append(_tactics_table_tex(tactics_rows))

    # ---- 6. Unit tests ----
    parts.append(r"\section{Unit Tests}")
    n = len(unit_rows)
    nf = len(failed_unit)
    if n == 0:
        parts.append(r"\textit{No unit-test data was recorded for this run.}")
    elif nf == 0:
        parts.append(
            f"All \\textbf{{{n}}} unit tests "
            r"(\texttt{ninja\,-C\,builddir\,test}) passed.")
    else:
        parts.append(
            f"{n - nf}/{n} unit tests passed; "
            f"{nf} failed or errored. Failure details:")
        parts.append(_unit_failures_tex(failed_unit))

    # ---- Appendix: build info ----
    parts.append(r"\appendix")
    parts.append(r"\section{Build Information}")
    parts.append(
        r"This appendix dumps the full key/value snapshot that "
        r"\texttt{eval.py} captured for this run, for traceability. "
        r"\emph{Phase} keys (\texttt{phase\_<name>\_status}, etc.) are "
        r"the per-phase outcome lines.")
    parts.append(_build_info_table_tex(build_info))

    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def _build_compare_tex(*, baseline_dir: Path, current_dir: Path,
                       b_scores, c_scores, b_info, c_info,
                       b_backend, c_backend,
                       b_search, c_search,
                       b_tactics, c_tactics,
                       sha: str, started: str) -> str:
    parts: List[str] = []
    parts.append(_preamble(
        title="lc0 Evaluation Compare Report",
        sha=sha,
        date=started,
    ))

    # ---- abstract ----
    bits: List[str] = []
    for key, label, prec in [
        ("correctness_score", "correctness", 1),
        ("stability_score",   "stability",   1),
        ("backend_peak_nps",  "backend peak NPS", 0),
        ("search_total_nps",  "search total NPS", 0),
    ]:
        bv = _to_float(b_scores.get(key))
        cv = _to_float(c_scores.get(key))
        if bv is None or cv is None:
            continue
        delta = cv - bv
        sign = "+" if delta >= 0 else ""
        if prec:
            bits.append(f"{label} {bv:.{prec}f} $\\to$ {cv:.{prec}f} ({sign}{delta:.{prec}f})")
        else:
            pct = (100.0 * delta / bv) if bv else 0
            bits.append(f"{label} {int(bv)} $\\to$ {int(cv)} ({sign}{int(delta)}, {sign}{pct:.2f}\\%)")
    abstract = ("Pairwise comparison of two evaluation runs. "
                "Baseline: \\texttt{" + _esc(baseline_dir.name) + "}; "
                "current: \\texttt{" + _esc(current_dir.name) + "}. "
                "Summary: " + ("; ".join(bits) if bits else "no comparable scores") + ".")
    parts.append(r"\begin{abstract}")
    parts.append(r"\noindent " + abstract)
    parts.append(r"\end{abstract}")

    # ---- 1. Subjects ----
    parts.append(r"\section{Subjects}")
    rows = []
    rows.append(("baseline path", _esc_tt(str(baseline_dir))))
    rows.append(("baseline sha",  _esc_tt(b_info.get("git_short", ""))))
    rows.append(("baseline date", _esc(b_info.get("started_utc", ""))))
    rows.append(("current path",  _esc_tt(str(current_dir))))
    rows.append(("current sha",   _esc_tt(c_info.get("git_short", ""))))
    rows.append(("current date",  _esc(c_info.get("started_utc", ""))))
    parts.append(_kv_table_tex_pretyped(rows))

    # ---- 2. Score deltas ----
    parts.append(r"\section{Score Deltas}")
    parts.append(
        r"Higher-is-better metrics shown with \textcolor{good}{green} for "
        r"improvements and \textcolor{bad}{red} for regressions. "
        r"Absolute NPS deltas are hardware-dependent; on a stable machine "
        r"they reflect engine-level changes."
    )
    parts.append(_score_delta_table_tex(b_scores, c_scores))
    parts.append(r"\begin{figure}[H]")
    parts.append(r"\centering")
    parts.append(r"\includegraphics[width=0.95\linewidth]{figs/score_deltas.pdf}")
    parts.append(r"\caption{Headline-metric deltas as percent of baseline.}")
    parts.append(r"\end{figure}")

    # ---- 3. Backend bench overlay ----
    if b_backend or c_backend:
        parts.append(r"\section{Backend Microbenchmark}")
        parts.append(
            r"NPS curves overlaid; positive bars in the delta panel "
            r"indicate batch sizes where the current build is faster.")
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.95\linewidth]{figs/backend_overlay.pdf}")
        parts.append(r"\caption{Backend NPS, baseline (grey) vs current (blue).}")
        parts.append(r"\end{figure}")
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.95\linewidth]{figs/backend_delta.pdf}")
        parts.append(r"\caption{Per-batch percent change in mean NPS.}")
        parts.append(r"\end{figure}")

    # ---- 4. Search bench overlay ----
    if b_search or c_search:
        parts.append(r"\section{Search Benchmark}")
        b_total = sum(_to_float(r["nps"]) or 0 for r in b_search)
        c_total = sum(_to_float(r["nps"]) or 0 for r in c_search)
        delta = c_total - b_total
        pct = (100.0 * delta / b_total) if b_total else 0.0
        sign = "+" if delta >= 0 else ""
        color = _delta_color(delta)
        parts.append(
            r"Total search NPS: baseline " + f"{int(b_total)}, current "
            f"{int(c_total)} "
            r"(\textcolor{" + color + "}{" + sign +
            f"{int(delta)}, {sign}{pct:.2f}" + r"\%}).\par\medskip"
        )
        parts.append(r"\begin{figure}[H]")
        parts.append(r"\centering")
        parts.append(r"\includegraphics[width=0.95\linewidth]{figs/search_overlay.pdf}")
        parts.append(r"\caption{Per-position NPS, baseline vs current.}")
        parts.append(r"\end{figure}")

    # ---- 5. Tactics flips ----
    if b_tactics or c_tactics:
        parts.append(r"\section{Tactics Flips}")
        by_b = {r["id"]: r for r in b_tactics}
        by_c = {r["id"]: r for r in c_tactics}
        flips = []
        for k in sorted(set(by_b) | set(by_c)):
            bs = (by_b.get(k, {}).get("solved") or "0") == "1"
            cs = (by_c.get(k, {}).get("solved") or "0") == "1"
            if bs == cs:
                continue
            flips.append((k, bs, cs,
                          by_b.get(k, {}).get("engine_bestmove", ""),
                          by_c.get(k, {}).get("engine_bestmove", "")))
        b_solved = sum(1 for r in b_tactics if (r.get("solved") or "0") == "1")
        c_solved = sum(1 for r in c_tactics if (r.get("solved") or "0") == "1")
        parts.append(
            f"Baseline {b_solved}/{len(b_tactics)} \\,$\\to$\\, "
            f"current {c_solved}/{len(c_tactics)}."
        )
        if not flips:
            parts.append(r"\par\medskip\textit{No solved-status changes between the two runs.}")
        else:
            parts.append(
                r"\par\medskip"
                r"\begin{tabularx}{\linewidth}{@{}lccL{0.20\linewidth}L{0.20\linewidth}l@{}}"
                "\n"
                r"\toprule"
                "\n"
                r"\textbf{id} & \textbf{base} & \textbf{cur} & "
                r"\textbf{engine (base)} & \textbf{engine (cur)} & "
                r"\textbf{change} \\"
                "\n"
                r"\midrule"
            )
            for k, bs, cs, be, ce in flips:
                base_g = (r"\textcolor{good}{\ensuremath{\checkmark}}" if bs
                          else r"\textcolor{bad}{\ensuremath{\times}}")
                cur_g  = (r"\textcolor{good}{\ensuremath{\checkmark}}" if cs
                          else r"\textcolor{bad}{\ensuremath{\times}}")
                change = (r"\textcolor{good}{improved}" if (cs and not bs)
                          else r"\textcolor{bad}{regressed}")
                parts.append(
                    f"\\texttt{{{_esc(k)}}} & {base_g} & {cur_g} & "
                    f"\\texttt{{{_esc(be)}}} & \\texttt{{{_esc(ce)}}} & "
                    f"{change} \\\\"
                )
            parts.append(r"\bottomrule")
            parts.append(r"\end{tabularx}")

    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"
