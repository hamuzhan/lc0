"""PDF report generator for eval-reports/ artifacts.

This module imports matplotlib at load time, so callers should lazy-import
it (only when --pdf is requested) to keep eval.py's default execution path
stdlib-only.

Public entry points:
    render_run_pdf(report_dir)       -> Path to report.pdf
    render_compare_pdf(baseline, current) -> Path to compare.pdf

Both produce A4-portrait, vector PDFs with the cover page summarising
metadata + scores, followed by one chart page per phase.
"""

from __future__ import annotations

import csv
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# ---------------------------------------------------------------------------
# constants & style
# ---------------------------------------------------------------------------

A4_PORTRAIT = (8.27, 11.69)   # inches
A4_LANDSCAPE = (11.69, 8.27)

COLOR_BASELINE = "#888888"
COLOR_CURRENT = "#1f77b4"
COLOR_GOOD = "#2ca02c"
COLOR_BAD = "#d62728"
COLOR_NEUTRAL = "#7f7f7f"
COLOR_HEADER_BG = "#e8eef7"
COLOR_ZEBRA = "#f6f8fa"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 100,
    "pdf.fonttype": 42,        # embed TrueType, not Type 3
    "ps.fonttype": 42,
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
                # blank or section break ends the table
                if not s or s.startswith("#"):
                    in_table = False
                    continue
                continue
            if "---" in s:
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 2:
                k, v = cells[0], cells[1].strip("`").strip()
                # collapse <br> joined GPU lists to space-separated
                v = v.replace("<br>", " ").replace("`", "").strip()
                out[k] = v
    return out


def _to_float(s: Any) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s: Any) -> Optional[int]:
    f = _to_float(s)
    return int(f) if f is not None else None


# ---------------------------------------------------------------------------
# helpers: figure / page chrome
# ---------------------------------------------------------------------------

def _new_page(landscape: bool = False) -> Figure:
    size = A4_LANDSCAPE if landscape else A4_PORTRAIT
    fig = plt.figure(figsize=size)
    return fig


def _add_footer(fig: Figure, page_num: int, total_pages: int,
                sha: Optional[str] = None,
                ts: Optional[str] = None,
                left_label: Optional[str] = None) -> None:
    bits: List[str] = []
    if left_label:
        bits.append(left_label)
    if sha:
        bits.append(f"sha {sha}")
    if ts:
        bits.append(ts)
    bits.append(f"page {page_num}/{total_pages}")
    fig.text(0.5, 0.012, " · ".join(bits), ha="center", va="bottom",
             fontsize=7, color="#777777")


def _add_title_block(fig: Figure, title: str, subtitle: Optional[str] = None,
                     y: float = 0.96) -> None:
    fig.text(0.06, y, title, fontsize=20, weight="bold", va="top")
    if subtitle:
        fig.text(0.06, y - 0.025, subtitle, fontsize=10,
                 color="#444", va="top")


def _draw_kv_table(ax, headers: List[str], rows: List[List[str]],
                   col_weights: Optional[List[float]] = None,
                   row_height: float = 0.045,
                   header_bg: str = COLOR_HEADER_BG,
                   zebra: str = COLOR_ZEBRA,
                   font_size: int = 9,
                   max_rows: Optional[int] = None) -> None:
    """Render a simple table using ax.text + Rectangle patches."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                color="#999")
        return
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows] + [["…"] * len(headers)]

    n_cols = len(headers)
    if col_weights is None:
        col_weights = [1.0] * n_cols
    total_w = sum(col_weights)
    widths = [w / total_w for w in col_weights]
    edges = [0.0]
    for w in widths:
        edges.append(edges[-1] + w)

    n_rows = len(rows)
    total_h = (n_rows + 1) * row_height
    y_top = 0.5 + total_h / 2

    # header
    yh = y_top - row_height
    ax.add_patch(Rectangle((0, yh), 1, row_height,
                           facecolor=header_bg, edgecolor="none"))
    for i, h in enumerate(headers):
        ax.text(edges[i] + 0.01, yh + row_height / 2, h,
                ha="left", va="center", fontsize=font_size, weight="bold")

    # body
    for ridx, row in enumerate(rows):
        y = y_top - (ridx + 2) * row_height
        if ridx % 2 == 1:
            ax.add_patch(Rectangle((0, y), 1, row_height,
                                   facecolor=zebra, edgecolor="none"))
        for cidx, cell in enumerate(row):
            ax.text(edges[cidx] + 0.01, y + row_height / 2,
                    str(cell), ha="left", va="center", fontsize=font_size)

    # bottom border
    ax.add_patch(Rectangle((0, y_top - total_h), 1, total_h,
                           fill=False, edgecolor="#cccccc", linewidth=0.6))


def _delta_color(delta: Optional[float], higher_is_better: bool = True) -> str:
    if delta is None or delta == 0:
        return COLOR_NEUTRAL
    if higher_is_better:
        return COLOR_GOOD if delta > 0 else COLOR_BAD
    return COLOR_GOOD if delta < 0 else COLOR_BAD


def _fmt_num(v: Any, precision: int = 1, unit: str = "",
             default: str = "—") -> str:
    f = _to_float(v)
    if f is None:
        return default
    if precision == 0:
        return f"{f:.0f}{unit}"
    return f"{f:.{precision}f}{unit}"


def _fmt_delta(cur: Optional[float], base: Optional[float],
               unit: str = "", precision: int = 1) -> str:
    if cur is None or base is None:
        return "—"
    delta = cur - base
    if base:
        pct = 100.0 * delta / base
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.{precision}f}{unit} ({sign}{pct:.2f}%)"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{precision}f}{unit}"


# ---------------------------------------------------------------------------
# RUN: per-page renderers
# ---------------------------------------------------------------------------

def _page_cover_run(fig: Figure,
                    scores: Dict[str, str],
                    build_info: Dict[str, str]) -> None:
    title = "lc0 eval report"
    sha_short = build_info.get("git_short", "")
    started = build_info.get("started_utc", "")
    sub_bits = []
    if sha_short:
        sub_bits.append(f"git {sha_short}")
    if build_info.get("dirty", "").lower() in ("true", "1"):
        sub_bits.append("dirty")
    if started:
        sub_bits.append(started)
    _add_title_block(fig, title, " · ".join(sub_bits) or None)

    # ---- meta block ----
    meta_keys = [
        ("git_branch", "branch"),
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
        ("quick", "quick"),
        ("started_utc", "started"),
        ("finished_utc", "finished"),
        ("total_seconds", "duration (s)"),
    ]
    rows = []
    for k, label in meta_keys:
        v = build_info.get(k, "").strip()
        if not v:
            continue
        if k == "lc0_path" or k == "net_path":
            v = _shorten_path(v, max_len=70)
        rows.append([label, v])

    ax_meta = fig.add_axes([0.06, 0.50, 0.88, 0.40])
    _draw_kv_table(ax_meta, ["key", "value"], rows,
                   col_weights=[1, 4], row_height=0.052, font_size=9)
    fig.text(0.06, 0.91, "Run metadata", fontsize=12, weight="bold")

    # ---- score table ----
    score_rows = []
    cs = scores.get("correctness_score", "")
    ss = scores.get("stability_score", "")
    if cs:
        score_rows.append(["correctness_score", f"{_to_float(cs):.1f} / 100"])
    if ss:
        score_rows.append(["stability_score", f"{_to_float(ss):.1f} / 100"])
    if scores.get("unit_tests_total"):
        score_rows.append([
            "unit tests",
            f"{scores.get('unit_tests_passed','')}/{scores.get('unit_tests_total','')} "
            f"({_fmt_num(scores.get('unit_tests_pct'), 1, '%')})",
        ])
    if scores.get("tactics_total"):
        score_rows.append([
            "tactics solved",
            f"{scores.get('tactics_solved','')}/{scores.get('tactics_total','')} "
            f"({_fmt_num(scores.get('tactics_pct'), 1, '%')})",
        ])
    if scores.get("backend_peak_nps"):
        score_rows.append([
            "backend peak NPS",
            f"{scores['backend_peak_nps']} @ batch {scores.get('backend_peak_batch','?')}",
        ])
    if scores.get("search_total_nps"):
        score_rows.append(["search total NPS", scores["search_total_nps"]])

    ax_scores = fig.add_axes([0.06, 0.18, 0.55, 0.30])
    _draw_kv_table(ax_scores, ["metric", "value"], score_rows,
                   col_weights=[2, 3], row_height=0.062, font_size=10)
    fig.text(0.06, 0.49, "Scores", fontsize=12, weight="bold")

    # ---- radar chart ----
    radar_metrics = []
    for name, key, scale_max in [
        ("correctness", "correctness_score", 100),
        ("stability",   "stability_score",   100),
        ("tactics",     "tactics_pct",       100),
        ("unit tests",  "unit_tests_pct",    100),
    ]:
        v = _to_float(scores.get(key))
        if v is None:
            continue
        radar_metrics.append((name, v / scale_max))

    if len(radar_metrics) >= 3:
        ax_radar = fig.add_axes([0.62, 0.18, 0.34, 0.30], projection="polar")
        _draw_radar(ax_radar, radar_metrics)


def _draw_radar(ax, metrics: List[Tuple[str, float]],
                color: str = COLOR_CURRENT) -> None:
    labels = [m[0] for m in metrics]
    values = [max(0.0, min(1.0, m[1])) for m in metrics]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]
    ax.plot(angles_closed, values_closed, color=color, linewidth=1.5)
    ax.fill(angles_closed, values_closed, color=color, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#777")
    ax.set_ylim(0, 1.0)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(True, linewidth=0.4, alpha=0.4)


def _page_backend_bench(fig: Figure, rows: List[Dict[str, str]],
                        net_label: str) -> None:
    _add_title_block(fig, "Backend bench", "lc0 backendbench — NPS / latency vs batch size")
    if not rows:
        fig.text(0.5, 0.5, "(no backend_bench data)", ha="center", color="#999")
        return

    batch = np.array([_to_float(r["batch_size"]) or 0 for r in rows])
    mean_nps = np.array([_to_float(r["mean_nps"]) or 0 for r in rows])
    median_nps = np.array([_to_float(r.get("median_nps")) or 0 for r in rows])
    mean_ms = np.array([_to_float(r["mean_ms"]) or 0 for r in rows])
    sdev_ms = np.array([_to_float(r.get("sdev_ms")) or 0 for r in rows])
    max_nps = np.array([_to_float(r.get("max_nps")) or 0 for r in rows])
    min_nps = np.array([_to_float(r.get("min_nps")) or 0 for r in rows])
    first_max_ms = np.array([_to_float(r.get("first_max_ms")) or np.nan for r in rows])

    # NPS panel
    ax1 = fig.add_axes([0.10, 0.55, 0.84, 0.34])
    ax1.fill_between(batch, min_nps, max_nps, color=COLOR_CURRENT,
                     alpha=0.12, label="min..max", linewidth=0)
    ax1.plot(batch, mean_nps, color=COLOR_CURRENT, marker="o", markersize=4,
             linewidth=1.6, label="mean NPS")
    if np.any(median_nps != mean_nps):
        ax1.plot(batch, median_nps, color=COLOR_CURRENT, linestyle="--",
                 linewidth=1, alpha=0.6, label="median")
    # peak marker
    peak_idx = int(np.argmax(mean_nps))
    ax1.scatter([batch[peak_idx]], [mean_nps[peak_idx]],
                color=COLOR_GOOD, s=80, zorder=5, edgecolor="white",
                linewidth=1.2, label=f"peak {int(mean_nps[peak_idx])} @ batch {int(batch[peak_idx])}")
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("NPS")
    ax1.set_title("NPS vs batch size")
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax1.set_ylim(bottom=0)

    # Latency panel
    ax2 = fig.add_axes([0.10, 0.10, 0.84, 0.34])
    ax2.errorbar(batch, mean_ms, yerr=sdev_ms, color=COLOR_CURRENT,
                 marker="s", markersize=4, linewidth=1.4, capsize=2,
                 label="mean ± stdev")
    if np.any(~np.isnan(first_max_ms)):
        ax2.plot(batch, first_max_ms, color=COLOR_BAD, linestyle=":",
                 marker="^", markersize=4, linewidth=1, alpha=0.85,
                 label="first-batch max (cold)")
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("latency (ms)")
    ax2.set_title("Per-batch forward-pass latency")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.set_ylim(bottom=0)

    fig.text(0.10, 0.04,
             f"net: {_shorten_path(net_label, 80)}",
             fontsize=7, color="#666")


def _page_search_bench(fig: Figure, rows: List[Dict[str, str]],
                       movetime_ms: Optional[int] = None) -> None:
    _add_title_block(fig, "Search bench", "lc0 bench — per-position NPS")
    if not rows:
        fig.text(0.5, 0.5, "(no search_bench data)", ha="center", color="#999")
        return

    idx = np.array([_to_int(r["position_idx"]) or 0 for r in rows])
    nps = np.array([_to_float(r["nps"]) or 0 for r in rows])
    nodes = np.array([_to_int(r.get("nodes")) or 0 for r in rows])
    bestmoves = [r.get("bestmove", "") for r in rows]

    # Chart on top half
    ax = fig.add_axes([0.10, 0.55, 0.84, 0.34])
    ax.bar(idx, nps, color=COLOR_CURRENT, edgecolor="white",
           linewidth=0.6, width=0.7)
    mean_nps = float(np.mean(nps))
    median_nps = float(np.median(nps))
    total_nps = float(np.sum(nps))
    ax.axhline(mean_nps, color=COLOR_NEUTRAL, linestyle="--", linewidth=1,
               label=f"mean {mean_nps:.0f}")
    ax.axhline(median_nps, color=COLOR_NEUTRAL, linestyle=":", linewidth=1,
               label=f"median {median_nps:.0f}")
    ax.set_xticks(idx)
    ax.set_xlabel("position")
    ax.set_ylabel("NPS")
    title = f"Per-position NPS (n={len(rows)}, total {total_nps:.0f})"
    if movetime_ms:
        title += f", movetime {movetime_ms} ms"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    # leave headroom for bestmove labels
    ymax = float(np.max(nps)) if len(nps) else 1.0
    ax.set_ylim(0, ymax * 1.18)
    for b, n, bm in zip(idx, nps, bestmoves):
        ax.text(b, n + ymax * 0.02, bm, ha="center", fontsize=7, color="#444")

    # Per-position table on bottom half
    table_rows = [
        [str(int(i)), f"{int(n_)}", f"{int(nv_)}", bm_]
        for i, n_, nv_, bm_ in zip(idx, nps, nodes, bestmoves)
    ]
    ax_t = fig.add_axes([0.10, 0.06, 0.84, 0.42])
    _draw_kv_table(ax_t, ["pos", "NPS", "nodes", "bestmove"], table_rows,
                   col_weights=[0.5, 1, 1, 1],
                   row_height=0.85 / max(len(rows) + 2, 12),
                   font_size=9)


def _page_tactics(fig: Figure, rows: List[Dict[str, str]]) -> None:
    if not rows:
        _add_title_block(fig, "Tactics", "(no tactics data)")
        return
    solved = sum(1 for r in rows if (r.get("solved") or "0") == "1")
    total = len(rows)
    pct = 100.0 * solved / total if total else 0
    _add_title_block(fig, "Tactics",
                     f"{solved}/{total} solved ({pct:.1f}%)")

    ids = [r.get("id", f"#{i}") for i, r in enumerate(rows)]
    sol_flags = [(r.get("solved") or "0") == "1" for r in rows]
    cps = [_to_int(r.get("score_cp")) for r in rows]
    nodes = [_to_int(r.get("nodes")) or 0 for r in rows]
    expected = [r.get("expected_bm", "") for r in rows]
    engine = [r.get("engine_bestmove", "") for r in rows]

    y_pos = np.arange(len(rows))[::-1]
    colors = [COLOR_GOOD if s else COLOR_BAD for s in sol_flags]

    ax = fig.add_axes([0.22, 0.25, 0.40, 0.65])
    ax.barh(y_pos, [1.0] * len(rows), color=colors, edgecolor="white",
            linewidth=0.7, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("solved (green) / unsolved (red)")
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    ax.tick_params(left=False)

    # right-side annotation table
    table_rows = []
    for i, r in enumerate(rows):
        s = "✔" if sol_flags[i] else "✘"
        cp = "" if cps[i] is None else (
            "mate" if abs(cps[i]) >= 50000 else f"{cps[i]/100:+.2f}"
        )
        # collapse multi-move expected list to its first entry + count suffix
        exp_list = expected[i].split()
        if len(exp_list) > 1:
            exp_disp = f"{exp_list[0]} (+{len(exp_list) - 1})"
        else:
            exp_disp = expected[i]
        table_rows.append([s, exp_disp, engine[i], cp, str(nodes[i])])
    ax_t = fig.add_axes([0.64, 0.25, 0.32, 0.65])
    _draw_kv_table(ax_t, ["", "expected", "engine", "eval", "nodes"],
                   table_rows,
                   col_weights=[0.5, 1.5, 1.3, 1.2, 1.2],
                   row_height=0.95 / max(len(rows) + 1, 8),
                   font_size=8)


def _page_unit_test_failures(fig: Figure, rows: List[Dict[str, str]]) -> None:
    _add_title_block(fig, "Unit test failures",
                     f"{len(rows)} failure(s) / errors")
    table_rows = [
        [r.get("suite", ""), r.get("name", ""), r.get("status", ""),
         _shorten_path(r.get("message", ""), 80)]
        for r in rows
    ]
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.80])
    _draw_kv_table(ax, ["suite", "name", "status", "message"], table_rows,
                   col_weights=[1.5, 2, 1, 4], row_height=0.04, font_size=8,
                   max_rows=20)


def _page_build_info(fig: Figure, build_info: Dict[str, str]) -> None:
    _add_title_block(fig, "Build info", "raw key/value snapshot")
    rows = []
    for k, v in build_info.items():
        rows.append([k, _shorten_path(v, 80)])
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.85])
    _draw_kv_table(ax, ["key", "value"], rows,
                   col_weights=[1, 4], row_height=0.038, font_size=8)


def _shorten_path(s: str, max_len: int = 70) -> str:
    if len(s) <= max_len:
        return s
    head = s[: max_len // 2 - 2]
    tail = s[-(max_len // 2 - 2):]
    return f"{head}…{tail}"


# ---------------------------------------------------------------------------
# COMPARE: per-page renderers
# ---------------------------------------------------------------------------

_SCORE_DELTA_METRICS: List[Tuple[str, str, str, int, bool]] = [
    # (key, label, unit, precision, higher_is_better)
    ("correctness_score", "correctness_score", "", 1, True),
    ("stability_score",   "stability_score",   "", 1, True),
    ("unit_tests_pct",    "unit_tests_pct",    "%", 2, True),
    ("tactics_pct",       "tactics_pct",       "%", 2, True),
    ("backend_peak_nps",  "backend_peak_nps",  "", 0, True),
    ("search_total_nps",  "search_total_nps",  "", 0, True),
]


def _page_cover_compare(fig: Figure,
                        baseline_dir: Path, current_dir: Path,
                        b_scores: Dict[str, str], c_scores: Dict[str, str],
                        b_info: Dict[str, str], c_info: Dict[str, str]) -> None:
    _add_title_block(fig, "lc0 eval — compare report",
                     f"{baseline_dir.name}  →  {current_dir.name}")

    # path / sha block
    rows = [
        ["baseline", _shorten_path(str(baseline_dir), 80)],
        ["  sha", b_info.get("git_short", "")],
        ["  started", b_info.get("started_utc", "")],
        ["current", _shorten_path(str(current_dir), 80)],
        ["  sha", c_info.get("git_short", "")],
        ["  started", c_info.get("started_utc", "")],
    ]
    ax_meta = fig.add_axes([0.06, 0.78, 0.88, 0.13])
    _draw_kv_table(ax_meta, ["", "value"], rows,
                   col_weights=[1, 4], row_height=0.16, font_size=9)

    # delta table
    table_rows: List[List[str]] = []
    bar_data: List[Tuple[str, Optional[float]]] = []
    for key, label, unit, prec, higher_better in _SCORE_DELTA_METRICS:
        bv = _to_float(b_scores.get(key))
        cv = _to_float(c_scores.get(key))
        if bv is None and cv is None:
            continue
        bv_s = "—" if bv is None else f"{bv:.{prec}f}{unit}"
        cv_s = "—" if cv is None else f"{cv:.{prec}f}{unit}"
        delta = None if (bv is None or cv is None) else cv - bv
        pct = None
        if delta is not None and bv:
            pct = 100.0 * delta / bv
        delta_s = _fmt_delta(cv, bv, unit=unit, precision=prec)
        table_rows.append([label, bv_s, cv_s, delta_s])
        bar_data.append((label, pct))

    ax_tbl = fig.add_axes([0.06, 0.43, 0.88, 0.32])
    _draw_kv_table(ax_tbl, ["metric", "baseline", "current", "Δ"], table_rows,
                   col_weights=[2, 1.3, 1.3, 2], row_height=0.085, font_size=10)
    fig.text(0.06, 0.76, "Score deltas", fontsize=12, weight="bold")

    # bar chart of Δ%
    ax_bar = fig.add_axes([0.18, 0.08, 0.68, 0.30])
    labels = [b[0] for b in bar_data if b[1] is not None]
    values = [b[1] for b in bar_data if b[1] is not None]
    if labels:
        y = np.arange(len(labels))[::-1]
        colors = [COLOR_GOOD if v >= 0 else COLOR_BAD for v in values]
        ax_bar.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)
        ax_bar.axvline(0, color="#444", linewidth=0.6)
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(labels, fontsize=9)
        ax_bar.set_xlabel("Δ %  (current vs baseline)")
        ax_bar.set_title("Δ% per metric")
        ax_bar.grid(True, axis="x", linewidth=0.4, alpha=0.4)
        ax_bar.grid(False, axis="y")
        for yi, vi in zip(y, values):
            ax_bar.text(vi + (0.4 if vi >= 0 else -0.4), yi,
                        f"{vi:+.2f}%", va="center",
                        ha="left" if vi >= 0 else "right",
                        fontsize=8, color="#222")
    else:
        ax_bar.axis("off")
        ax_bar.text(0.5, 0.5, "(no comparable metrics)",
                    ha="center", va="center", color="#999")


def _page_compare_backend(fig: Figure,
                          b_rows: List[Dict[str, str]],
                          c_rows: List[Dict[str, str]]) -> None:
    _add_title_block(fig, "Backend bench — overlay",
                     "baseline (grey) vs current (blue)")
    if not (b_rows or c_rows):
        fig.text(0.5, 0.5, "(no data)", ha="center", color="#999")
        return

    def col(rows, key, cast=_to_float):
        return np.array([cast(r.get(key)) or 0 for r in rows])

    b_x, b_y = col(b_rows, "batch_size"), col(b_rows, "mean_nps")
    c_x, c_y = col(c_rows, "batch_size"), col(c_rows, "mean_nps")

    ax1 = fig.add_axes([0.10, 0.55, 0.84, 0.34])
    if len(b_rows):
        ax1.plot(b_x, b_y, color=COLOR_BASELINE, marker="o", markersize=3,
                 linewidth=1.2, label="baseline")
    if len(c_rows):
        ax1.plot(c_x, c_y, color=COLOR_CURRENT, marker="o", markersize=4,
                 linewidth=1.6, label="current")
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("mean NPS")
    ax1.set_title("NPS vs batch size")
    ax1.legend(loc="lower right", fontsize=9, frameon=False)
    ax1.set_ylim(bottom=0)

    # delta % per batch (intersect)
    by_b = {int(r["batch_size"]): _to_float(r["mean_nps"]) for r in b_rows
            if _to_int(r.get("batch_size")) is not None}
    by_c = {int(r["batch_size"]): _to_float(r["mean_nps"]) for r in c_rows
            if _to_int(r.get("batch_size")) is not None}
    common = sorted(set(by_b) & set(by_c))
    deltas = []
    for bs in common:
        bv, cv = by_b[bs], by_c[bs]
        if bv:
            deltas.append((bs, 100.0 * (cv - bv) / bv))
    ax2 = fig.add_axes([0.10, 0.10, 0.84, 0.34])
    if deltas:
        xs = [d[0] for d in deltas]
        ys = [d[1] for d in deltas]
        colors = [COLOR_GOOD if y >= 0 else COLOR_BAD for y in ys]
        ax2.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.6,
                width=max(min(xs) * 0.6, 1) if xs else 1)
        ax2.axhline(0, color="#444", linewidth=0.6)
        ax2.set_xlabel("batch size")
        ax2.set_ylabel("Δ% (current vs baseline)")
        ax2.set_title("Per-batch NPS delta")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "(no overlapping batch sizes)",
                 ha="center", va="center", color="#999")


def _page_compare_search(fig: Figure,
                         b_rows: List[Dict[str, str]],
                         c_rows: List[Dict[str, str]]) -> None:
    _add_title_block(fig, "Search bench — overlay",
                     "per-position NPS, baseline vs current")
    by_b = {int(r["position_idx"]): _to_float(r["nps"]) for r in b_rows
            if _to_int(r.get("position_idx")) is not None}
    by_c = {int(r["position_idx"]): _to_float(r["nps"]) for r in c_rows
            if _to_int(r.get("position_idx")) is not None}
    common = sorted(set(by_b) | set(by_c))
    if not common:
        fig.text(0.5, 0.5, "(no data)", ha="center", color="#999")
        return
    width = 0.4
    x = np.arange(len(common))
    bvals = [by_b.get(p) or 0 for p in common]
    cvals = [by_c.get(p) or 0 for p in common]
    ax = fig.add_axes([0.10, 0.30, 0.84, 0.58])
    ax.bar(x - width / 2, bvals, width=width, color=COLOR_BASELINE,
           label="baseline", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, cvals, width=width, color=COLOR_CURRENT,
           label="current", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in common], fontsize=8)
    ax.set_xlabel("position idx")
    ax.set_ylabel("NPS")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Per-position NPS")

    # totals callout
    sb, sc = sum(bvals), sum(cvals)
    delta = sc - sb
    pct = (100.0 * delta / sb) if sb else 0.0
    sign = "+" if delta >= 0 else ""
    fig.text(0.10, 0.20,
             f"total NPS — baseline {sb:.0f}  →  current {sc:.0f}   "
             f"({sign}{delta:.0f}, {sign}{pct:.2f}%)",
             fontsize=10,
             color=_delta_color(delta, higher_is_better=True))


def _page_compare_tactics(fig: Figure,
                          b_rows: List[Dict[str, str]],
                          c_rows: List[Dict[str, str]]) -> None:
    _add_title_block(fig, "Tactics — flips",
                     "positions whose solved status changed")
    by_b = {r["id"]: r for r in b_rows}
    by_c = {r["id"]: r for r in c_rows}
    flips = []
    for k in sorted(set(by_b) | set(by_c)):
        bs = (by_b.get(k, {}).get("solved") or "0") == "1"
        cs = (by_c.get(k, {}).get("solved") or "0") == "1"
        if bs == cs:
            continue
        flips.append([
            k,
            "✔" if bs else "✘",
            "✔" if cs else "✘",
            by_b.get(k, {}).get("engine_bestmove", ""),
            by_c.get(k, {}).get("engine_bestmove", ""),
            "improved" if (cs and not bs) else "regressed",
        ])
    if not flips:
        b_solved = sum(1 for r in b_rows if (r.get("solved") or "0") == "1")
        c_solved = sum(1 for r in c_rows if (r.get("solved") or "0") == "1")
        fig.text(0.5, 0.55,
                 f"no solved-status changes\nbaseline {b_solved}/{len(b_rows)} → "
                 f"current {c_solved}/{len(c_rows)}",
                 ha="center", va="center", fontsize=12, color="#444")
        return
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.80])
    _draw_kv_table(ax, ["id", "base", "cur", "engine (base)", "engine (cur)", "change"],
                   flips,
                   col_weights=[2, 0.6, 0.6, 1.4, 1.4, 1.2],
                   row_height=0.06, font_size=9)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _read_scores(report_dir: Path) -> Dict[str, str]:
    rows = _read_csv(report_dir / "scores.csv")
    return rows[0] if rows else {}


def _movetime_from_summary(report_dir: Path) -> Optional[int]:
    """Best-effort extraction of search-bench movetime from logs/summary."""
    log = report_dir / "logs" / "search_bench.log"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# command:") and "--movetime=" in line:
                for tok in line.split():
                    if tok.startswith("--movetime="):
                        return _to_int(tok.split("=", 1)[1])
    return None


# ---------------------------------------------------------------------------
# public API
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
                   if (r.get("status") or "").lower() in ("failed", "errored", "error")]
    build_info = _read_build_info_md(report_dir / "build_info.md")
    movetime = _movetime_from_summary(report_dir)

    # decide pages
    pages: List[str] = ["cover"]
    if backend_rows:
        pages.append("backend")
    if search_rows:
        pages.append("search")
    if tactics_rows:
        pages.append("tactics")
    if failed_unit:
        pages.append("unit_failures")
    if build_info:
        pages.append("build_info")
    total = len(pages)

    sha_short = build_info.get("git_short", "") or build_info.get("git_sha", "")[:7]
    started = build_info.get("started_utc", "")

    out = report_dir / "report.pdf"
    with PdfPages(str(out)) as pdf:
        for i, kind in enumerate(pages, start=1):
            fig = _new_page()
            if kind == "cover":
                _page_cover_run(fig, scores, build_info)
            elif kind == "backend":
                _page_backend_bench(fig, backend_rows,
                                    build_info.get("net_path", ""))
            elif kind == "search":
                _page_search_bench(fig, search_rows, movetime)
            elif kind == "tactics":
                _page_tactics(fig, tactics_rows)
            elif kind == "unit_failures":
                _page_unit_test_failures(fig, failed_unit)
            elif kind == "build_info":
                _page_build_info(fig, build_info)
            _add_footer(fig, i, total, sha=sha_short, ts=started,
                        left_label="lc0 eval")
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = f"lc0 eval report — {report_dir.name}"
        d["Author"] = "scripts/eval/eval.py"
        d["Subject"] = "lc0 evaluation report"
        d["Keywords"] = "lc0 chess eval"
        d["CreationDate"] = datetime.now(timezone.utc)

    return out


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

    pages = ["cover"]
    if b_backend or c_backend:
        pages.append("backend")
    if b_search or c_search:
        pages.append("search")
    if b_tactics or c_tactics:
        pages.append("tactics")
    total = len(pages)

    sha_short = c_info.get("git_short", "")
    started = c_info.get("started_utc", "")

    out = current_dir / "compare.pdf"
    with PdfPages(str(out)) as pdf:
        for i, kind in enumerate(pages, start=1):
            fig = _new_page()
            if kind == "cover":
                _page_cover_compare(fig, baseline_dir, current_dir,
                                    b_scores, c_scores, b_info, c_info)
            elif kind == "backend":
                _page_compare_backend(fig, b_backend, c_backend)
            elif kind == "search":
                _page_compare_search(fig, b_search, c_search)
            elif kind == "tactics":
                _page_compare_tactics(fig, b_tactics, c_tactics)
            _add_footer(fig, i, total, sha=sha_short, ts=started,
                        left_label="lc0 eval compare")
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = (f"lc0 eval compare — {baseline_dir.name} vs "
                      f"{current_dir.name}")
        d["Author"] = "scripts/eval/eval.py"
        d["Subject"] = "lc0 evaluation comparison"
        d["CreationDate"] = datetime.now(timezone.utc)

    return out
