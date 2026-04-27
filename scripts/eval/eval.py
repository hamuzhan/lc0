#!/usr/bin/env python3
"""scripts/eval/eval.py — orchestrate lc0 evaluation runs and collect tabular reports.

Subcommands:
  run      — run unit tests + backendbench + bench/benchmark + tactics, write CSVs.
  compare  — diff two prior report directories.
  latest   — print the path of the most recent report directory.

The orchestrator only reads existing tool stdout/stderr; no C++ changes required.
See scripts/eval/README.md for column definitions and CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = REPO_ROOT / "eval-reports"
DEFAULT_EPD = Path(__file__).resolve().parent / "positions" / "bench.epd"

PHASE_UNIT_TESTS = "unit_tests"
PHASE_BACKEND_BENCH = "backend_bench"
PHASE_SEARCH_BENCH = "search_bench"
PHASE_TACTICS = "tactics"
ALL_PHASES = (PHASE_UNIT_TESTS, PHASE_BACKEND_BENCH, PHASE_SEARCH_BENCH, PHASE_TACTICS)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def run_capture(cmd: List[str], cwd: Optional[Path] = None,
                timeout: Optional[float] = None,
                env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, timeout=timeout,
                          env=env, check=False)


def git(*args: str) -> str:
    cp = run_capture(["git"] + list(args), cwd=REPO_ROOT)
    return cp.stdout.strip() if cp.returncode == 0 else ""


def short_sha() -> str:
    return git("rev-parse", "--short", "HEAD") or "nogit"


def git_dirty() -> bool:
    cp = run_capture(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    return bool(cp.stdout.strip())


def file_sha256_prefix(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def write_log(path: Path, stdout: str, stderr: str, cmd: List[str], rc: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# command: " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        f.write(f"# exit_code: {rc}\n")
        f.write("# ===== stdout =====\n")
        f.write(stdout or "")
        if stderr:
            f.write("\n# ===== stderr =====\n")
            f.write(stderr)


def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in header})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# binary / net discovery
# ---------------------------------------------------------------------------

def resolve_lc0(arg: Optional[str]) -> Path:
    candidates: List[Path] = []
    if arg:
        candidates.append(Path(arg).expanduser().resolve())
    candidates.extend([
        REPO_ROOT / "build" / "release" / "lc0",
        REPO_ROOT / "build" / "debugoptimized" / "lc0",
        REPO_ROOT / "builddir" / "lc0",
        REPO_ROOT / "build" / "lc0",
    ])
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "lc0 binary not found. Pass --lc0-path PATH, build with "
        "`./build.sh release`, or use --build to invoke build.sh from this script."
    )


def resolve_builddir(lc0: Optional[Path], explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if (p / "build.ninja").is_file() else None
    if lc0 is not None and (lc0.parent / "build.ninja").is_file():
        return lc0.parent
    for c in (REPO_ROOT / "build" / "release", REPO_ROOT / "builddir"):
        if (c / "build.ninja").is_file():
            return c
    return None


def resolve_net(arg: Optional[str]) -> Optional[Path]:
    if arg:
        p = Path(arg).expanduser().resolve()
        return p if p.is_file() else None
    for d in (REPO_ROOT / "weights", REPO_ROOT):
        if d.is_dir():
            for pattern in ("*.pb.gz", "*.pb", "*.onnx"):
                hits = sorted(d.glob(pattern))
                if hits:
                    return hits[0]
    return None


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------

_BACKENDBENCH_HEADER_RE = re.compile(r"^\s*size\s*,")


def parse_backendbench_stdout(text: str) -> List[Dict[str, Any]]:
    """Parse the CSV-with-padding output from `lc0 backendbench`.

    Header line (reprinted every 32 batches by default, or once with
    --header-only-once): "size, mean nps, mean ms, sdev, cv, max nps, median,
    min nps, first max, first mean".
    """
    rows: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        if _BACKENDBENCH_HEADER_RE.match(raw):
            continue
        s = raw.strip()
        if not s or "," not in s:
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 10:
            continue
        try:
            row = {
                "batch_size": int(parts[0]),
                "mean_nps": int(float(parts[1])),
                "mean_ms": float(parts[2]),
                "sdev_ms": float(parts[3]),
                "cv": float(parts[4]),
                "max_nps": int(float(parts[5])),
                "median_nps": int(float(parts[6])),
                "min_nps": int(float(parts[7])),
                "first_max_ms": float(parts[8]),
                "first_mean_ms": float(parts[9]),
            }
        except ValueError:
            continue
        rows.append(row)
    return rows


_POSITION_RE = re.compile(r"^\s*Position:\s*(\d+)/(\d+)\s+(.+?)\s*$")
_BENCHTIME_RE = re.compile(
    r"^\s*Benchmark time\s+(\d+)\s+ms,\s*(\d+)\s+nodes,\s*(\d+)\s+nps"
    r"(?:,\s*move\s+(\S+))?\s*$"
)
_BESTMOVE_RE = re.compile(r"^\s*bestmove\s+(\S+)")


def parse_benchmark_stdout(text: str) -> List[Dict[str, Any]]:
    """Parse `lc0 bench` / `lc0 benchmark` stdout into per-position rows.

    Each position emits one or more `Benchmark time ... ms, ... nodes, ... nps`
    lines (final line is the cumulative result for that position) followed by a
    `bestmove ...` line. We capture the last benchmark line + bestmove.
    """
    rows: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for raw in text.splitlines():
        m = _POSITION_RE.match(raw)
        if m:
            if cur is not None:
                rows.append(cur)
            cur = {
                "position_idx": int(m.group(1)),
                "fen": m.group(3).strip(),
                "time_ms": None, "nodes": None, "nps": None, "bestmove": None,
            }
            continue
        if cur is None:
            continue
        m = _BENCHTIME_RE.match(raw)
        if m:
            cur["time_ms"] = int(m.group(1))
            cur["nodes"] = int(m.group(2))
            cur["nps"] = int(m.group(3))
            continue
        m = _BESTMOVE_RE.match(raw)
        if m:
            cur["bestmove"] = m.group(1)
            rows.append(cur)
            cur = None
    if cur is not None:
        rows.append(cur)
    return rows


def parse_junit_xml(path: Path) -> List[Dict[str, Any]]:
    """Walk a JUnit-style XML produced by gtest --gtest_output=xml:..."""
    rows: List[Dict[str, Any]] = []
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return rows
    root = tree.getroot()
    # Either <testsuites> wrapping <testsuite>s or a single <testsuite>.
    testsuites = root.findall(".//testsuite") or [root]
    for ts in testsuites:
        suite = ts.attrib.get("name", "?")
        for tc in ts.findall("testcase"):
            name = tc.attrib.get("name", "?")
            classname = tc.attrib.get("classname", "")
            full = f"{classname}.{name}" if classname and classname != suite else name
            time_s = float(tc.attrib.get("time", "0") or 0)
            status = "passed"
            message = ""
            if tc.find("failure") is not None:
                status = "failed"
                fnode = tc.find("failure")
                message = (fnode.attrib.get("message", "") or "").splitlines()[0][:200]
            elif tc.find("error") is not None:
                status = "error"
                enode = tc.find("error")
                message = (enode.attrib.get("message", "") or "").splitlines()[0][:200]
            elif tc.find("skipped") is not None or tc.attrib.get("status") == "notrun":
                status = "skipped"
            rows.append({
                "suite": suite,
                "name": full,
                "status": status,
                "time_ms": round(time_s * 1000.0, 3),
                "message": message,
            })
    return rows


# ---------------------------------------------------------------------------
# EPD parsing
# ---------------------------------------------------------------------------

def parse_epd_file(path: Path) -> List[Dict[str, Any]]:
    """Parse a small EPD-like file. Convention: `bm`/`am` operands are UCI
    long-algebraic moves (e.g. `e2e4`) so we don't need a SAN→UCI converter.
    Lines starting with '#' or empty are ignored.
    """
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # FEN: piece-placement stm castling ep — 4 tokens, then operations.
        toks = line.split(" ", 4)
        if len(toks) < 5:
            continue
        fen = " ".join(toks[:4]) + " 0 1"
        ops = _parse_epd_ops(toks[4])
        out.append({
            "fen": fen,
            "bm": ops.get("bm", []),
            "am": ops.get("am", []),
            "id": (ops.get("id") or [f"pos{len(out)+1}"])[0],
        })
    return out


def _parse_epd_ops(s: str) -> Dict[str, List[str]]:
    ops: Dict[str, List[str]] = {}
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        bits = chunk.split(None, 1)
        opcode = bits[0]
        operand = bits[1].strip() if len(bits) > 1 else ""
        if opcode == "id":
            m = re.match(r'^"(.*)"$', operand)
            ops.setdefault(opcode, []).append(m.group(1) if m else operand)
        elif opcode in ("bm", "am"):
            ops.setdefault(opcode, []).extend(operand.split())
        else:
            ops.setdefault(opcode, []).append(operand)
    return ops


# ---------------------------------------------------------------------------
# UCI subprocess driver (used by tactics phase)
# ---------------------------------------------------------------------------

class UciEngine:
    def __init__(self, cmd: List[str], log_path: Path,
                 startup_timeout: float = 30.0):
        import queue
        import threading

        self._cmd = cmd
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self._log_path, "w")
        self._log.write("# command: " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        self._log.flush()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        # Background reader: turns blocking pipe readlines into a queue we can
        # poll with a timeout. TextIOWrapper buffering breaks selectors-based
        # approaches, so just spawn a thread.
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._wait_for("uciok", send="uci", timeout=startup_timeout)

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self._q.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            self._q.put(None)  # EOF sentinel

    def _send(self, line: str) -> None:
        self._log.write(f"> {line}\n")
        self._log.flush()
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _readline(self, timeout: float) -> Optional[str]:
        import queue
        try:
            line = self._q.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return None
        if line is None:
            return None
        self._log.write(f"< {line}\n")
        self._log.flush()
        return line

    def _wait_for(self, marker: str, send: Optional[str] = None,
                  timeout: float = 30.0) -> List[str]:
        if send is not None:
            self._send(send)
        captured: List[str] = []
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            line = self._readline(end - time.monotonic())
            if line is None:
                break
            captured.append(line)
            if line.strip() == marker or line.strip().startswith(marker + " "):
                return captured
        raise TimeoutError(f"timed out waiting for '{marker}'")

    def isready(self, timeout: float = 30.0) -> None:
        self._wait_for("readyok", send="isready", timeout=timeout)

    def setoption(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")

    def newgame(self, timeout: float = 30.0) -> None:
        self._send("ucinewgame")
        self.isready(timeout)

    def position_fen(self, fen: str) -> None:
        self._send(f"position fen {fen}")

    def go_movetime(self, ms: int, timeout: float) -> Tuple[Optional[str], Dict[str, Any]]:
        self._send(f"go movetime {ms}")
        last_info: Dict[str, Any] = {}
        bestmove: Optional[str] = None
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            line = self._readline(end - time.monotonic())
            if line is None:
                break
            if line.startswith("info "):
                _absorb_info(line, last_info)
            elif line.startswith("bestmove"):
                m = _BESTMOVE_RE.match(line)
                if m:
                    bestmove = m.group(1)
                break
        return bestmove, last_info

    def quit(self, timeout: float = 5.0) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=timeout)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            try:
                self._log.close()
            except Exception:
                pass


def _absorb_info(line: str, last: Dict[str, Any]) -> None:
    toks = line.split()
    i = 1
    while i < len(toks):
        k = toks[i]
        if k == "depth" and i + 1 < len(toks):
            last["depth"] = _safe_int(toks[i + 1]); i += 2
        elif k == "nodes" and i + 1 < len(toks):
            last["nodes"] = _safe_int(toks[i + 1]); i += 2
        elif k == "time" and i + 1 < len(toks):
            last["time_ms"] = _safe_int(toks[i + 1]); i += 2
        elif k == "nps" and i + 1 < len(toks):
            last["nps"] = _safe_int(toks[i + 1]); i += 2
        elif k == "score" and i + 2 < len(toks):
            kind, val = toks[i + 1], toks[i + 2]
            if kind == "cp":
                last["score_cp"] = _safe_int(val)
            elif kind == "mate":
                m = _safe_int(val)
                last["score_cp"] = (100000 if (m or 0) > 0 else -100000)
            i += 3
        else:
            i += 1


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def compute_scores(phases_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce phase summaries to a single set of headline metrics.

    correctness_score (0-100): mean of per-phase correctness rates that were
    actually measured (unit-test pass-rate, tactics solved-rate). Phases that
    failed or weren't run are excluded from the average — a partial run still
    yields a meaningful number.

    stability_score (0-100): fraction of non-skipped phases that completed OK.

    Performance metrics are kept as raw numbers (no normalization to a magic
    "100"); hardware-dependent absolute NPS is only meaningful relative to a
    baseline, which `compare` handles separately.
    """
    s: Dict[str, Any] = {
        "correctness_score": None,
        "stability_score": None,
        "unit_tests_pct": None,
        "unit_tests_passed": None,
        "unit_tests_total": None,
        "tactics_pct": None,
        "tactics_solved": None,
        "tactics_total": None,
        "backend_peak_nps": None,
        "backend_peak_batch": None,
        "search_total_nps": None,
        "search_total_nodes": None,
        "phases_ok": 0,
        "phases_failed": 0,
        "phases_skipped": 0,
        "phases_total": 0,
    }

    correctness_components: List[float] = []

    ut = phases_summary.get(PHASE_UNIT_TESTS, {})
    if ut.get("status") == "OK" and ut.get("total"):
        total = int(ut.get("total") or 0)
        passed = int(ut.get("passed") or 0)
        s["unit_tests_total"] = total
        s["unit_tests_passed"] = passed
        if total > 0:
            pct = 100.0 * passed / total
            s["unit_tests_pct"] = round(pct, 2)
            correctness_components.append(pct)

    tac = phases_summary.get(PHASE_TACTICS, {})
    if tac.get("status") == "OK" and tac.get("positions"):
        total = int(tac.get("positions") or 0)
        solved = int(tac.get("solved") or 0)
        s["tactics_total"] = total
        s["tactics_solved"] = solved
        if total > 0:
            pct = 100.0 * solved / total
            s["tactics_pct"] = round(pct, 2)
            correctness_components.append(pct)

    bb = phases_summary.get(PHASE_BACKEND_BENCH, {})
    if bb.get("status") == "OK":
        s["backend_peak_nps"] = bb.get("peak_nps")
        s["backend_peak_batch"] = bb.get("peak_batch")

    sb = phases_summary.get(PHASE_SEARCH_BENCH, {})
    if sb.get("status") == "OK":
        s["search_total_nps"] = sb.get("total_nps")
        s["search_total_nodes"] = sb.get("total_nodes")

    if correctness_components:
        s["correctness_score"] = round(
            sum(correctness_components) / len(correctness_components), 2
        )

    for info in phases_summary.values():
        st = info.get("status")
        s["phases_total"] += 1
        if st == "OK":
            s["phases_ok"] += 1
        elif st == "FAILED":
            s["phases_failed"] += 1
        elif st == "SKIPPED":
            s["phases_skipped"] += 1

    if s["phases_total"]:
        ran = s["phases_total"] - s["phases_skipped"]
        if ran:
            s["stability_score"] = round(100.0 * s["phases_ok"] / ran, 2)

    return s


_SCORES_CSV_COLS = [
    "correctness_score", "stability_score",
    "unit_tests_passed", "unit_tests_total", "unit_tests_pct",
    "tactics_solved", "tactics_total", "tactics_pct",
    "backend_peak_nps", "backend_peak_batch",
    "search_total_nps", "search_total_nodes",
    "phases_ok", "phases_failed", "phases_skipped", "phases_total",
]


def write_scores_csv(report_dir: Path, scores: Dict[str, Any]) -> None:
    write_csv(report_dir / "scores.csv", _SCORES_CSV_COLS, [scores])


def _fmt_rate(numerator: Optional[int], denominator: Optional[int],
              pct: Optional[float]) -> str:
    if denominator is None:
        return "_n/a_"
    if pct is None:
        return f"{numerator}/{denominator}"
    return f"{numerator}/{denominator} ({pct:.1f}%)"


def _fmt_peak(peak_nps: Optional[int], peak_batch: Optional[int]) -> str:
    if peak_nps is None:
        return "_n/a_"
    if peak_batch is None:
        return str(peak_nps)
    return f"{peak_nps} @ batch {peak_batch}"


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def _common_lc0_args(net: Optional[Path], args: argparse.Namespace) -> List[str]:
    extra: List[str] = []
    if net:
        extra.append(f"--weights={net}")
    if args.backend:
        extra.append(f"--backend={args.backend}")
    if args.backend_opts:
        extra.append(f"--backend-opts={args.backend_opts}")
    return extra


def run_unit_tests(report_dir: Path, builddir: Optional[Path],
                   logs: Path, run_ninja: bool) -> Dict[str, Any]:
    if builddir is None:
        raise RuntimeError(
            "no meson build directory found; pass --builddir or build first"
        )
    if run_ninja:
        cmd = ["ninja", "-C", str(builddir), "test"]
        cp = run_capture(cmd, timeout=600)
        write_log(logs / "unit_tests.ninja.log", cp.stdout, cp.stderr, cmd, cp.returncode)
        if cp.returncode != 0 and not any(builddir.glob("*.xml")):
            raise RuntimeError(f"`ninja test` failed (rc={cp.returncode}); see logs/unit_tests.ninja.log")

    rows: List[Dict[str, Any]] = []
    for xml in sorted(builddir.glob("*.xml")):
        rows.extend(parse_junit_xml(xml))
    write_csv(report_dir / "unit_tests.csv",
              ["suite", "name", "status", "time_ms", "message"], rows)

    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "total": len(rows),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "errored": counts.get("error", 0),
    }


def run_backend_bench(report_dir: Path, lc0: Path, net: Optional[Path],
                      args: argparse.Namespace, logs: Path,
                      quick: bool) -> Dict[str, Any]:
    cmd = [str(lc0), "backendbench", "--header-only-once=true"]
    if quick:
        cmd += ["--start-batch-size=1", "--max-batch-size=64",
                "--batch-step=8", "--batches=20"]
    else:
        cmd += ["--start-batch-size=1", "--max-batch-size=256",
                "--batch-step=16", "--batches=100"]
    cmd += _common_lc0_args(net, args)
    cp = run_capture(cmd, timeout=1800)
    write_log(logs / "backend_bench.log", cp.stdout, cp.stderr, cmd, cp.returncode)
    rows = parse_backendbench_stdout(cp.stdout)
    if cp.returncode != 0 and not rows:
        raise RuntimeError(
            f"backendbench failed (rc={cp.returncode}); see logs/backend_bench.log"
        )
    if not rows:
        raise RuntimeError(
            "backendbench produced no parseable rows; see logs/backend_bench.log"
        )
    write_csv(report_dir / "backend_bench.csv",
              ["batch_size", "mean_nps", "mean_ms", "sdev_ms", "cv",
               "max_nps", "median_nps", "min_nps",
               "first_max_ms", "first_mean_ms"], rows)
    peak = max(rows, key=lambda r: r["mean_nps"])
    return {
        "rows": len(rows),
        "peak_nps": peak["mean_nps"],
        "peak_batch": peak["batch_size"],
        "min_batch": min(r["batch_size"] for r in rows),
        "max_batch": max(r["batch_size"] for r in rows),
    }


def run_search_bench(report_dir: Path, lc0: Path, net: Optional[Path],
                     args: argparse.Namespace, logs: Path,
                     quick: bool) -> Dict[str, Any]:
    mode = "bench" if quick else "benchmark"
    cmd = [str(lc0), mode]
    if quick:
        cmd += ["--movetime=500", "--num-positions=10"]
    else:
        cmd += ["--movetime=5000", "--num-positions=34"]
    cmd += _common_lc0_args(net, args)
    cp = run_capture(cmd, timeout=3600)
    write_log(logs / "search_bench.log", cp.stdout, cp.stderr, cmd, cp.returncode)
    rows = parse_benchmark_stdout(cp.stdout)
    if cp.returncode != 0 and not rows:
        raise RuntimeError(
            f"`lc0 {mode}` failed (rc={cp.returncode}); see logs/search_bench.log"
        )
    if not rows:
        raise RuntimeError(
            f"`lc0 {mode}` produced no parseable rows; see logs/search_bench.log"
        )
    write_csv(report_dir / "search_bench.csv",
              ["position_idx", "fen", "time_ms", "nodes", "nps", "bestmove"], rows)
    total_nodes = sum((r["nodes"] or 0) for r in rows)
    total_time = sum((r["time_ms"] or 0) for r in rows)
    return {
        "positions": len(rows),
        "total_nodes": total_nodes,
        "total_time_ms": total_time,
        "total_nps": int(round(1000.0 * total_nodes / max(total_time, 1))),
    }


def run_tactics(report_dir: Path, lc0: Path, net: Optional[Path],
                args: argparse.Namespace, logs: Path, epd_path: Path,
                movetime_ms: int, max_positions: Optional[int]) -> Dict[str, Any]:
    positions = parse_epd_file(epd_path)
    if not positions:
        raise RuntimeError(f"no positions in {epd_path}")
    if max_positions is not None:
        positions = positions[:max_positions]

    cmd = [str(lc0)]
    cmd += _common_lc0_args(net, args)
    eng = UciEngine(cmd, logs / "tactics.uci.log", startup_timeout=60.0)
    try:
        eng.isready(timeout=60.0)
        rows: List[Dict[str, Any]] = []
        per_position_timeout = max(60.0, movetime_ms / 1000.0 * 4 + 30.0)
        for pos in positions:
            try:
                eng.newgame(timeout=30.0)
                eng.position_fen(pos["fen"])
                bestmove, last_info = eng.go_movetime(movetime_ms, per_position_timeout)
            except TimeoutError as e:
                bestmove, last_info = None, {"error": str(e)}
            solved = _check_tactics_match(bestmove, pos.get("bm", []), pos.get("am", []))
            rows.append({
                "id": pos["id"],
                "fen": pos["fen"],
                "expected_bm": " ".join(pos.get("bm", [])),
                "expected_am": " ".join(pos.get("am", [])),
                "engine_bestmove": bestmove or "",
                "score_cp": last_info.get("score_cp"),
                "depth": last_info.get("depth"),
                "nodes": last_info.get("nodes"),
                "time_ms": last_info.get("time_ms"),
                "solved": int(bool(solved)),
            })
    finally:
        eng.quit()

    write_csv(report_dir / "tactics.csv",
              ["id", "fen", "expected_bm", "expected_am", "engine_bestmove",
               "score_cp", "depth", "nodes", "time_ms", "solved"], rows)
    solved = sum(r["solved"] for r in rows)
    return {
        "positions": len(rows),
        "solved": solved,
        "solved_pct": round(100.0 * solved / max(len(rows), 1), 2),
        "movetime_ms": movetime_ms,
    }


def _check_tactics_match(bestmove: Optional[str],
                         expected_bm: List[str],
                         expected_am: List[str]) -> bool:
    if not bestmove:
        return False
    bm = bestmove.lower()
    expected_bm = [m.lower() for m in expected_bm]
    expected_am = [m.lower() for m in expected_am]
    if expected_am and bm in expected_am:
        return False
    if expected_bm:
        return bm in expected_bm
    # If only `am` was specified and the engine avoided it, that's a pass.
    return bool(expected_am)


# ---------------------------------------------------------------------------
# build_info.md / summary.md
# ---------------------------------------------------------------------------

def write_build_info(report_dir: Path, lc0: Optional[Path],
                     net: Optional[Path], args: argparse.Namespace,
                     started_utc: str, finished_utc: str, total_seconds: float,
                     phases_summary: Dict[str, Dict[str, Any]],
                     builddir: Optional[Path]) -> None:
    info: Dict[str, Any] = {
        "git_sha": git("rev-parse", "HEAD"),
        "git_short": short_sha(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": git_dirty(),
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "total_seconds": round(total_seconds, 2),
        "lc0_path": str(lc0) if lc0 else "",
        "lc0_sha256_prefix": (file_sha256_prefix(lc0) if lc0 and lc0.is_file() else ""),
        "net_path": str(net) if net else "",
        "net_sha256_prefix": (file_sha256_prefix(net) if net and net.is_file() else ""),
        "builddir": str(builddir) if builddir else "",
        "backend": args.backend or "",
        "backend_opts": args.backend_opts or "",
        "quick": bool(args.quick),
    }
    # gpu list (best-effort)
    gpus: List[str] = []
    try:
        cp = run_capture(["nvidia-smi", "-L"], timeout=10)
        if cp.returncode == 0:
            gpus = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    info["gpu"] = gpus
    # lc0 version banner (printed to stderr, so capture stderr of `lc0 --help`)
    lc0_version = ""
    if lc0 and lc0.is_file():
        try:
            cp = run_capture([str(lc0), "--help"], timeout=15)
            for ln in (cp.stdout + "\n" + cp.stderr).splitlines():
                m = re.search(r"v(\d+\.\d+\.\d+\S*)", ln)
                if m:
                    lc0_version = m.group(1)
                    break
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    info["lc0_version"] = lc0_version

    # build options snapshot
    build_options = ""
    parsed_options: List[Dict[str, Any]] = []
    if builddir is not None:
        opts_file = builddir / "meson-info" / "intro-buildoptions.json"
        if opts_file.is_file():
            try:
                build_options = opts_file.read_text()
                parsed_options = json.loads(build_options)
            except (OSError, json.JSONDecodeError):
                parsed_options = []
    info["build_options_snapshot"] = bool(build_options)

    relevant_options = _filter_lc0_build_options(parsed_options)
    if relevant_options:
        write_csv(report_dir / "build_options.csv",
                  ["name", "value", "type", "section", "description"],
                  relevant_options)

    md_lines = [
        "# build_info",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for k in ("git_sha", "git_short", "git_branch", "dirty", "host", "os",
              "cpu", "python", "started_utc", "finished_utc", "total_seconds",
              "lc0_path", "lc0_sha256_prefix", "lc0_version", "net_path",
              "net_sha256_prefix", "builddir", "backend", "backend_opts",
              "quick"):
        v = info.get(k, "")
        md_lines.append(f"| {k} | `{v}` |")
    md_lines.append(f"| gpu | {('<br>'.join(f'`{g}`' for g in gpus)) or '_none_'} |")
    md_lines.append("")
    if relevant_options:
        md_lines.append("## build configuration")
        md_lines.append("")
        md_lines.append("Relevant lc0 + meson flags from "
                        "`meson-info/intro-buildoptions.json` "
                        "(full dump preserved below).")
        md_lines.append("")
        md_lines.append("| flag | value |")
        md_lines.append("| --- | --- |")
        for opt in relevant_options:
            v = opt["value"]
            md_lines.append(f"| `{opt['name']}` | `{v}` |")
        md_lines.append("")
    if build_options:
        md_lines.append("## meson build options")
        md_lines.append("")
        md_lines.append("<details><summary>intro-buildoptions.json</summary>")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(build_options.strip())
        md_lines.append("```")
        md_lines.append("</details>")
        md_lines.append("")
    (report_dir / "build_info.md").write_text("\n".join(md_lines))


# Subset of meson options worth surfacing in the report. Order matters:
# this is the order they appear in the report tables.
_LC0_OPTION_FILTER: Tuple[str, ...] = (
    # build basics
    "buildtype", "optimization", "b_lto", "b_ndebug",
    # CUDA family
    "plain_cuda", "cudnn", "cutlass",
    "native_cuda", "cc_cuda", "nvcc_ccbin",
    # other backend families
    "onnx", "xla", "sycl", "dx", "metal",
    # binaries / behavior
    "default_backend", "default_search",
    "dag_classic", "rescorer", "python_bindings", "trace_library",
    "gtest", "lc0", "build_backends",
)


def _filter_lc0_build_options(options: List[Dict[str, Any]]
                              ) -> List[Dict[str, Any]]:
    """Reduce the full intro-buildoptions list to the lc0-relevant subset
    in stable display order. Each output row is a dict with name/value/
    type/section/description as plain strings."""
    by_name = {o.get("name"): o for o in options if isinstance(o, dict)}
    out: List[Dict[str, Any]] = []
    for name in _LC0_OPTION_FILTER:
        if name not in by_name:
            continue
        o = by_name[name]
        v = o.get("value")
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        elif isinstance(v, bool):
            v = "True" if v else "False"
        else:
            v = "" if v is None else str(v)
        out.append({
            "name": name,
            "value": v,
            "type": o.get("type", ""),
            "section": o.get("section", ""),
            "description": (o.get("description") or "").strip(),
        })
    return out


def write_summary(report_dir: Path,
                  phases_summary: Dict[str, Dict[str, Any]],
                  started_utc: str, finished_utc: str,
                  total_seconds: float, lc0: Optional[Path],
                  net: Optional[Path], args: argparse.Namespace,
                  scores: Optional[Dict[str, Any]] = None) -> None:
    lines = [
        "# eval summary",
        "",
        f"- started:  `{started_utc}`",
        f"- finished: `{finished_utc}`",
        f"- duration: `{total_seconds:.1f}s`",
        f"- lc0:      `{lc0}`",
        f"- net:      `{net or '(autodiscover)'}`",
        f"- backend:  `{args.backend or '(default)'}`",
        f"- quick:    `{args.quick}`",
        "",
    ]

    # build flags compact view (shown as one-liner if available)
    build_opts_csv = report_dir / "build_options.csv"
    if build_opts_csv.is_file():
        rows = read_csv(build_opts_csv)
        flagged = [(r["name"], r["value"]) for r in rows]
        if flagged:
            lines += ["## build flags", ""]
            lines += ["| flag | value |", "| --- | --- |"]
            for name, value in flagged:
                lines.append(f"| `{name}` | `{value}` |")
            lines += [""]

    if scores is not None:
        lines += [
            "## scores",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| **correctness_score** | "
            f"{('%.1f / 100' % scores['correctness_score']) if scores['correctness_score'] is not None else '_n/a_'} |",
            f"| **stability_score**   | "
            f"{('%.1f / 100' % scores['stability_score']) if scores['stability_score'] is not None else '_n/a_'} |",
            f"| unit tests passed     | "
            f"{_fmt_rate(scores['unit_tests_passed'], scores['unit_tests_total'], scores['unit_tests_pct'])} |",
            f"| tactics solved        | "
            f"{_fmt_rate(scores['tactics_solved'], scores['tactics_total'], scores['tactics_pct'])} |",
            f"| backend peak NPS      | "
            f"{_fmt_peak(scores['backend_peak_nps'], scores['backend_peak_batch'])} |",
            f"| search total NPS      | "
            f"{scores['search_total_nps'] if scores['search_total_nps'] is not None else '_n/a_'} |",
            f"| phases ok / total     | "
            f"{scores['phases_ok']}/{scores['phases_total']}"
            f"{(' (skipped %d)' % scores['phases_skipped']) if scores['phases_skipped'] else ''} |",
            "",
        ]

    lines += [
        "## phases",
        "",
        "| phase | status | duration_s | summary |",
        "| --- | --- | --- | --- |",
    ]
    for phase in ALL_PHASES:
        info = phases_summary.get(phase, {"status": "MISSING"})
        status = info.get("status", "?")
        dur = info.get("duration_s", "")
        if isinstance(dur, (int, float)):
            dur = f"{dur:.1f}"
        summary_bits: List[str] = []
        for k, v in info.items():
            if k in ("status", "duration_s", "error"):
                continue
            summary_bits.append(f"{k}={v}")
        if status == "FAILED" and info.get("error"):
            summary_bits.append(f"error={info['error']}")
        lines.append(f"| {phase} | {status} | {dur} | {'; '.join(summary_bits)} |")

    files_section = [
        "",
        "## files",
        "",
        "- [build_info.md](build_info.md)",
        "- [scores.csv](scores.csv)",
    ]
    if build_opts_csv.is_file():
        files_section.append("- [build_options.csv](build_options.csv)")
    lines += files_section
    lines += [
        "- [unit_tests.csv](unit_tests.csv)",
        "- [backend_bench.csv](backend_bench.csv)",
        "- [search_bench.csv](search_bench.csv)",
        "- [tactics.csv](tactics.csv)",
        "- `logs/` — raw stdout/stderr per phase",
        "",
    ]
    failed_phases = [p for p, v in phases_summary.items() if v.get("status") == "FAILED"]
    if failed_phases:
        lines += [
            "## failed phases",
            "",
            *(f"- **{p}**: {phases_summary[p].get('error','?')}" for p in failed_phases),
            "",
        ]
    (report_dir / "summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    started_t = time.time()
    started_utc = utc_iso()

    if args.build:
        cmd = [str(REPO_ROOT / "build.sh"), "release"]
        print(f"[eval] building: {' '.join(cmd)}", flush=True)
        cp = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if cp.returncode != 0:
            print("[eval] build failed", file=sys.stderr)
            return cp.returncode

    try:
        lc0 = resolve_lc0(args.lc0_path)
    except FileNotFoundError as e:
        print(f"[eval] {e}", file=sys.stderr)
        return 2

    net = resolve_net(args.net)
    if not net:
        print("[eval] warning: no net found; phases that need weights will fail",
              file=sys.stderr)

    builddir = resolve_builddir(lc0, args.builddir)

    reports_root = Path(args.reports_root).resolve()
    if args.out_dir:
        report_dir = reports_root / args.out_dir
    else:
        suffix = "-dirty" if git_dirty() else ""
        report_dir = reports_root / f"{utc_stamp()}-{short_sha()}{suffix}"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir = report_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] writing report to {report_dir}", flush=True)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    for s in skip:
        if s not in ALL_PHASES:
            print(f"[eval] warning: --skip got unknown phase '{s}'; "
                  f"valid: {', '.join(ALL_PHASES)}", file=sys.stderr)

    phases_summary: Dict[str, Dict[str, Any]] = {}
    epd_path = Path(args.epd).resolve()

    phase_runners = [
        (PHASE_UNIT_TESTS, lambda: run_unit_tests(
            report_dir, builddir, log_dir, run_ninja=not args.skip_ninja)),
        (PHASE_BACKEND_BENCH, lambda: run_backend_bench(
            report_dir, lc0, net, args, log_dir, args.quick)),
        (PHASE_SEARCH_BENCH, lambda: run_search_bench(
            report_dir, lc0, net, args, log_dir, args.quick)),
        (PHASE_TACTICS, lambda: run_tactics(
            report_dir, lc0, net, args, log_dir, epd_path,
            movetime_ms=(200 if args.quick else 1000),
            max_positions=(20 if args.quick else None))),
    ]

    for name, fn in phase_runners:
        if name in skip:
            phases_summary[name] = {"status": "SKIPPED"}
            print(f"[eval] {name}: SKIPPED", flush=True)
            continue
        t0 = time.time()
        print(f"[eval] {name}: running...", flush=True)
        try:
            res = fn() or {}
            phases_summary[name] = {
                "status": "OK",
                "duration_s": time.time() - t0,
                **res,
            }
            print(f"[eval] {name}: OK ({phases_summary[name]['duration_s']:.1f}s)",
                  flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            (report_dir / f"{name}.failed").write_text(err + "\n")
            phases_summary[name] = {
                "status": "FAILED",
                "duration_s": time.time() - t0,
                "error": err,
            }
            print(f"[eval] {name}: FAILED — {err}", file=sys.stderr, flush=True)

    finished_t = time.time()
    finished_utc = utc_iso()

    scores = compute_scores(phases_summary)
    write_scores_csv(report_dir, scores)

    write_build_info(report_dir, lc0, net, args, started_utc, finished_utc,
                     finished_t - started_t, phases_summary, builddir)
    write_summary(report_dir, phases_summary, started_utc, finished_utc,
                  finished_t - started_t, lc0, net, args, scores=scores)

    score_blurbs: List[str] = []
    if scores["correctness_score"] is not None:
        score_blurbs.append(f"correctness={scores['correctness_score']:.1f}")
    if scores["stability_score"] is not None:
        score_blurbs.append(f"stability={scores['stability_score']:.1f}")
    if scores["backend_peak_nps"] is not None:
        score_blurbs.append(f"peak_nps={scores['backend_peak_nps']}")
    if scores["search_total_nps"] is not None:
        score_blurbs.append(f"search_nps={scores['search_total_nps']}")
    if score_blurbs:
        print(f"[eval] scores: {' '.join(score_blurbs)}", flush=True)

    if getattr(args, "pdf", False):
        try:
            pdf_path = _render_run_pdf_or_die(report_dir)
            print(f"[eval] pdf: {pdf_path}", flush=True)
        except _PdfError as e:
            print(f"[eval] pdf: skipped — {e}", file=sys.stderr, flush=True)

    print(f"[eval] done in {finished_t - started_t:.1f}s -> {report_dir}", flush=True)
    return 1 if any(v.get("status") == "FAILED" for v in phases_summary.values()) else 0


# ---------------------------------------------------------------------------
# subcommand: compare
# ---------------------------------------------------------------------------

def _fmt_delta(cur: Optional[float], base: Optional[float],
               unit: str = "", precision: int = 0) -> str:
    if cur is None or base is None:
        return ""
    delta = cur - base
    pct = (100.0 * delta / base) if base else float("inf") if delta else 0.0
    sign = "+" if delta >= 0 else ""
    if precision == 0:
        return f"{sign}{delta:.0f}{unit} ({sign}{pct:.2f}%)"
    return f"{sign}{delta:.{precision}f}{unit} ({sign}{pct:.2f}%)"


def _to_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _compare_unit_tests(base: Path, cur: Path) -> str:
    b = {(r["suite"], r["name"]): r["status"] for r in read_csv(base / "unit_tests.csv")}
    c = {(r["suite"], r["name"]): r["status"] for r in read_csv(cur / "unit_tests.csv")}
    if not b and not c:
        return ""
    flips = []
    for k in sorted(set(b) | set(c)):
        bs, cs = b.get(k, "MISSING"), c.get(k, "MISSING")
        if bs != cs:
            flips.append((k, bs, cs))
    out = ["## unit tests"]
    out.append(f"- baseline: {sum(1 for v in b.values() if v == 'passed')}/{len(b)} passed")
    out.append(f"- current:  {sum(1 for v in c.values() if v == 'passed')}/{len(c)} passed")
    if flips:
        out += ["", "| suite | test | baseline | current |",
                "| --- | --- | --- | --- |"]
        for (suite, name), bs, cs in flips:
            out.append(f"| {suite} | {name} | {bs} | {cs} |")
    else:
        out.append("- no status flips")
    return "\n".join(out)


def _compare_backend_bench(base: Path, cur: Path) -> str:
    b_rows = read_csv(base / "backend_bench.csv")
    c_rows = read_csv(cur / "backend_bench.csv")
    if not b_rows and not c_rows:
        return ""
    by_b = {r["batch_size"]: r for r in b_rows}
    by_c = {r["batch_size"]: r for r in c_rows}
    keys = sorted(set(by_b) | set(by_c), key=lambda x: _to_int(x) or 0)
    rows = []
    for k in keys:
        b = by_b.get(k, {})
        c = by_c.get(k, {})
        b_nps = _to_int(b.get("mean_nps", "")) if b else None
        c_nps = _to_int(c.get("mean_nps", "")) if c else None
        delta = (c_nps - b_nps) if (b_nps is not None and c_nps is not None) else None
        rows.append((k, b_nps, c_nps, delta))
    rows_sorted = sorted(rows, key=lambda r: abs(r[3] or 0), reverse=True)
    out = ["## backend bench (mean NPS)",
           "",
           "| batch_size | baseline | current | Δ |",
           "| --- | --- | --- | --- |"]
    for k, b_nps, c_nps, delta in rows_sorted:
        out.append(f"| {k} | {b_nps if b_nps is not None else '—'} | "
                   f"{c_nps if c_nps is not None else '—'} | "
                   f"{_fmt_delta(c_nps, b_nps)} |")

    def peak(xs):
        ints = [_to_int(r["mean_nps"]) for r in xs if _to_int(r.get("mean_nps", "")) is not None]
        return max(ints) if ints else None
    out += ["", f"- peak baseline: {peak(b_rows)}",
            f"- peak current:  {peak(c_rows)}",
            f"- Δ peak: {_fmt_delta(peak(c_rows), peak(b_rows))}"]
    return "\n".join(out)


def _compare_search_bench(base: Path, cur: Path) -> str:
    b_rows = read_csv(base / "search_bench.csv")
    c_rows = read_csv(cur / "search_bench.csv")
    if not b_rows and not c_rows:
        return ""
    by_b = {r["position_idx"]: r for r in b_rows}
    by_c = {r["position_idx"]: r for r in c_rows}
    keys = sorted(set(by_b) | set(by_c), key=lambda x: _to_int(x) or 0)
    rows = []
    for k in keys:
        b = by_b.get(k, {})
        c = by_c.get(k, {})
        b_nps = _to_int(b.get("nps", "")) if b else None
        c_nps = _to_int(c.get("nps", "")) if c else None
        delta = (c_nps - b_nps) if (b_nps is not None and c_nps is not None) else None
        rows.append((k, b_nps, c_nps, delta, b.get("bestmove", ""), c.get("bestmove", "")))
    rows_sorted = sorted(rows, key=lambda r: abs(r[3] or 0), reverse=True)
    out = ["## search bench (per-position NPS)",
           "",
           "| pos | baseline NPS | current NPS | Δ | base bestmove | cur bestmove |",
           "| --- | --- | --- | --- | --- | --- |"]
    for k, b_nps, c_nps, delta, b_bm, c_bm in rows_sorted:
        out.append(f"| {k} | {b_nps if b_nps is not None else '—'} | "
                   f"{c_nps if c_nps is not None else '—'} | "
                   f"{_fmt_delta(c_nps, b_nps)} | {b_bm} | {c_bm} |")

    def total_nps(xs):
        n = sum((_to_int(r.get("nodes", "")) or 0) for r in xs)
        t = sum((_to_int(r.get("time_ms", "")) or 0) for r in xs)
        return int(round(1000.0 * n / t)) if t else None
    out += ["", f"- total NPS baseline: {total_nps(b_rows)}",
            f"- total NPS current:  {total_nps(c_rows)}",
            f"- Δ total NPS: {_fmt_delta(total_nps(c_rows), total_nps(b_rows))}"]
    return "\n".join(out)


def _compare_tactics(base: Path, cur: Path) -> str:
    b_rows = read_csv(base / "tactics.csv")
    c_rows = read_csv(cur / "tactics.csv")
    if not b_rows and not c_rows:
        return ""
    by_b = {r["id"]: r for r in b_rows}
    by_c = {r["id"]: r for r in c_rows}
    flips = []
    for k in sorted(set(by_b) | set(by_c)):
        b_solved = (by_b.get(k, {}).get("solved", "") == "1")
        c_solved = (by_c.get(k, {}).get("solved", "") == "1")
        if b_solved != c_solved:
            flips.append((k, b_solved, c_solved,
                          by_b.get(k, {}).get("engine_bestmove", ""),
                          by_c.get(k, {}).get("engine_bestmove", "")))
    b_solved = sum(1 for r in b_rows if r.get("solved") == "1")
    c_solved = sum(1 for r in c_rows if r.get("solved") == "1")
    out = ["## tactics",
           f"- baseline: {b_solved}/{len(b_rows)} solved",
           f"- current:  {c_solved}/{len(c_rows)} solved",
           f"- Δ solved: {c_solved - b_solved}"]
    if flips:
        out += ["",
                "| id | baseline | current | base bestmove | cur bestmove |",
                "| --- | --- | --- | --- | --- |"]
        for k, bs, cs, bm, cm in flips:
            out.append(f"| {k} | {'solved' if bs else 'failed'} | "
                       f"{'solved' if cs else 'failed'} | {bm} | {cm} |")
    return "\n".join(out)


def _compare_scores(base: Path, cur: Path) -> str:
    b_rows = read_csv(base / "scores.csv")
    c_rows = read_csv(cur / "scores.csv")
    if not b_rows or not c_rows:
        return ""
    b, c = b_rows[0], c_rows[0]

    def num(d: Dict[str, str], k: str, cast=float) -> Optional[float]:
        v = d.get(k, "")
        if v == "" or v is None:
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    rows: List[Tuple[str, Optional[float], Optional[float], str, int]] = [
        ("correctness_score", num(b, "correctness_score"), num(c, "correctness_score"), "", 1),
        ("stability_score",   num(b, "stability_score"),   num(c, "stability_score"),   "", 1),
        ("unit_tests_pct",    num(b, "unit_tests_pct"),    num(c, "unit_tests_pct"),    "%", 2),
        ("tactics_pct",       num(b, "tactics_pct"),       num(c, "tactics_pct"),       "%", 2),
        ("backend_peak_nps",  num(b, "backend_peak_nps", int),
                              num(c, "backend_peak_nps", int), "", 0),
        ("search_total_nps",  num(b, "search_total_nps", int),
                              num(c, "search_total_nps", int), "", 0),
    ]
    out = ["## score deltas", "",
           "| metric | baseline | current | Δ |",
           "| --- | --- | --- | --- |"]
    for name, bv, cv, unit, prec in rows:
        if bv is None and cv is None:
            continue
        bv_s = "—" if bv is None else (f"{bv:.{prec}f}{unit}" if prec else f"{int(bv)}{unit}")
        cv_s = "—" if cv is None else (f"{cv:.{prec}f}{unit}" if prec else f"{int(cv)}{unit}")
        delta = _fmt_delta(cv, bv, unit=unit, precision=prec)
        out.append(f"| **{name}** | {bv_s} | {cv_s} | {delta} |")
    return "\n".join(out)


def cmd_compare(args: argparse.Namespace) -> int:
    base = Path(args.baseline).resolve()
    cur = Path(args.current).resolve()
    if not base.is_dir() or not cur.is_dir():
        print("usage: eval.py compare BASELINE CURRENT (both must be report dirs)",
              file=sys.stderr)
        return 2
    sections = [
        f"# compare report\n\n- baseline: `{base}`\n- current:  `{cur}`\n",
        _compare_scores(base, cur),
        _compare_unit_tests(base, cur),
        _compare_backend_bench(base, cur),
        _compare_search_bench(base, cur),
        _compare_tactics(base, cur),
    ]
    out = cur / "compare.md"
    out.write_text("\n\n".join(s for s in sections if s).rstrip() + "\n")
    print(f"wrote {out}")

    if getattr(args, "pdf", False):
        try:
            pdf_path = _render_compare_pdf_or_die(base, cur)
            print(f"wrote {pdf_path}")
        except _PdfError as e:
            print(f"pdf: skipped — {e}", file=sys.stderr)
            return 1
    return 0


# ---------------------------------------------------------------------------
# subcommand: pdf
# ---------------------------------------------------------------------------

class _PdfError(RuntimeError):
    pass


def _import_report_pdf():
    try:
        from . import report_pdf  # type: ignore
    except (ImportError, ValueError):
        # script is run directly; fall back to sibling import
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import report_pdf  # type: ignore
        except ImportError as e:
            missing = "matplotlib" if "matplotlib" in str(e) else (
                "numpy" if "numpy" in str(e) else str(e))
            raise _PdfError(
                f"PDF generation requires matplotlib + numpy ({missing} missing). "
                f"install with: pip install matplotlib numpy"
            ) from e
    return report_pdf


def _render_run_pdf_or_die(report_dir: Path) -> Path:
    rpdf = _import_report_pdf()
    return rpdf.render_run_pdf(report_dir)


def _render_compare_pdf_or_die(base: Path, cur: Path) -> Path:
    rpdf = _import_report_pdf()
    return rpdf.render_compare_pdf(base, cur)


def cmd_pdf(args: argparse.Namespace) -> int:
    report_dir = Path(args.report_dir).resolve()
    if not report_dir.is_dir():
        print(f"not a report dir: {report_dir}", file=sys.stderr)
        return 2
    try:
        if args.baseline:
            base = Path(args.baseline).resolve()
            if not base.is_dir():
                print(f"baseline not a report dir: {base}", file=sys.stderr)
                return 2
            out = _render_compare_pdf_or_die(base, report_dir)
        else:
            out = _render_run_pdf_or_die(report_dir)
    except _PdfError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# subcommand: latest
# ---------------------------------------------------------------------------

def cmd_latest(args: argparse.Namespace) -> int:
    root = Path(args.reports_root).resolve()
    if not root.is_dir():
        print(f"no reports root at {root}", file=sys.stderr)
        return 1
    dirs = [d for d in root.iterdir() if d.is_dir()]
    if not dirs:
        print(f"no report directories in {root}", file=sys.stderr)
        return 1
    latest = max(dirs, key=lambda d: d.stat().st_mtime)
    print(latest)
    return 0


# ---------------------------------------------------------------------------
# argparse / main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval.py",
        description="Run lc0 evaluation phases and collect tabular reports.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run an evaluation pass")
    pr.add_argument("--lc0-path", default=None,
                    help="path to lc0 binary (auto-discovers build/release/lc0 etc.)")
    pr.add_argument("--net", default=None,
                    help="path to network weights file (auto-discovers ./weights/*)")
    pr.add_argument("--backend", default=None,
                    help="lc0 --backend value (e.g. cuda-fp16, blas)")
    pr.add_argument("--backend-opts", default=None,
                    help="lc0 --backend-opts value forwarded verbatim")
    pr.add_argument("--quick", action="store_true",
                    help="reduced ranges (smaller batches, shorter movetimes)")
    pr.add_argument("--build", action="store_true",
                    help="run ./build.sh release before evaluating")
    pr.add_argument("--builddir", default=None,
                    help="meson builddir for unit tests (default: parent of lc0 binary)")
    pr.add_argument("--skip-ninja", action="store_true",
                    help="don't invoke ninja test; only parse existing *.xml in builddir")
    pr.add_argument("--out-dir", default=None,
                    help="override report directory name (defaults to <ts>-<sha>)")
    pr.add_argument("--reports-root", default=str(DEFAULT_REPORTS_DIR),
                    help="root directory for report dirs")
    pr.add_argument("--skip", default="",
                    help=f"comma-separated phases to skip "
                         f"({','.join(ALL_PHASES)})")
    pr.add_argument("--epd", default=str(DEFAULT_EPD),
                    help="path to EPD file for tactics phase")
    pr.add_argument("--pdf", action="store_true",
                    help="also write report.pdf (requires matplotlib+numpy)")
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", help="diff two prior report dirs")
    pc.add_argument("baseline", help="baseline report directory")
    pc.add_argument("current", help="current report directory (compare.md is written here)")
    pc.add_argument("--pdf", action="store_true",
                    help="also write compare.pdf alongside compare.md")
    pc.set_defaults(func=cmd_compare)

    pp = sub.add_parser("pdf",
                        help="(re)generate report.pdf or compare.pdf from existing CSVs")
    pp.add_argument("report_dir",
                    help="report directory to render (target for compare PDF too)")
    pp.add_argument("--baseline", default=None,
                    help="render compare PDF against this baseline dir")
    pp.set_defaults(func=cmd_pdf)

    pl = sub.add_parser("latest", help="print path of newest report directory")
    pl.add_argument("--reports-root", default=str(DEFAULT_REPORTS_DIR))
    pl.set_defaults(func=cmd_latest)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
