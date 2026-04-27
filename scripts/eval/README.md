# scripts/eval — lc0 evaluation harness

`eval.py` orchestrates the existing lc0 tools (`bench`/`benchmark`, `backendbench`,
the gtest unit tests) and a small UCI tactics suite, parks the results as CSV
tables under `eval-reports/<UTC-timestamp>-<git-short-sha>[-dirty]/`, and lets
you diff two runs.

No C++ changes — the orchestrator just tees and parses the existing tool stdout.

## Quick start

```bash
# Build first (if you haven't already).
./build.sh release

# Quick eval (a few minutes — reduced batches and movetimes).
python3 scripts/eval/eval.py run --quick \
    --lc0-path build/release/lc0 \
    --net path/to/weights.pb.gz

# Full eval (longer — full batch sweep + 34-position bench + full tactics).
python3 scripts/eval/eval.py run \
    --lc0-path build/release/lc0 \
    --net path/to/weights.pb.gz

# Compare two runs.
python3 scripts/eval/eval.py compare \
    eval-reports/20260426-220000-abc1234 \
    eval-reports/20260426-230000-def5678

# Print the most recent report directory.
python3 scripts/eval/eval.py latest
```

## Subcommands

### `run`

Runs four phases in order — each soft-fails (writes `<phase>.failed` and
continues) so one broken backend doesn't lose the unit-test or tactics signal.

| Phase | Underlying tool | Output |
|---|---|---|
| `unit_tests` | `ninja -C <builddir> test` (then parses gtest JUnit XML) | `unit_tests.csv` |
| `backend_bench` | `lc0 backendbench --header-only-once=true` | `backend_bench.csv` |
| `search_bench` | `lc0 bench` (quick) or `lc0 benchmark` (full) | `search_bench.csv` |
| `tactics` | `lc0` UCI loop driven over `positions/bench.epd` | `tactics.csv` |

### Flags

- `--lc0-path PATH` — path to the lc0 binary. Auto-discovers
  `build/release/lc0`, `build/debugoptimized/lc0`, `builddir/lc0`,
  `build/lc0`.
- `--net PATH` — network weights file. If omitted, lc0's own autodiscover
  (under `./` and `./weights/`) takes over.
- `--backend NAME` — forwarded as `--backend=NAME` to all lc0 invocations
  (e.g. `cuda-fp16`, `blas`, `onnx-cuda`).
- `--backend-opts STR` — forwarded as `--backend-opts=STR` verbatim.
- `--quick` — reduced ranges: backend batch sweep 1→64 step 8 over 20 batches,
  `lc0 bench` (10 positions × 500 ms movetime), tactics movetime 200 ms over
  20 positions.
- `--build` — runs `./build.sh release` before evaluating.
- `--builddir PATH` — meson builddir for unit tests. Defaults to the parent
  of the resolved lc0 binary if it contains a `build.ninja`.
- `--skip-ninja` — don't invoke `ninja test`; only parse `*.xml` already in
  the builddir.
- `--skip a,b` — skip phases (any of `unit_tests,backend_bench,search_bench,tactics`).
- `--out-dir NAME` — override the report directory name. By default we use
  `<UTC-timestamp>-<git-short-sha>` and append `-dirty` when the working tree
  has uncommitted changes — that surfaces "this number came from uncommitted
  code" in the directory name itself.
- `--reports-root PATH` — override `eval-reports/` root.
- `--epd PATH` — override the EPD file (default: `positions/bench.epd`).

### `compare BASELINE CURRENT`

Reads CSVs from both directories, computes deltas joined on natural keys, and
writes `compare.md` inside `CURRENT`. Sections:

- Unit-test status flips (pass→fail / fail→pass).
- Per-batch backend NPS deltas (sorted by absolute Δ).
- Per-position search-bench NPS deltas.
- Tactics flips (`solved` 0↔1) plus aggregate solved counts.

### `latest`

Prints the path of the most recent directory under `eval-reports/`. Handy as
input to `compare`:

```bash
python3 scripts/eval/eval.py compare \
    "$(python3 scripts/eval/eval.py latest)" \
    /path/to/older-run
```

## Output files

Every successful `run` writes:

```
eval-reports/<UTC>-<sha>[-dirty]/
├── build_info.md           # git, host, GPU, lc0 version, net hash, build options
├── summary.md              # one-screen overview + scores, links to the rest
├── scores.csv              # one-row CSV: headline metrics for tracking over time
├── unit_tests.csv          # suite, name, status, time_ms, message
├── backend_bench.csv       # batch_size, mean_nps, mean_ms, sdev_ms, cv,
│                           # max_nps, median_nps, min_nps, first_max_ms, first_mean_ms
├── search_bench.csv        # position_idx, fen, time_ms, nodes, nps, bestmove
├── tactics.csv             # id, fen, expected_bm, expected_am, engine_bestmove,
│                           # score_cp, depth, nodes, time_ms, solved
└── logs/
    ├── unit_tests.ninja.log
    ├── backend_bench.log
    ├── search_bench.log
    └── tactics.uci.log
```

`<phase>.failed` markers appear next to the CSVs when a phase soft-fails.
`compare.md` is written in-place by `eval.py compare`.

## Scores

`summary.md` includes a `## scores` section and a one-row `scores.csv` for
tracking headline numbers across runs:

| field | meaning |
|---|---|
| `correctness_score` | mean of measured correctness rates (unit-test pass-rate, tactics solved-rate) — 0-100. Phases that didn't run or failed are excluded from the average, so a partial run still yields a meaningful number. |
| `stability_score` | fraction of non-skipped phases that completed OK — 0-100. |
| `unit_tests_passed` / `unit_tests_total` / `unit_tests_pct` | gtest pass-rate. |
| `tactics_solved` / `tactics_total` / `tactics_pct` | EPD tactics solved rate. |
| `backend_peak_nps` / `backend_peak_batch` | highest mean NPS seen in `backendbench` and the batch size at which it was reached. Hardware/net dependent — meaningful only relative to a baseline. |
| `search_total_nps` / `search_total_nodes` | aggregate from `lc0 bench`/`benchmark`. |
| `phases_ok` / `phases_failed` / `phases_skipped` / `phases_total` | counts. |

Performance numbers are intentionally **not** normalized to a "/100" — absolute
NPS is hardware-dependent, so a magic number would lie. Use `eval.py compare`
to get a meaningful Δ% between two runs on the same machine.

## PDF reports (optional)

Pass `--pdf` to `run` or `compare` to render a paper-style PDF alongside
the markdown/CSV outputs. The renderer assembles a LaTeX `article`
document (title page, abstract, sections per phase, booktabs tables,
appendix), with charts embedded as vector PDFs from matplotlib, then
compiles it via `pdflatex` (run twice for cross-references).

Requirements:
- **matplotlib** + **numpy** (Python deps for the figures)
- **pdflatex** + standard packages (booktabs, tabularx, hyperref,
  fancyhdr, microtype, caption, siunitx, amssymb, float, enumitem,
  seqsplit). On Debian/Ubuntu, install with:

  ```bash
  sudo apt-get install texlive-latex-recommended texlive-latex-extra \
                       texlive-fonts-recommended texlive-science
  ```

The default `run`/`compare` paths stay stdlib-only — `--pdf` is opt-in,
and a missing `pdflatex` produces a friendly error rather than crashing
the eval.

```bash
# render report.pdf into the run dir
python3 scripts/eval/eval.py run --quick --net weights/foo.pb.gz --pdf

# render compare.pdf alongside compare.md
python3 scripts/eval/eval.py compare BASELINE_DIR CURRENT_DIR --pdf

# (re)generate from existing CSVs without re-running the eval
python3 scripts/eval/eval.py pdf  REPORT_DIR
python3 scripts/eval/eval.py pdf  CURRENT_DIR --baseline BASELINE_DIR
```

`report.pdf` (~5 pages, A4 portrait, `article` class):
1. **Cover** — title block, automated abstract, run-metadata `tabularx` table.
2. **Headline scores** — methodology paragraph, scores table, score radar chart.
3. **Backend Microbenchmark** — NPS vs batch (with min/max ribbon and peak marker), latency vs batch (mean ± stdev, cold-batch overlay).
4. **Search Benchmark** — per-position NPS bars (mean/median refs), bestmove labels above bars, full per-position table.
5. **Tactics Suite** — solved/unsolved bars by position id, expected vs engine bestmove + eval + nodes table (✓/✗ glyphs in green/red).
6. **Unit Tests** — pass-count summary; failure table only when failures exist.
7. **Appendix A: Build Information** — full key/value snapshot in a `description` list (long paths wrap via `\seqsplit`).

`compare.pdf` (~3-4 pages):
1. **Subjects + Score Deltas** — paths/sha block, score-delta `tabularx` (green/red), Δ% horizontal bar chart.
2. **Backend Microbenchmark** — NPS curves overlaid + per-batch Δ% bars.
3. **Search Benchmark** — side-by-side per-position bars + total-NPS Δ callout.
4. **Tactics Flips** — only positions whose `solved` status changed (or "no changes" message).

PDFs embed TrueType fonts (DejaVu Sans for figures, Computer Modern for body),
use `tab10` colours, and are vector — zoom in without quality loss.

**Compilation artefacts** (`.aux`, `.log`, `.out`) are written to a temp dir
and removed automatically; only the final PDF lands in the report dir.

## EPD format

`positions/bench.epd` uses **UCI long-algebraic** moves for `bm`/`am` operands
(e.g. `bm e2e4`, not `bm e4`). This deviates from standard EPD but avoids
shipping a SAN→UCI converter. Multiple alternatives separated by spaces are
all considered solutions.

```
<piece-placement> <stm> <castling> <ep> bm <uci> [<uci>...]; id "<name>";
```

Curate your own positions over time — the file is short and meant to be
edited.

## Limitations

- The tactics phase scores correctness against the EPD's `bm`/`am`. A move
  the engine plays at `--quick` movetime may differ from what it would play
  at tournament speed; use full mode for a fairer reading.
- `backend_bench`/`search_bench` numbers vary with GPU clock, thermals, and
  what else is running. Compare runs on the same machine in the same session
  for reliable Δs.
- No CI integration is wired up. The script is invocable by humans; wiring
  into CircleCI is a follow-up if/when the team wants regression checks on
  every PR.
- No plotting. The CSVs are the contract; render them with whatever tool
  you prefer.
