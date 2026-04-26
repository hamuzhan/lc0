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
├── summary.md              # one-screen overview, links to the rest
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
