# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build system

Lc0 uses **Meson + Ninja**. The `./build.sh [release|debug|debugoptimized|plain|minsize]` wrapper does `meson setup` (or reconfigure) followed by `meson compile`. Build options are passed through:

```bash
./build.sh release -Dcudnn=true -Dnvcc_ccbin=g++-12 -Dgtest=false
```

For active development the upstream convention is `meson setup builddir/ && ninja -C builddir lc0` (use `ninja lc0`, not bare `ninja`, because `ninja` builds tests too). VSCode autodiscovers `builddir/`.

Key options in `meson_options.txt` (defaults shown only when surprising):

- `plain_cuda` *(true)* — modern CUDA backend (`cuda`, `cuda-fp16`, `cuda-auto`). Uses cublasLt + custom kernels, supports both residual and transformer-body nets.
- `cudnn` *(false)* — legacy cuDNN backend (`cudnn`, `cudnn-fp16`). Both `network_cudnn.cc` and `network_cuda.cc` are compiled when this is on.
- `cutlass` *(true)* — only kicks in when `max_cuda >= 800` (Ampere+). Adds `cutlass_kernels.cu` to the cuda backend; produces faster GEMMs on Ampere/Hopper.
- `native_cuda` *(true)* / `cc_cuda` — `native_cuda` uses `nvcc -arch=native`; override with `-Dcc_cuda=90` for Hopper, `100` for Blackwell. If neither is set, falls back to `-arch=all-major`.
- `nvcc_ccbin` — set when nvcc rejects the system compiler (e.g. `-Dnvcc_ccbin=g++-12`).
- `cudnn_libdirs` / `cudnn_include` — defaults look in `/opt/cuda` and `/usr/local/cuda`. Pass arrays for non-standard installs.
- `onnx` *(true)* — ONNX Runtime backends (`onnx-cpu`, `onnx-cuda`, `onnx-trt`, `onnx-rocm`, `onnx-dml`). `onnx_libdir`/`onnx_include` point at an unpacked onnxruntime release.
- `xla`, `sycl=l0|amd|nvidia`, `dx`, `metal` — additional backend families, all off by default on Linux.
- `default_backend`, `default_search` — bake a default into the binary (`-Ddefault_backend=cuda-fp16`).
- `dag_classic` *(true)*, `gtest` *(true)*, `lc0` *(true)*, `rescorer` *(false)*, `python_bindings` *(false)*, `trace_library=off|perfetto|nvtx`.

For a binary that contains *only* selected backends, run `meson configure builddir -D<other_backend>=false`. The build fails with "No usable computation backends" if every backend is off and `build_backends=true`.

## Tests

Unit tests live next to source in `*_test.cc` files (gtest). They are compiled into separate executables that share `lc0_lib` (a static lib of `common_files`). Run all:

```bash
ninja -C builddir test
```

Run one test executable:

```bash
ninja -C builddir engine_test && ./builddir/engine_test --gtest_filter='SomeFixture.*'
```

Available test targets: `chessboard_test`, `fp16_test`, `hashcat_test`, `position_test`, `optionsparser_test`, `syzygy_test`, `encoder_test`, `engine_test` (see `meson.build` bottom).

Pre-flight on a Linux box without onnxruntime: `-Dgtest=false -Donnx=false` to skip those deps. CircleCI uses `-Dgtest=false -Donnx_include=... -Donnx_libdir=...`.

## Eval harness

`scripts/eval/eval.py` orchestrates `ninja test`, `lc0 backendbench`, `lc0 bench`/`benchmark`, and a small EPD tactics suite, and writes per-phase CSV reports under `eval-reports/<ts>-<sha>[-dirty]/`. Use `--quick` for a few-minute pass during iteration; drop `--quick` for the full sweep. `eval.py compare BASELINE CURRENT` writes a diff `compare.md` into `CURRENT`. See `scripts/eval/README.md` for column definitions and flag list.

## Runtime modes

`lc0` is a single binary with three modes selected by the first arg:

| Mode | Implementation |
|------|----------------|
| `uci` *(default)* | `engine_loop.cc` → `engine.cc` (UCI protocol, search) |
| `selfplay` | `selfplay/loop.cc` → `selfplay/tournament.cc` (mass game generation, training data) |
| `debug` | Generates per-position debug data |

`./lc0 selfplay --help` lists the selfplay-specific flags (the `FLAGS.md` table only documents UCI). Important selfplay options: `--parallelism=N` (concurrent games — also the NN batch size), `--visits=N` / `--playouts=N`, `--training=true` (write V6/V7 chunks via `trainingdata/writer.cc`), `--openings-pgn`, `--share-trees`, `--total-games`.

A `lc0.config` file in CWD (or `--config=path`) supplies long flags one per line; CLI overrides config.

## Architecture

### Top-level flow

```
main.cc → CommandLine::ConsumeCommand()
  ├─ uci/(empty)  → engine_loop.cc → engine.cc → SearchBase + Backend
  ├─ selfplay     → selfplay/loop.cc → SelfPlayTournament → many SelfPlayGame
  └─ debug        → tools/...
```

### Search abstraction (`src/search/`)

`SearchBase` (`search/search.h`) is the runtime interface every search algorithm implements: `SetBackend`, `SetPosition`, `StartSearch`, `WaitSearch`, `StopSearch`, `GetArtifacts` (training data). `SearchFactory` registers algorithms; selection via `--search` flag or `default_search` build option.

Implementations:

- `search/classic/` — production MCTS. `node.cc` is the tree, `search.cc` is the worker pool, `params.cc` exposes UCI options, `stoppers/` decide when to stop. **`search/classic/node.cc` is in `common_files`** (linked into every binary including tests) — the classic node type is shared infrastructure even for non-classic searches.
- `search/dag_classic/` — DAG-based variant guarded by `-Ddag_classic=true`.
- `search/instamove/` — trivial "play first legal move" search, useful for unit tests and as a backend latency probe.

### Neural network layer (`src/neural/`)

There are **two parallel registration systems**, do not confuse them:

- **Legacy** `NetworkFactory` (`neural/factory.cc`) with `REGISTER_NETWORK(name, fn, prio)` macro. Almost every concrete backend file (`network_cuda.cc`, `network_cudnn.cc`, `network_onnx.cc`, `network_metal.cc`, `network_dx.cc`, `network_random.cc`, `network_check.cc`, `network_demux.cc`, `network_mux.cc`, `network_record.cc`, `network_rr.cc`, `network_trivial.cc`) registers here. They produce `Network`/`NetworkComputation` (older, batch-shaped API).
- **Modern** `BackendManager` (`neural/register.h`) with `REGISTER_BACKEND` macro. Produces `Backend`/`BackendComputation` (`neural/backend.h`) — a per-batch streaming API with cache integration.

`neural/wrapper.cc` adapts legacy `Network` → modern `Backend` so the search layer (which calls `Backend::CreateComputation()`) works uniformly. When adding a new backend, prefer the modern `Backend` interface; when modifying CUDA/cuDNN/ONNX, you are still in legacy-Network territory.

### Network file format

Lc0 nets are gzipped protobuf (`proto/net.proto` → `pblczero::Net`). Loaded by `neural/loader.cc`; raw weights live in `pblczero::Weights` and are turned into runtime structs by `neural/network_legacy.{h,cc}`:

- `BaseWeights` — common pieces: input conv, embedding, smolgen globals, encoder layers (MHA + FFN + smolgen), residual tower, moves-left head.
- `LegacyWeights : BaseWeights` — single policy + value head (older nets).
- `MultiHeadWeights : BaseWeights` — `unordered_map<string, PolicyHead/ValueHead>` for multi-head nets (current generation, e.g. BT3/BT4).

Two architectural families coexist in the same proto:
1. **Residual tower** (AlphaZero-style ConvBlock + SE), policy via `policy1/policy` conv heads or "attention policy" with extra encoder.
2. **Transformer body** (BT2/BT3/BT4 lineage) — `encoder` stack with `MHA + Smolgen + FFN`, no residual tower. A net can use both, but new nets are encoder-only.

Backend code branches on `weights.encoder.size() > 0` vs `weights.residual.size() > 0`.

### CUDA / cuDNN backend internals (`src/neural/backends/cuda/`)

- `network_cuda.cc` — modern path (`-Dplain_cuda=true`). Templated `CudaNetwork<DataType>` (fp32 / fp16). Supports CUDA Graphs (`CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS` for CUDART ≥ 11.1). Registered as `cuda`, `cuda-fp16`, `cuda-auto`.
- `network_cudnn.cc` — legacy path, only compiled when `-Dcudnn=true`. Same templated layout. Registered as `cudnn`, `cudnn-fp16`. The two files share helpers like `getMaxAttentionHeadSize`/`getMaxAttentionBodySize`.
- `layers.{cc,h}` — layer abstractions (Conv, FC, MHA, Encoder, SE, etc.). 2.5kloc, the heart of the backend.
- `kernels.h` + `common_kernels.cu` — fp32 (and gpu-arch-agnostic fp16) custom kernels.
- `fp16_kernels.cu` — fp16-specific kernels that need newer hardware.
- `cutlass_kernels.cu` — only built with `-Dcutlass=true` and `max_cuda >= 800`. CUTLASS GEMMs for Ampere+.
- `inputs_outputs.h` — pinned host buffers / device staging for the CudaNetworkComputation.
- `cuda_common.h` — `ReportCUDAErrors`, common macros.

The cuda backend includes `winograd_helper.inc` as a header dep for both `common_kernels.cu` and `fp16_kernels.cu` — `meson.build` declares this in `depend_files`.

`utils/fp8_utils.h` provides scalar FP8 E5M2 / E4M3FN ↔ FP32 conversions but **no FP8 backend exists in tree yet**; FP8 is currently CPU-only utility code.

### Selfplay & training data (`src/selfplay/` + `src/trainingdata/`)

- `tournament.cc` — orchestrates `parallelism` worker threads, each running `PlayOneGame` or `PlayMultiGames`.
- `multigame.cc` — `Evaluator` abstraction: a worker may play multiple games whose NN evals are batched into one `BackendComputation`. Tied to `classic::NodeTree`.
- `game.cc` — single-game state machine.
- `trainingdata/writer.cc` — emits V6 (`trainingdata_v6.h`) or V7 (`trainingdata_v7.h`) gzipped chunks. `reader.cc` for round-tripping.
- `trainingdata/rescorer.cc` — Gaviota tablebase rescoring; lives behind `-Drescorer=true` and pulls the `gaviotatb` subproject.

### Other neural surfaces

- `neural/onnx/` — converter from internal `Net` → ONNX (`leela2onnx` tool) and back (`onnx2leela`). The ONNX backend (`backends/onnx/network_onnx.cc`) is the path for `onnx-trt` / `onnx-cuda` execution providers.
- `neural/xla/` — XLA backend (build flag `-Dxla=true`); HLO builder converts ONNX to HLO and submits via PJRT.
- `neural/tables/` — fixed-size policy index maps (chess-move ↔ policy slot).
- `neural/backends/network_*.cc` (without subdirectory) — meta-backends: `mux` (round-robin pin to multiple real backends), `demux` (one input → many backends, useful for testing), `check` (compare two backends position-by-position), `record` (dump evaluations), `rr`, `random`, `trivial`. These are pure C++ wrappers, no GPU code.

## Conventions

- **C++20**, GCC ≥ 10 or clang ≥ 12 supported. CI builds with both `gcc-10` and `clang-12` on Ubuntu 20.04.
- **Logging**: `CERR << ...` (stderr + log), `LOGFILE << ...` (log only).
- **Exceptions**: only `lczero::Exception` (`utils/exception.h`).
- **Headers**: `#pragma once`, no traditional guards.
- **Style**: Google C++ with the modifications listed in `CONTRIBUTING.md`. Run `clang-format` before committing.
- **Abseil** (`absl::`) is used since v0.32.
- **Tests** are gtest/gmock, file naming `xxx_test.cc` next to `xxx.cc`, registered in `meson.build`.
- Every new file in the engine core needs the GPL v3 banner **with the additional NVIDIA CUDA-libraries permission** (see existing headers). New non-NVIDIA backend code can omit the exception, but then NVIDIA libs cannot be linked into that file's binary.
- Non-`absl::` flat_hash_map / threads / zlib are project deps; everything else falls back to a meson subproject (`subprojects/`, e.g. `eigen`, `zlib`, `cutlass`, `gaviotatb`, `abseil-cpp`).

## Contribution norms (from `CONTRIBUTING.md`)

- PRs are **squash-merged**; if you have stacked branches, rebase with `git rebase --update-refs --onto upstream/master <merged-branch>`.
- Anything that may affect playing strength must be tested (the project does not have an automated strength CI; coordinate in Discord `#testing-discuss`).
- AI-generated code: must be disclosed in the PR description. The project explicitly notes that AI assistance has "failed so far on core lc0 engine code" and discourages agentic coding for engine internals.

## Repo nav cheatsheet

- Top-level entry: `src/main.cc`.
- UCI protocol parsing: `src/chess/uciloop.cc`.
- Bitboards / moves / position: `src/chess/{board,position,gamestate}.cc`.
- Engine glue: `src/engine.cc`, `src/engine_loop.cc`.
- Syzygy tablebases: `src/syzygy/syzygy.cc`.
- Tools (`./lc0` subcommands and standalone): `src/tools/{benchmark,backendbench,describenet,leela2onnx,onnx2leela}.cc`.
- Submodules: `proto/` (training-shared protobufs) — `git submodule update --init` is required after a fresh clone if `.gitmodules` is non-empty.
