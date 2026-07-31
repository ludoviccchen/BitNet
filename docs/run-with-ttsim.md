# Running BitNet-TT with the ttsim simulator

This guide explains how to build and run this repo using
[ttsim](https://github.com/tenstorrent/ttsim), Tenstorrent's functional
simulator, instead of a real Blackhole/Wormhole chip. It documents the
state of the port as described in [`PORTING_PLAN.md`](../PORTING_PLAN.md),
which remains the reference for technical detail and known limitations.

## 0. Where the port stands

The `ggml-ttnn` backend (directory
[`3rdparty/llama.cpp/ggml/src/ggml-ttnn/`](../3rdparty/llama.cpp/ggml/src/ggml-ttnn/))
currently:

- opens a real TT-Metalium device (silicon or ttsim, chosen via
  `TT_METAL_SIMULATOR`, never branched on in the backend's own code);
- allocates one DRAM buffer per tensor (weights, activations, KV-cache...)
  and stores data **as-is**, with no type conversion - including `I2_S`
  weights, packed byte-for-byte identical to the GGUF file;
- offloads `GGML_OP_MUL_MAT` when `src0` is `I2_S` and `src1`/`dst` are
  `F32`, via a custom trio of TT-Metalium kernels (reader/compute/writer,
  "Option B") that unpacks the ternary weights *on-device*, without going
  through TT-NN or a bf16 dequantize-on-upload step;
- offloads `GGML_OP_SET_ROWS` (the KV-cache write) and the pure view ops
  (`VIEW`/`RESHAPE`/`TRANSPOSE`/`PERMUTE`, e.g. reading the KV-cache back
  for attention) - which makes a **complete** offload of a real model
  possible, KV-cache included, with no workaround flag needed.

Everything else (RMSNorm, RoPE, softmax, embeddings...) stays on CPU - this
is normal ggml-backend graph-splitter behavior, not a limitation being
worked around. A full forward pass on a real model works end to end under
ttsim (section 6) - but this is still a **functional correctness** test, not
a performance one (ttsim is a functional simulator; see section 7).

## 1. Prerequisites

In this environment, everything is already in place (verified):

- `TT_METAL_HOME=~/stage_bitnet/Bitnet-TT/tt-metal` - a prebuilt tt-metal
  tree, with `build_Release/` containing the `tt-metalium` CMake config.
- `~/stage_bitnet/Bitnet-TT/sim/libttsim_bh.so` +
  `~/stage_bitnet/Bitnet-TT/sim/soc_descriptor.yaml` (a copy of
  `blackhole_140_arch.yaml`) - the Blackhole ttsim binary and its SoC
  descriptor, already sitting side by side as ttsim requires.
- SFPI installed (`/opt/tenstorrent/sfpi`).
- `tt-umd`/`tt-exalens` with the TTSim fixes (already validated: the
  `metal_example_add_2_integers_in_riscv` example runs cleanly under ttsim
  in this environment).

If you're starting from a fresh environment, follow
[`tt_metal/tt-llk/tests/TTSIM.md`](../../tt-metal/tt_metal/tt-llk/tests/TTSIM.md)
in the `tt-metal` repo to set this back up.

## 2. Environment variables

```bash
export TT_METAL_HOME=~/stage_bitnet/Bitnet-TT/tt-metal
# The tt-metal runtime (rtoptions.cpp) does NOT read TT_METAL_HOME itself:
# without this, any binary not launched from the tt-metal repo root fails
# with "TT_FATAL: Root Directory is not set."
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME

export TT_METAL_SIMULATOR=~/stage_bitnet/Bitnet-TT/sim/libttsim_bh.so
export TT_METAL_SLOW_DISPATCH_MODE=1     # ttsim only supports slow dispatch
export TT_METAL_DISABLE_SFPLOADMACRO=1   # SFPLOADMACRO isn't implemented by ttsim

# Needed for any binary launched outside the tt-metal build tree
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:$LD_LIBRARY_PATH
```

The first four variables are exactly the ones used to validate the ttsim
stack end to end in this environment; the last two (`RUNTIME_ROOT`,
`LD_LIBRARY_PATH`) are pitfalls specific to running *outside* the tt-metal
tree (relevant here, since the final binary is `BitNet/build/bin/llama-cli`).

## 3. Building BitNet with the TTNN backend

The repo's CMake has a dedicated flag, `-DGGML_TTNN=ON`, separate from the
normal CPU build:

```bash
cd ~/stage_bitnet/Bitnet-TT/BitNet
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_TTNN=ON \
  -DBITNET_X86_TL2=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-cli -j"$(nproc)"
```

The four `LLAMA_BUILD_*` flags are the same ones `setup_env.py` passes for
the normal CPU build - without them, `LLAMA_BUILD_TOOLS` defaults to `OFF`
(the `3rdparty/llama.cpp` subproject isn't "standalone" here), and
`add_subdirectory(tools)` never runs: the `llama-cli` target simply doesn't
exist and `cmake --build` fails with `No rule to make target 'llama-cli'`.

`3rdparty/llama.cpp/ggml/src/ggml-ttnn/CMakeLists.txt` derives
`TT-Metalium_DIR` from `TT_METAL_HOME` on its own (so export the section 2
variables **before** running `cmake -B`) - unlike an earlier version of this
backend, you no longer need to pass `-DTT-Metalium_DIR=...`/`-Dtt-nn_DIR=...`
by hand: since the backend now consumes packed ternary weights directly via
custom kernels ("Option B", see `PORTING_PLAN.md` §13-15), TT-NN isn't
linked at all anymore, only TT-Metalium is needed. If CMake still picks the
wrong `tt-metalium-config.cmake` (a second, broken copy with no
`Metalium.cmake` next to it sits directly under
`$TT_METAL_HOME/build_Release/`), pass the override explicitly - see
section 8.

## 4. Verifying the backend loads under ttsim

Once built, `llama-cli` should enumerate a `TT_METALIUM0` device:

```bash
./build/bin/llama-cli --list-devices
```

You should see a device named `TT_METALIUM0` (description `TT_Metalium`)
alongside the CPU. If `TT_METAL_SIMULATOR` is set correctly, this device
opens against ttsim rather than silicon - nothing more to do on the BitNet
side, the silicon/ttsim choice is entirely decided by tt-metal via that
variable.

Actual output observed in this environment, section 2 variables exported:

```
Available devices:
  TT_METALIUM0: Tenstorrent Blackhole (TT-Metalium/TT-NN backend, registration skeleton) (0 MiB, 0 MiB free)
```

(The "0 MiB" and "registration skeleton" come from
`ggml_backend_dev_get_memory` not being implemented yet for this backend -
no consequence for enumeration or offload.)

If you get `TT_FATAL: Root Directory is not set.`, `TT_METAL_RUNTIME_ROOT`
is missing (section 2).

## 5. Preparing an I2_S model

```bash
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

On a system with an "externally managed" Python (recent Debian/Ubuntu,
PEP 668 - `setup_env.py` then fails on
`pip install 3rdparty/llama.cpp/gguf-py` with
`error: externally-managed-environment`), create a venv first and install
dependencies there - `setup_env.py` uses `sys.executable` internally, so
running it from inside the venv is enough, no script changes needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

This repo (current branch) can't produce an `I2_S` GGUF locally with its own
`llama-quantize` (`LLAMA_FTYPE_MOSTLY_I2_S` isn't in the `QUANT_OPTIONS`
table) - only an already-quantized `I2_S` GGUF (like
`microsoft/BitNet-b1.58-2B-4T-gguf`, downloaded by the command above) works.
See `PORTING_PLAN.md` §12 for details.

## 6. Running inference with full offload on ttsim

`run_inference.py` hardcodes `-ngl 0` - it will **never** exercise the TTNN
offload. To exercise the backend, invoke `llama-cli` directly:

```bash
./build/bin/llama-cli \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "You are a helpful assistant" \
  -n 1 -t 2 --single-turn \
  -ngl 99 -dev TT_METALIUM0
```

- `-ngl 99` moves the `I2_S` weights of every layer onto the TTNN backend's
  device buffer; ggml-backend's graph-splitter automatically keeps
  everything `supports_op()` refuses (norms, RoPE, softmax...) on CPU.
- `--single-turn` makes `llama-cli` exit after one response instead of
  dropping into the interactive chat loop (which, with no useful stdin,
  spins on empty lines).
- **No `-nkvo` needed**: the KV-cache also runs on `TT_METALIUM0` now
  (written via `SET_ROWS`, read back via `VIEW` - see `PORTING_PLAN.md`
  §19). Passing it still forces the KV-cache onto CPU; that remains a valid
  configuration if you want to compare, not a required workaround.

**Known issue: this command currently does not complete.** Since
`PORTING_PLAN.md` §26, `MUL_MAT` offload is no longer gated on `M` being
tile-aligned - decode-step matmuls genuinely reach `TT_METALIUM0` now,
confirmed via `GGML_SCHED_DEBUG` and extensive standalone (isolated
matmul) testing at every real projection shape. A real multi-layer model
graph originally hit a *different*, pre-existing gap this surfaced for
the first time (`GGML_ASSERT(... "mul_mat dst must not be a view")`) -
**root-caused and fixed in §27**: ggml's graph allocator recycles a dead
scratch tensor's raw address for a later, differently-sized tensor
without always calling this backend's `init_tensor` callback again,
leaving a stale, wrongly-sized entry in this backend's per-address buffer
registry (nothing to do with an actual view). With that fixed, the same
command ran for **93 minutes** - dramatically further than ever before -
before hitting a *different*, not-yet-root-caused `SIGSEGV` inside
tt-metal's `read_shard_from_device`, only reachable this deep into
sustained real execution (see §27). **Until that's resolved**, expect the
command above to eventually crash rather than complete; the per-shape
correctness and timing numbers in §26/§27 come from the standalone kernel
harness, not a completed end-to-end `llama-cli` run.

Actual output observed in this environment (section 2 variables exported,
section 5 model in place, from an earlier `-n 16` run before §26's change,
back when the `M%32==0` gate kept most decode-step matmuls on CPU and
neither of the above was ever reached - kept here as a correctness/
output-shape reference, not a currently-reproducible example):

```
> You are a helpful assistant
a helpful assistant, you are an assistant, you are a helpful assistant. However
[ Prompt: 13.4 t/s | Generation: 1.3 t/s ]
Exiting...
```

The quality of the generated text doesn't matter here (a 2B-parameter model,
no sampling tuning) - what matters is that the offload pipeline runs
without asserting/crashing, which - per the known issue above - it
currently does not once `M` is unconstrained. See §26 for current,
real per-projection standalone timing, and §7 for why ttsim's absolute
numbers were never a performance proxy to begin with.

## 7. Known limitations to keep in mind

Full detail in `PORTING_PLAN.md` §9, §13-28; summary:

- **Performance is not representative**: every offloaded `mul_mat` and
  `SET_ROWS` does a full host↔device round trip per call (the buffer type
  doesn't store an optimized native representation, and ttsim is a
  functional simulator, not a timing model). The t/s numbers shown say
  nothing about real silicon - performance work is explicitly deferred
  until real hardware exists (`PORTING_PLAN.md` §8).
- **Broader coverage**: verified (`PORTING_PLAN.md` §21) - longer
  generations, varied prompts, and batch>1 (`llama-batched -np 4 -kvu`) all
  complete cleanly. Along the way, found that ordinary single-token decode
  steps (`M=1`) never actually place `MUL_MAT` on `TT_METALIUM0` - the
  kernel's `M % 32 == 0` tile-alignment requirement means only large enough
  prefill batches offload; steady-state generation's matmuls fall back to
  CPU via the normal scheduler mechanism (not a bug, just a narrower win
  than "full offload" suggests).
- **Decode-step (`M=1`) offload attempted, reverted**: `PORTING_PLAN.md` §22
  implemented and verified (in isolation) a fix for the §21 gap - padding
  the activation to a tile boundary so any `M >= 1` is offloadable. Enabling
  it, though, surfaced a much bigger problem: the reader kernel's ternary
  unpack loop takes minutes per call at this model's real projection sizes
  under ttsim (measured: one `K=2560,N=2560` call didn't finish in over
  580s), invisible until now because the `M%32==0` gate had incidentally
  kept real per-layer decode matmuls away from this kernel entirely. The
  `M%32==0` check is restored by default so the smoke test below stays fast
  (~15-30s) - the padding fix is in place and correct, just not switched on,
  pending a real optimization pass on the reader kernel's unpack loop.
- **Reader kernel unpack loop optimized (scalar tidy-up), still not
  enough**: `PORTING_PLAN.md` §23 cut real per-element cost (precomputed
  tile-face index table instead of recomputing it via division/modulo on
  every element; four K-tiles sharing a packing superblock unpacked from
  one shared byte load instead of four redundant passes) - verified
  correct at every scale tested. Real effect measured directly: a
  `K=2560,N=2560` matmul call that never finished in over 580s before now
  completes in ~930s (15.5 min) - a genuine speedup, but nowhere near
  practical.
- **Unpack moved to the SFPU - ~6x faster, still not enough for full
  offload**: `PORTING_PLAN.md` §24 went further than §23's scalar tidy-up:
  the actual 2-bit decode now runs as vectorized SFPU ops (bitwise AND,
  shift, typecast, subtract) on the Tensix compute engine, not a scalar
  loop on the data-movement RISC-V core - the reader's only job now is
  gathering undecoded raw bytes. Found and fixed a real concurrency
  deadlock along the way (an activation-tile circular buffer sized for the
  old design blocked forever once a matmul call needed more than one
  packing superblock - see §24 for the full mechanism). Verified correct
  (dense random + an exact sparse diagnostic, both matching §23's own
  numbers almost to the bit) and measured ~6x faster: the same
  `K=2560,N=2560` call that took ~930s under §23 now completes in ~154s.
  Still not practical for full decode-step offload (~7 such calls per
  layer x 30 layers), so the `M%32==0` gate from §21/22 stays in place -
  but this is a real architectural win verified end to end, and since
  ttsim's absolute numbers are known not to represent real hardware
  timing, trading scalar RISC-V work for vector SFPU work plausibly
  matters more on real silicon than these numbers alone suggest.
- **Multi-core dispatch - another ~10-15x**: `PORTING_PLAN.md` §25 went
  from §24's single core (`CoreCoord core({0,0})`, every other core on the
  chip idle) to spreading the N-tile dimension across the device's
  available cores via the stock `split_work_to_cores` utility - each core
  computes complete output tiles for its own slice of N independently, no
  cross-core communication. Verified correct (dense random + an exact
  sparse diagnostic spanning 12 N-tiles/cores, 0/1769472 mismatches) and
  measured faster again: the `K=2560,N=2560` call that took ~154s under
  §24 now completes in ~9.9s (~15.5x), and `K=1280,N=1280` dropped from
  ~44s to ~4.3s (~10.3x) - both still matching prior sections' error
  metrics exactly. Cumulative effect since §23: roughly 94x faster.
  Available parallelism (140 worker cores in this Blackhole config) is not
  fully realized in wall-clock terms - measured and explained in §28
  (below), not a code inefficiency to fix. The `M%32==0` gate stays in
  place for now, though the combined §24+§25 win may be worth revisiting
  that decision over.
- **Sub-linear multi-core scaling, explained**: `PORTING_PLAN.md` §28
  measured it directly (phase-by-phase timing, then removed once the data
  was in) rather than leaving §25's question open. All wall-clock time is
  in device dispatch, not host-side prep (every host phase measured under
  5ms). The per-core cost (~1000ms fixed + ~24ms per K-tile of genuine
  SFPU decode work) does not shrink as more cores run concurrently -
  going from 1 to 80 cores at `K=2560` grows total time from ~2960ms to
  ~8291ms, when independent, non-communicating cores under real hardware
  parallelism should cost about the same as one. No bug found in this
  backend's host or kernel code (kernel logic unchanged from §24, no
  cross-core coupling by design); the most plausible explanation is that
  ttsim itself doesn't give multiple simulated cores genuine wall-clock
  parallelism - consistent with, and now backed by measurement of, this
  guide's standing point that ttsim's absolute numbers aren't a real-
  hardware performance proxy. No code change made; expected to look
  different on real silicon.
- **`M%32==0` gate lifted - real decode-step offload, but a new blocker
  found**: `PORTING_PLAN.md` §26 acted on §25's suggestion. Verified
  correct at every M tested (including exact diagnostics at M=1 and M=5 -
  genuinely new coverage, since no §24/§25 test had ever exercised the
  M-padding logic's real zero-fill path before) and confirmed via
  `GGML_SCHED_DEBUG` that decode-step `MUL_MAT` now actually reaches
  `TT_METALIUM0` (2730 nodes on-device vs 673 CPU in one partial run,
  versus §21's finding of 0% before). Real per-shape M=1 timing: ~3-19s
  per projection depending on shape, ~62s/layer, ~31 min/token if fully
  offloaded - not fast, but no longer "many hours." **However**: a real
  end-to-end run hit a separate, pre-existing gap
  (`dst must not be a view`, sec 16) that this change was the first thing
  to actually surface - see §27.
- **dst-view assert root-caused and fixed; a new, deeper blocker found**:
  `PORTING_PLAN.md` §27 diagnosed §26's blocker directly (debug
  instrumentation on a real run, not guesswork): not an actual view, but
  ggml's graph allocator recycling a dead tensor's address for a new,
  differently-sized one without always re-invoking this backend's
  `init_tensor` - `ggml_backend_ttnn_locate()` now self-heals that by
  size-mismatch detection. Verified no regression, and the real
  `llama-cli -ngl 99 -n 1` run went from aborting in 14s to running for
  **93 minutes** - further into real computation than anything in this
  port's history - before hitting a *different*, not-yet-root-caused
  `SIGSEGV` inside tt-metal's `read_shard_from_device`. Each reproduction
  costs ~90+ minutes, so not chased further yet - see the known issue in
  §6 and the full story in §27.
- **L1 capacity**: fixed (`PORTING_PLAN.md` §20) - the ternary matmul kernel
  (Option B) now streams one N-tile's packed weight row-block at a time
  instead of keeping the whole blob resident, bounding L1 usage to 20-54KB
  per weight tensor on this model regardless of tensor size (previously up
  to 4.3MB for the largest FFN projections - more than a Blackhole core's
  L1 budget, so the old whole-blob design would not have fit outside
  ttsim). The tradeoff is more DRAM traffic (the weight chunk is re-fetched
  once per M-tile); irrelevant while performance work stays deferred to
  real silicon (§7 below).
- **One `MeshBuffer` per root tensor, always accessed whole** - two
  tt-metal bugs (miscalculated host offset, "interior" writes/reads landing
  at offset 0) prevent direct access to a sub-region of a buffer. Views
  (`view_src != NULL`) are supported (since `PORTING_PLAN.md` §16) but never
  get their own buffer: they're resolved to their root tensor's buffer plus
  an offset, and that offset is always applied **host-side** (read the
  whole buffer, patch, write the whole buffer back) - never via an
  "interior" device access, which sidesteps both bugs rather than fixing
  them.
- **The `MeshDevice` is never destroyed** (deliberate leak) - a tt-metal bug
  crashes destruction once an op has populated the program cache. No
  consequence for a process that's exiting anyway, but worth knowing if you
  write tests that recreate devices in a loop within the same process.
- These tt-metal bugs are documented as likely present on real silicon too
  (host-side C++ logic, not a simulation artifact) - not yet reported
  upstream.

## 8. Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| `TT_FATAL: Root Directory is not set.` | `TT_METAL_RUNTIME_ROOT` not exported (section 2) |
| `error while loading shared libraries: libtt_metal.so` | `LD_LIBRARY_PATH` doesn't point at `$TT_METAL_HOME/build_Release/lib` |
| No `TT_METALIUM0` in `--list-devices` | Built without `-DGGML_TTNN=ON`, or `find_package(TT-Metalium)` failed at configure time (re-read the `cmake -B` logs) |
| `include could not find requested file: .../build_Release/Metalium.cmake` | A second, broken `tt-metalium-config.cmake` sits at the root of `build_Release/` (no `Metalium.cmake` next to it) and was found first. Pass `-DTT-Metalium_DIR=$TT_METAL_HOME/build_Release/lib/cmake/tt-metalium` explicitly (a CLI `-D` always overrides the cache), and if needed `-DCMAKE_PREFIX_PATH=$TT_METAL_HOME/build_Release` for TT-Metalium's own transitive `find_dependency()` calls (umd, spdlog, tt-logger, fmt...). |
| `TT-Metalium_DIR-NOTFOUND` or a path starting with `/build_Release/...` | `TT_METAL_HOME` wasn't exported in the shell that ran `cmake` - export it first (section 2), in the **same** shell. |
| `error: externally-managed-environment` running `setup_env.py` | System Python protected by PEP 668 (recent Debian/Ubuntu) - use a venv (section 5), not `--break-system-packages`. |
| `GGML_ABORT("pre-allocated tensor (...) ... cannot run the operation (X)")` | A new op that creates/consumes a view isn't in `supports_op()` yet. Same shape of problem as `PORTING_PLAN.md` §12/§16/§19 - check whether the plan already documents this case before starting a fresh investigation. |
| `UnimplementedFunctionality: <opcode>` from ttsim | A gap in the simulated ISA, not a BitNet bug - isolate and report on [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim/issues) |
| `Getting NOC translation status is not supported...` | `tt-umd` too old (missing TTSim fixes) |
| Crash on process exit after a successful inference run | The §11 "MeshDevice destruction" bug - the process is exiting anyway, no impact |

For everything else (SFPI compilation, low-level LLK pytest, supported
architectures), see
[`tt_metal/tt-llk/tests/TTSIM.md`](../../tt-metal/tt_metal/tt-llk/tests/TTSIM.md)
directly.
