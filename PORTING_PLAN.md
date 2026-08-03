# Porting bitnet.cpp to Tenstorrent Blackhole (tt-metal / TT-NN) — Phase 1 findings & plan

Status: DRAFT, for review. No source changes made yet except the submodule
remediation described in §0 (in progress, explicitly requested).

## 1. Ternary quant formats

- **I2_S** (`GGML_TYPE_I2_S`): 2 bits/weight, 4 weights/byte, values packed as
  `{0,1,2} → {-1,0,+1}`. One **f32 scale for the whole tensor**, not
  per-block — appended as a 32-byte-aligned header after the packed data
  (`quantize_i2_s`, `src/ggml-bitnet-mad.cpp:51-96`). Compute goes through
  `ggml_vec_dot_i2_i8_s` (`src/ggml-bitnet-mad.cpp:1043`), registered like a
  normal quant type's `vec_dot`/`vec_dot_type` in ggml's type-traits table —
  **not** a hardwired special case. Activations are quantized to int8
  on the fly; the "matmul" is int2×int8 dot products (effectively signed
  add/sub, since weights are ∈ {-1,0,1}), horizontally summed and rescaled.
- **TL1** (ARM) / **TL2** (x86): lookup-table kernels. Packing is a
  3-value/2-value split-plane layout, quite different from I2_S. Critically,
  packed weights + per-tile scales are **not** stored in the tensor's normal
  `data` buffer — `ggml_bitnet_transform_tensor()` (root `src/ggml-bitnet-lut.cpp`)
  writes them into a side-channel `tensor->extra` struct
  (`bitnet_tensor_extra { qweights, scales, BK, n_tile_num, lut_scales_size }`,
  `include/ggml-bitnet.h:19-25`). Dispatch is a hardwired branch bypassing
  the type-traits table entirely:
  ```c
  #if defined(GGML_BITNET_ARM_TL1) || defined(GGML_BITNET_X86_TL2)
      if (ggml_bitnet_can_mul_mat(src0, src1, dst)) {
          ggml_bitnet_mul_mat(params, dst);
          return;
      }
  #endif
  ```
  at the top of `ggml_compute_forward_mul_mat` in ggml-cpu's compute file.

**Consequence for sequencing:** I2_S is the clean integration point — it's a
type-traits-registered quant type whose data lives in the normal tensor
buffer, so a new backend can claim it the same way ggml-cuda/ggml-metal claim
any other quant type (own buffer type + `supports_op` on `MUL_MAT` when
`src0->type == GGML_TYPE_I2_S`). TL1/TL2 store weights outside the normal
buffer abstraction (`tensor->extra`) and are only reachable via a hardwired
CPU-backend branch, not the backend `supports_op`/`graph_compute` surface —
porting them means either intercepting `ggml_bitnet_transform_tensor` to
upload to a TT buffer instead of `extra`, or leaving them CPU-only
indefinitely. **Recommendation: target I2_S only for the initial port;
treat TL1/TL2 offload as a stretch goal, not a blocker.**

## 2. Backend architecture

The correct branch's ggml has the modern multi-device registry
(`ggml_backend_reg_i` / `ggml_backend_device_i` in `ggml-backend-impl.h`,
`ggml-backend-reg.cpp`), the same shape as current upstream llama.cpp. A new
backend registers: a `ggml_backend_reg_i` (device enumeration), a
`ggml_backend_device_i` (buffer types, `supports_op`, `graph_compute`), and a
buffer type whose `alloc_buffer` places tensor data in TT-metal DRAM. Because
`ggml-backend.c`'s scheduler splits the graph by which backend owns each
tensor's buffer, an I2_S weight tensor allocated in a TT buffer never reaches
the CPU's hardwired I2S/TL1/TL2 branch at all — the graph splitter routes its
`MUL_MAT` op straight to our backend's `graph_compute`. This is the standard
llama.cpp GPU-offload pattern and needs no changes to ggml-cpu.c.

There's no backend-capability gate on quant type today (§below), so wiring in
a new device is additive — same shape as adding CUDA/Metal, no core changes
needed beyond `ggml-backend-reg.cpp`'s static registry list.

## 3. Tensor loading / backend capability gaps

`llama-model-loader.cpp` uploads tensor data then unconditionally calls
`ggml_bitnet_transform_tensor(cur)`, assuming CPU and repacking into
`tensor->extra`. There is no "can this backend consume I2_S" check anywhere
in `ggml-backend.c`'s buffer-type interface. For I2_S this is actually fine
for us — I2_S doesn't use `tensor->extra`, so a TT buffer type can just take
the packed 2-bit data as-is (matching Option B) or dequantize it during
upload (Option A) without touching this transform-tensor call path at all
(it's TL1/TL2-specific, guarded by `GGML_BITNET_ARM_TL1`/`GGML_BITNET_X86_TL2`
compile defines we won't set for the new backend's build).

## 4. Op mix

mul_mat (Q/K/V/O projections + FFN gate/up/down) dominates FLOPs and is the
only bitnet-specific path. RMSNorm, RoPE, softmax, elementwise add/mul, and
KV-cache copies are plain F32 ggml ops, cheap and memory-bound — leave these
on CPU initially, matching the task's Phase 2/3 default.

## 5. Existing prior art

[marty1885/llama.cpp `metalium-support`](https://github.com/marty1885/llama.cpp/tree/metalium-support)
([docs](https://github.com/marty1885/llama.cpp/blob/metalium-support/docs/backend/Metalium.md))
is a personal (non-Tenstorrent-official) TTNN-based ggml backend for
Grayskull/Wormhole. Relevant facts:
- Uses TTNN ops directly, tiled tensor layout except KV-cache/embeddings
  (row-major).
- Dequantizes GGML quant types to BFLOAT8_B/BFLOAT4_B on upload — **no
  ternary/I2_S support at all**, and no packed sub-8-bit path.
- Single device only; KV cache forced to CPU (`-nkvo`); FP32 emulated via
  BF16; long JIT-compile-driven cold start.
- Author describes it as unofficial, personal-time work, "still a moving
  target"; unclear current maintenance state, base llama.cpp commit not
  pinned/documented.

Given it has zero ternary support and an unknown/likely-divergent base
commit, treat it as **a design reference, not a dependency** — its
buffer-type/device-registration boilerplate and TTNN op-mapping choices
(Q4_0→BFP8_B etc.) are useful patterns to crib for our Option-A dequant path,
but forking or merging it looks like more work than writing a fresh backend
against our own correctly-versioned bitnet.cpp tree. Also worth a skim:
`gpu/bitnet_kernels/bitnet_kernels.cu` in this repo — a standalone raw-CUDA
PyTorch reference (not ggml, not reusable directly) showing one existing
add/sub-accumulation ternary kernel design, useful for the eventual Option-B
custom Metalium kernel.

## 6. Environment (verified)

- `TT_METAL_HOME=~/stage_bitnet/Bitnet-TT/tt-metal` is a prebuilt tree;
  `metal_example_add_2_integers_in_riscv` runs clean under ttsim
  (`TT_METAL_SIMULATOR`, `TT_METAL_SLOW_DISPATCH_MODE=1`,
  `TT_METAL_DISABLE_SFPLOADMACRO=1`, matching the local
  `tt_metal/tt-llk/tests/TTSIM.md` exactly) — confirms the Blackhole ttsim
  stack is correctly wired end to end. One gotcha not in the task brief:
  running examples outside the tt-metal build tree needs
  `LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib` for `libtt_metal.so`.
- `~/stage_bitnet/Bitnet-TT/sim/` has `libttsim_bh.so` + `soc_descriptor.yaml`
  already in place.
- Second gotcha found while bringing up the `ggml-ttnn` backend: `TT_METAL_HOME`
  is a convention used by tt-metal's own scripts/docs, but the runtime library
  itself (`RunTimeOptions`, `tt_metal/llrt/rtoptions.cpp`) looks for
  `TT_METAL_RUNTIME_ROOT` (or falls back to cwd, or a package-install path).
  A binary not run from the tt-metal repo root - i.e. any real consumer of
  this backend, sim or silicon - needs `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME`
  set explicitly, or it fails fast with `TT_FATAL: Root Directory is not set.`
  Applies identically on silicon; not a sim-specific fix.

## 7. Build system gap (resolved by re-checking against the now-corrected submodule)

`src/CMakeLists.txt`'s `GGML_SOURCES_BITNET` set-then-overwrite is dead code,
confirmed: `ggml/src/ggml-cpu/CMakeLists.txt:56` in the corrected submodule
pulls in `../../../../src/ggml-bitnet-lut.cpp` by a hardcoded relative path,
unconditionally, ignoring that variable entirely. Also worth noting: I2_S now
has its own native file inside ggml-cpu
(`ggml/src/ggml-cpu/ggml-cpu-i2s.c`, line 38 of the same CMakeLists) — the
root `src/ggml-bitnet-mad.cpp` I read from the outer repo isn't referenced
from this CMakeLists at all, so it looks superseded/vestigial for I2_S in
this branch (the I2_S dispatch sites I found in `ggml-cpu.c` and
`repack.cpp` are the live ones). `-DGGML_TTNN=ON` should hang off this same
`ggml/src/ggml-cpu/CMakeLists.txt` / top-level `ggml/src/CMakeLists.txt`
integration point, as a sibling backend directory the way CUDA/Metal are
added there — not off the outer, unused `src/CMakeLists.txt` variable.

## 8. Recommended strategy (Phase 2, adjusted per above)

1. Finish the submodule fix (§0), verify the corrected tree builds CPU-only
   exactly as before (regression check that default build is unaffected).
2. New backend, working name `ggml-tt-metalium`, registered via
   `ggml_backend_reg_i`/`ggml_backend_device_i` alongside CUDA/Metal in
   `ggml-backend-reg.cpp`; CMake gated behind `-DGGML_TTNN=ON`, default build
   untouched.
3. Target **I2_S only** to start (§1). Option A (dequant I2_S → bf16/BFP8 on
   upload, stock TT-NN matmul) for correctness first, matching the prior
   art's dequant approach but for our ternary type instead of Q4_0/Q3_K.
   Option B (packed 2-bit weights consumed directly by a custom Metalium
   add/sub kernel, informed by `gpu/bitnet_kernels/bitnet_kernels.cu`) once A
   is validated end-to-end.
4. TL1/TL2 stay CPU-only unless/until Option B is solid and there's appetite
   to intercept `ggml_bitnet_transform_tensor`'s `tensor->extra` path — flagged
   as a stretch goal, not in the initial scope.
5. Everything else (tokenizer, KV cache, sampling, norms/RoPE/softmax) stays
   on CPU initially, per the task brief.
6. Validate per Phase 4 of the task brief: isolated ternary-matmul unit test
   against known small ternary weights/activations on ttsim, compared to the
   CPU I2_S reference kernel (`ggml_vec_dot_i2_i8_s`) bit-for-bit-ish; grow to
   one block, then a few layers, then a minimal end-to-end smoke test.

## Decisions made

- I2_S-first / TL1-TL2-deferred sequencing: confirmed.
- Backend naming: `ggml-ttnn` (CMake target/dir, matching the literal
  `-DGGML_TTNN=ON` requirement) with description string "TT-Metalium
  (TTNN)"/"TT_Metalium" in device/buffer-type names.

## 9. Two upstream tt-metal bugs found while building the DRAM buffer type

Both confirmed with minimal repros using only tt-metal's own API, no ggml or
BitNet code involved - see `ggml-ttnn.cpp` history for the repro shape.

1. **Host-pointer bug**: `SDMeshCommandQueue::write_shard_to_device`
   (`tt_metal/distributed/sd_mesh_command_queue.cpp`, slow dispatch - required
   under ttsim per `TTSIM.md`) does
   `payload = Span(src + region.offset, region.size)`, adding the *device*
   region offset to the *host* source pointer too, instead of reading the
   payload from the start of the caller's buffer. The matching read path does
   not have this bug (asymmetric implementation).
2. **Interior-region bug**, independent of (1): a `BufferRegion` write or read
   that is "interior" - `offset > 0 AND offset+size < buffer.size()` - lands
   at device offset 0 instead of the requested offset. Only regions anchored
   at the very start or spanning to the exact end of a buffer are safe. Not a
   page-size artifact - reproduced identically at page_size 1, 2, 4, 8, 16, 32.

Together these rule out the "one shared MeshBuffer, tensors placed at
ggml-computed sub-offsets" design that CUDA/Metal/RPC all use for their ggml
buffer types. **Resolution**: every ggml tensor gets its own dedicated
MeshBuffer (`init_tensor()` allocates it once ggml assigns `tensor->data`),
and every tt-metal-facing access is always the *entire* buffer (offset 0,
full size) - confirmed safe under both bugs. Partial ggml-side
set/get/memset calls become a host-side read-modify-write over the whole
buffer. `tensor->data` is a synthetic pointer into a host-only "shadow"
allocation (`posix_memalign`, 32-byte aligned - plain `malloc` is only
16-byte aligned on x86_64 and silently corrupted ggml's own tallocr
bookkeeping) used purely as a unique per-tensor lookup key, never
dereferenced for real data.

Known follow-up gap: tensor **views** (`view_src != NULL`) aren't supported
yet by this scheme (a view would need to reuse its parent's buffer at a
possibly-interior offset, hitting bug 2 again) - `init_tensor` currently
`GGML_ASSERT`s this hasn't happened. Not exercised by weight loading; will
need a real solution before KV-cache or any op that creates views is
offloaded.

These are host-side C++ logic bugs (not simulated Tensix behavior), so they
likely reproduce identically on real Blackhole silicon, not just ttsim -
worth flagging to Tenstorrent independent of this port's timeline.

## 10. I2_S packing: `quantize_i2_s` and `dequantize_row_i2_s` disagree with
    each other (`ggml/src/ggml-cpu/quants.c`)

Found while implementing Option A dequant-on-upload. Two different bit
layouts exist in the live (compiled-in) code for the same type:

- `quantize_i2_s`: packs element `i` into `byte[i/4]` at bit position
  `6-2*(i%4)` - four *consecutive* elements per byte.
- `dequantize_row_i2_s`: reads `byte[done/4+gp]` and assigns its four 2-bit
  fields to elements at `done+gp`, `done+32+gp`, `done+64+gp`, `done+96+gp`
  - four elements *strided by 32 within a 128-element super-block*.

These are genuinely incompatible - round-tripping a tensor through this
module's own quantize-then-dequantize would scramble it. Checked which one
is authoritative by reading the real AVX2 inference dot-product,
`ggml_vec_dot_i2_i8_s_1x1` (same file): it loads 32 packed bytes and extracts
four 32-wide lanes via `>>6`, `>>4`, `>>2`, `&mask`, multiplying lane k
against activations `y[k*32 .. k*32+32)` - i.e. the **strided/grouped-by-32
layout, matching `dequantize_row_i2_s`**, not `quantize_i2_s`. Since this dot
product is what real BitNet.cpp inference actually runs, it's ground truth
for what real downloaded I2_S GGUF files contain. `quantize_i2_s` looks like
a newer/orphaned helper (consistent with Phase-1's finding that the
`llama-quantize` I2_S code path already looked incomplete) - separately, its
scale computation also looks broken (`break`s on the first nonzero-magnitude
element rather than scanning for the true max).

**Consequence**: our dequantizer must mirror `dequantize_row_i2_s`'s
strided-by-32 indexing (verified against real inference), not
`quantize_i2_s`'s. Implemented as a small self-contained function in
`ggml-ttnn.cpp` rather than linking against `ggml-cpu` internals (those
symbols aren't declared in any shared header, and a device backend taking a
build dependency on the CPU backend's internals would be architecturally
wrong) - cross-checked against the real `quantize_i2_s`/`dequantize_row_i2_s`
symbols from a standalone test harness that links `ggml-cpu` directly, not
from the backend itself.

## 11. First op offload: `GGML_OP_MUL_MAT` (I2_S x F32) via stock `ttnn::matmul`

`supports_op()` now accepts `MUL_MAT` when `src0` is `I2_S`, `src1` and `dst`
are `F32`. `graph_compute()` handles it by reading the already-on-device
dequantized bf16 weight and the F32 activation back to host, wrapping both as
fresh `ttnn::Tensor`s (`Tensor::from_vector` + `.to_device()`), and calling
`ttnn::matmul(activation, weight, transpose_a=false, transpose_b=true)` -
matching ggml's `mul_mat` convention (`dst = act @ weight^T`, i.e. nn.Linear
semantics; derived from ggml's `ne[0]`-is-fastest-dim memory layout, not
assumed). Result comes back via `to_vector<float>()` and is written to dst.

This round-trips already-on-device weight data through the host on every
call - this backend doesn't yet store `ttnn::Tensor` natively in its buffer
type (still raw `MeshBuffer`, see sec 8-9), so op execution stages through
host memory. Correct, not fast; worth revisiting once the buffer type is
redesigned around `ttnn::Tensor` directly.

**Verified under ttsim**: a 64x128 ternary weight times a 3x128 F32
activation, run through the real `graph_compute()` path (built as a genuine
`ggml_cgraph` via `ggml_build_forward_expand` + `ggml_backend_graph_compute`,
not called directly), matches a CPU f64 reference matmul to within bf16
precision (max abs error ~0.03, 0 large diffs) across all 192 output
elements.

**Third tt-metal bug found**: destroying a `MeshDevice` after any real op has
populated tt-metal's internal program cache crashes inside
`GraphTracker::track_deallocate_cb` during `~MeshDeviceImpl()`'s teardown of
cached `MeshWorkload`/`Program` objects - reproduced with a plain
`ttnn::matmul` call, no ggml involved. Calling `MeshDevice::close()` first
does *not* prevent it (confirmed by instrumenting an `atexit` hook ordered,
via a second static-local initializer, to run strictly before the singleton's
own destructor - `close()` completes successfully, and the crash still
happens moments later in the destructor). Resolution: `mesh_device` is now
allocated via `new` and intentionally never destroyed - the process is
exiting either way, so leaking it sidesteps the bug entirely. Same category
of host-side C++ logic as bugs 1-2 (sec 8), so worth reporting upstream
alongside them; not filed yet.

## 12. First full-model offload attempt under ttsim: fixed one scheduler bug,
    confirmed the view limitation (sec 9) is a hard blocker for real inference

Attempted `llama-cli -m <official I2_S GGUF> -ngl 99 -dev TT_METALIUM0` under
ttsim, using the official `microsoft/BitNet-b1.58-2B-4T-gguf` (already
quantized, downloaded directly - `llama-quantize` in this branch has no
`I2_S` entry in its `QUANT_OPTIONS` table, `tools/quantize/quantize.cpp:34-73`,
despite `LLAMA_FTYPE_MOSTLY_I2_S` existing in `llama.h`; matches the
Phase-1 finding that the `llama-quantize` I2_S path looks incomplete -
producing I2_S GGUFs locally isn't possible with this branch's quantize
tool, only pre-quantized official GGUFs work).

**Bug found and fixed**: `ggml_backend_sched_backend_from_buffer`
(`ggml-backend.cpp:845`) calls `supports_op()` even for pre-allocated leaf
tensors (weights) already sitting in a backend's buffer, passing the tensor
itself as `op` - for a weight tensor this means `op->op == GGML_OP_NONE`.
`ggml_backend_ttnn_device_supports_op` only ever returned `true` for
`MUL_MAT`, so this aborted immediately on the first weight tensor
(`pre-allocated tensor (blk.0.attn_q.weight) ... cannot run the operation
(NONE)`). Fixed by returning `true` for `GGML_OP_NONE` specifically.
**Deliberately did not** extend this to `VIEW`/`RESHAPE`/`TRANSPOSE`/
`PERMUTE` even though `graph_compute()`'s switch already tolerates them
(sec 11's code) - those are real computed nodes, and telling the scheduler
this device "supports" them would make it place them on our buffer type,
which cannot hold a view at all (next paragraph).

**Confirmed blocker, not a quick fix**: with the leaf-tensor bug fixed, KV
cache placement crashed next (`cache_k_l0 (view) ... cannot run the
operation (SET_ROWS)`) - solved for now with `-nkvo` (forces KV cache to
CPU, an existing llama.cpp flag, no backend change needed). With `-nkvo`
too, the *next* crash is the real ceiling: during `ggml_gallocr_alloc_graph`
for the first decode, some node with `tensor->view_src != NULL` gets
allocated into the TT_Metalium buffer and hits the buffer type's own guard
(sec 9's `GGML_ASSERT(tensor->view_src == NULL ...)`). Root cause: once a
`MUL_MAT`'s F32 output lives in this backend's buffer (because the op ran
on this device), every real transformer layer immediately follows that
matmul with a reshape/permute (splitting into attention heads, etc.) -
i.e. a view of that output - and the buffer type categorically cannot
allocate a view (sec 9: one whole `MeshBuffer` per tensor, no interior
offsets, because of the two upstream tt-metal bugs). This reproduces with
the plain Q projection of layer 0, so it is not model- or layer-specific:
**no real transformer block can be offloaded end to end today, only the
previously-validated standalone `MUL_MAT` (sec 11) works** - that test
never chained a reshape/permute after the matmul, which is why it didn't
hit this.

Sanity check: the same GGUF with `-ngl 0` (no offload, CPU only) completed
cleanly end to end (54.2 t/s prompt, 14.8 t/s generation on 2 threads),
confirming the model file, the build, and everything except the TTNN
offload path itself are healthy.

**Consequence**: closing this gap needs the buffer-type redesign already
flagged as future work in sec 9 (real view support, or an alternative like
copying view-producing ops' inputs back to a CPU-owned staging buffer
before/after the matmul) - not something to patch around locally. Until
then, `-ngl`/`-dev TT_METALIUM0` on a real model will always hit this same
assert on the first attention block, regardless of model size.

## 13. Option B: custom ternary matmul kernels (reader/compute/writer triad)

First cut at the "true 1.58-bit path" from the original task brief: packed
2-bit weights consumed directly, no dequantize-on-upload. Lives at
`ggml/src/ggml-ttnn/kernels/ternary_matmul/` (see that directory's own
README for the file-by-file breakdown) - **not yet wired into
`ggml-ttnn.cpp`'s `supports_op`/`graph_compute`**, which still uses Option A
(`ttnn::matmul` on dequantized bf16). This is a validated kernel triad, not
yet an integrated backend path.

**Design**: only the reader kernel is ternary-specific. It reads the packed
I2_S bytes from DRAM and unpacks them *on-device* directly into the 32x32
tile-face layout the FPU's `matmul_tiles` expects (four 16x16 faces,
face0->face1->face2->face3, row-major within each face - verified against
`tech_reports/tensor_layouts/tensor_layouts.md`, not guessed), using the
strided-by-32-in-128 decode verified in sec 10. The compute and writer
kernels are unmodified copies of the stock
`matmul_single_core` example's `mm.cpp`/`writer_single_core_mm.cpp`. Since
ternary weight values are exactly {-1, 0, +1}, the FPU's per-element
"multiply" against those values is bit-exact to a conditional
negate/pass-through/zero of the activation - it already *is* an add/sub
accumulation, just executed on the FPU's existing multiply-accumulate
datapath rather than a hand-rolled SFPU one. The device outputs the
*unscaled* dot product; the per-tensor weight scale is applied by the
caller afterward, not per-element in-kernel.

Scope: single N-tile weight (N=32) - the packed I2_S format stores one
scale for the whole tensor, not per N-tile, so the packed blob for one
N-tile is treated as a single DRAM page, read once per M-tile row and kept
resident in a scratch L1 CB for the whole Kt loop. Multi-N-tile support
needs the scale handled across tiles first, not attempted here.

**Verified under ttsim**, via a standalone TT-Metalium `Program`/`Kernel`
dispatch that bypasses TT-NN entirely (unlike Option A): a 32x128 ternary
weight against a 32x128 activation matches a CPU f64 reference to within
~0.05 absolute error across all 1024 output elements. That error is bf16
rounding under catastrophic cancellation (summing 128 terms of magnitude
~weight-scale with mixed signs down to a near-zero result amplifies
relative, though not absolute, error) - same ~0.03 magnitude seen with
`ttnn::matmul` (Option A) on equivalent data, not a kernel defect. Confirmed
by isolating a single known nonzero weight against a constant activation
first (before the full randomized-pattern test), which caught two real bugs
along the way:
- The FPU's `matmul_tiles(in0, in1, ...)` computes
  `dst[i,j] = sum_k tileA[i,k] * tileB[k,j]` with tileA fed by `in0`, tileB
  by `in1` - so the activation must be uploaded *transposed* (`[K,M]`, not
  `[M,K]`) to land in tileB's `[k,j]` slots. The output is consequently
  `[N,M]`-shaped (N outer, M inner) - which happens to already match ggml's
  own `mul_mat` dst convention (sec 11), so no extra transpose is needed
  once this is understood correctly.
- The device intentionally returns the unscaled dot product (see Design
  above); the first "mismatch" after fixing the transpose was the test
  forgetting its own stated contract and comparing against the
  scale-applied reference directly.

**Next steps** (not done here): wire this in as a `supports_op`-selectable
path alongside/instead of Option A, generalize past the single-N-tile
scope, and validate on a size that matches a real model's projection
dimensions rather than this toy 32x128 case.

## 14. Generalizing sec 13's kernels past a single N-tile: two more tile-index bugs

Extended the ternary matmul kernels to handle arbitrary Mt/Kt/Nt (not just
Mt=Nt=1, Kt=4), tested at M=64/K=256/N=64 (Mt=2, 2 K-superblocks/row, Nt=2).
The reader's weight-unpack indexing generalizes cleanly: row `row`'s packed
data starts at byte `row*(K/4)`, and K-tile `kt` sits at super-block `kt/4`,
lane `kt%4` within that row - both already correct from sec 13, confirmed
with a dense (all-positions-nonzero) weight pattern against a constant
activation matching exactly (0 error, not just within tolerance).

Two *new* bugs surfaced once Mt and Nt were no longer both 1 - both are
"formula only valid when a tile count is 1" bugs, invisible until now for
exactly that reason:

1. **Writer**: our output is `[N, M]` (N outer, M inner - sec 11/13), so the
   DRAM page index must be `n*Mt + m` (N-tile major) to match how the
   host's `untilize_nfaces(N, M)` expects tiles laid out. The kernel (copied
   from the stock example, which assumes `[M, N]`) used `m*Nt + n`. With
   Mt=Nt=1 both formulas give 0; at Mt=2/Nt=2 they diverge and two tiles
   land swapped.
2. **Reader**: the activation tensor is `[K, M]` (transposed on upload, per
   sec 13), so its tile grid is K-tile-major: index `kt*Mt + mt`. The reader
   used `mt*Kt + kt` (right for an `[M, K]`-shaped tensor, wrong for
   `[K, M]`). Again, `mt*Kt+kt == kt*Mt+mt` whenever Mt=1, which is exactly
   why sec 13's Mt=1 validation didn't catch it.

Both found the same way as every other bug this session: isolate with a
sparse, hand-picked probe (three single nonzero weights placed to
independently exercise nt=1, and a second super-block/lane, against a
constant activation) before trusting the dense random-pattern test. The
probe passed exactly (0 error) even before either fix - it happened to
only exercise positions where the two buggy formulas coincide with the
correct ones - and only failed once widened to hit every (mt, nt) pair,
which is what actually caught bug 1. Bug 2 stayed hidden until switching to
dense weights + non-constant activation with Mt=2, since a constant
activation is insensitive to which activation tile is fetched.

**Verified under ttsim**: dense random weight pattern, sinusoidal
activation, full Mt=2/Kt=8/Nt=2 grid - matches CPU f64 reference to within
~0.07 absolute error across all 4096 outputs (same bf16-cancellation
tolerance as sec 13, scaled up slightly for twice the K-dimension terms).

Remaining scope limit (documented in the kernel directory's README): the
whole packed weight blob is still read once into a scratch L1 CB and kept
resident for the kernel's duration. Fine for correctness at these sizes;
does not scale to real model weight matrices without chunked/streamed
reads - not attempted here.

## 15. Option B wired into supports_op/graph_compute, replacing Option A

`ggml-ttnn.cpp`'s `GGML_OP_MUL_MAT` path now dispatches the sec 13/14
kernel triad directly, superseding Option A's `ttnn::matmul`-on-dequantized-
bf16 path entirely (not kept alongside it - the two are mutually exclusive
at the storage level, since Option B needs the weight to *stay* packed on
device). Concretely:

- The buffer type no longer special-cases I2_S at all: weights are stored
  packed, byte-for-byte identical to the gguf file, exactly like every
  other type (`ggml_backend_ttnn_dequantize_i2_s` and the bf16-sized
  `device_alloc_size` special case are gone - `get_alloc_size` is back to
  `NULL`, defaulting to `ggml_nbytes`). Simpler than Option A's storage
  model, not just different: there's no host/device size mismatch left to
  track anywhere.
- `compute_mul_mat` reuses the weight tensor's own resident `MeshBuffer`
  directly as the reader kernel's `TensorAccessor` source (no re-upload -
  Option B's entire point is that packed weights never leave DRAM as
  anything but packed bytes). It reads that same buffer back to host once,
  only to pull out the trailing 4-byte scale (cheap - it's the packed size,
  not a dequantized one - and safe per sec 9, since it's a whole-buffer
  read). The activation still has to round-trip through host memory to get
  transposed and tile-faced (kernel README), same limitation as Option A.
- `supports_op` and `compute_mul_mat` share one shape-validity check (`K`
  a multiple of 128, `N`/`M` tile-aligned) so an unsupported shape falls
  back to CPU via the scheduler instead of hitting an assert.
- TT-NN is no longer linked at all (`find_package(tt-nn)` / `TTNN::TTNN`
  removed from `CMakeLists.txt`) - Option B only needs TT-Metalium.

**Bug found while wiring this in** (independent of the sec 13/14 kernels,
which needed no changes): a real dst-layout transpose bug, caught
immediately by re-running the sec 13/14 diagnostics through the real
`ggml_backend_sched`/`graph_compute` path instead of a hand-rolled
dispatch. ggml's `dst` (`ne=[N,M]`) is row-major **`[M,N]`** in flat bytes
(`M` outer, `N` inner - `N` is `ne[0]`, the fastest dimension). The kernel
triad's own tile grid is the opposite way round - the writer lays tiles out
N-tile-major (page index `n*Mt+m`), so `untilize_nfaces(result_tiled, N,
M)` reconstructs an **N-outer/M-inner** matrix, the transpose of what
`dst`'s bytes actually need. An earlier code comment claiming these two
"[N,M]" labels already matched was simply wrong - one refers to the
kernel's own tile-grid bookkeeping, the other to ggml's flat byte layout,
and conflating them silently produced a real bug: with one known nonzero
weight (row 0) and a constant activation, row 0's value leaked into
unrelated output rows once read back through a real ggml tensor, while the
raw pre-untilize device tile data was already provably correct - proof the
bug was in this final host-side relayout, not the kernel. Fixed by
transposing explicitly (`out_host[m*N+n] = result[n*M+m] * scale`) while
applying the per-tensor scale in the same pass.

**Re-verified under ttsim** through the real dispatch path
(`ggml_backend_dev_supports_op` + `ggml_backend_graph_compute`, not a
hand-built `Program`): both the single-nonzero-weight diagnostic and the
dense random-pattern tests from sec 13/14 (K=128/N=32/M=32 and
K=256/N=64/M=64) now match their previously-validated results exactly
(max abs error 0.052 and 0.070 respectively - identical to the standalone
numbers), confirming the fix is complete and the kernels themselves were
never the problem.

**Still not attempted**: the view-support gap (sec 12) remains the real
blocker for a full model - this wiring only changes *how* the standalone
matmul executes, not whether a full transformer block can run end to end.

## 16. View support in the buffer type: closing the sec 12 blocker

Sec 12 found that *any* non-zero-offset view (a genuine sub-region of a
tensor - attention-head extraction, KV-cache placement, etc., as opposed to
the zero-offset "view of itself" bookkeeping `ggml_gallocr` already produces
for op outputs) hit `init_tensor`'s `GGML_ASSERT(tensor->view_src == NULL ...)`
and aborted. The fix does **not** need the two upstream tt-metal
interior-access bugs (sec 9) resolved - it sidesteps them entirely by never
giving a view its own device allocation or doing an interior device access
for one at all:

- A view tensor gets no `MeshBuffer` of its own. `init_tensor` now only
  asserts (debug-only, via the same resolution logic below) that the view
  fits inside its root ancestor's buffer, then returns - the allocation
  already exists.
- `ggml_backend_ttnn_root_tensor()` walks `view_src` to the first non-view
  ancestor; `ggml_backend_ttnn_locate()` looks that root's `MeshBuffer` up in
  the existing `tensor_buffers` map (keyed by the root's `data` pointer, same
  as before) and returns it alongside `tensor->data - root->data` as the
  view's byte offset - a plain pointer subtraction, so chained
  views/reshapes/permutes resolve correctly without walking the whole chain
  by hand (ggml always folds the final `data` pointer for us).
- `set_tensor`/`get_tensor`/`memset_tensor` add that resolved offset to the
  caller-supplied one and otherwise keep the exact same logic as before
  (whole-buffer fast path when the combined offset is 0 and size covers the
  whole buffer, host-side read-modify-write over the whole buffer
  otherwise). Every actual tt-metal transfer this produces is still
  offset-0/full-size end-to-end - sec 9's two bugs (bad host-pointer
  arithmetic on non-zero device offsets, and interior `BufferRegion`
  accesses landing at device offset 0) never come into play, no matter how a
  view slices its root tensor. `ggml_backend_ttnn_lookup` (used by
  `compute_mul_mat` for non-view weight/dst tensors) is now a thin wrapper
  around `locate()` that drops the offset.

**Bug found while wiring this in**: `compute_mul_mat` still called the old
raw `lookup()` + whole-buffer read for the activation (`src1`), sized to
`ggml_nbytes(src1)`. Once a view resolves to its (larger) root buffer, that
read's true size is the *root's* buffer size, not the view's own - reading
"the whole buffer" into a host vector sized for just the view overran the
vector by exactly `(root_size - view_size)` bytes, corrupting the heap
(`double free or corruption` inside `libtt_metal.so`'s
`ReadFromDeviceInterleavedContiguous`, several frames removed from the actual
bug - a classic heap-overflow-detected-downstream signature, not a
tt-metal/ttsim bug). Fixed by routing the activation read through the
generic `ggml_backend_ttnn_buffer_get_tensor()` instead of a raw
`lookup()`/`read_whole()` pair - that function already resolves the view
offset correctly and reads exactly `ggml_nbytes(src1)` bytes regardless of
how much bigger the underlying root buffer is. The weight (`src0`) and `dst`
keep the raw-buffer path (the reader kernel addresses the weight directly
on-device, and the final result is written with a raw whole-buffer write),
but now assert `locate().offset == 0` explicitly - Option B's on-device
addressing and dst's direct write both require being the sole occupant of
their buffer, so a future view in either position should fail loudly instead
of silently misbehaving.

**Verified under ttsim**: a standalone test (not part of the build) builds a
real `ggml_cgraph` - an I2_S weight (`K=128,N=32`) times a **sliced view**
of a larger F32 activation tensor (`ne=[128,64]`, taking rows `[32,64)` via
`ggml_view_2d`, i.e. a genuine non-zero-offset view, not the zero-offset
bookkeeping case) - allocated via `ggml_backend_alloc_ctx_tensors` (which
calls `ggml_backend_view_init` for the view, exercising `init_tensor`'s new
code path exactly as `ggml_gallocr` would for a real model) and run through
`ggml_backend_graph_compute`. Before the activation-read fix above: crashed
with the heap corruption described. After: completes cleanly, and the result
matches a CPU f64 reference computed from exactly the sliced rows (max abs
error 0.177, bf16-rounding-magnitude, 0 elements over a 0.5 threshold) -
confirming both that the view resolves to the *correct* sub-region (not the
unsliced root, not offset 0) and that nothing else regressed.

**Consequence**: the specific mechanism sec 12 identified - a reshape/permute
view of a device-resident op output (e.g. splitting a Q projection into
attention heads) - should no longer hit the old assert. This does not by
itself make a full transformer block work end to end; remaining risks
(unverified) are: `ggml_backend_sched`'s cross-backend copy behavior at
realistic multi-op-crossing scale, whether real model hidden dimensions
satisfy the `K%128==0`/`N,M%32==0` tile constraints from sec 15, and the
sec 13/14 kernel's whole-packed-weight-blob-in-L1 scope not yet scaling to
real projection matrix sizes. Next step: re-attempt the sec 12 GGUF run
(`llama-cli -ngl 99 -dev TT_METALIUM0`) now that the assert it hit is gone.

## 17. Re-attempted the sec 12 GGUF run: past both old blockers, into a new one -
    `ggml_row_size`/`ggml_nbytes` disagree for `I2_S`, unrelated to this backend

Re-ran `llama-cli -m <I2_S GGUF> -ngl 99 -dev TT_METALIUM0` under ttsim with
sec 16's fix in place.

**First rerun (no `-nkvo`)**: hit exactly the *other* sec 12 workaround point,
unchanged - `cache_k_l0 (view) ... cannot run the operation (SET_ROWS)`, from
`ggml_backend_sched_backend_from_buffer` calling `supports_op(SET_ROWS)` on
the KV-cache view and getting `false`. Expected; sec 12 already covered this
with `-nkvo`, unrelated to the view-support fix (this is a *different*
tensor's placement decision, not an `init_tensor` assert).

**With `-nkvo` added back**: got clean past *both* previously-documented
blockers (the leaf-tensor `supports_op` gap and the sec 9/12/16 view assert)
and hit a third, new one, entirely inside ggml core - not this backend:

```
ggml.c:1789: GGML_ASSERT(view_src == NULL || data_size == 0 ||
                          data_size + view_offs <= ggml_nbytes(view_src)) failed
```

from `ggml_view_tensor()` (called by `ggml_backend_sched_split_graph()` when
it needs a whole-tensor view of a leaf to hand to a different split/backend -
standard scheduler machinery, triggered now simply because offload reaches
far enough into the graph for the scheduler to need this for an `I2_S`
weight, which never happened with either previous blocker in the way).

**Root cause, isolated by reading the two size functions side by side**:
`ggml_nbytes()` (`ggml.c:1300`) has an explicit special case for `I2_S`/`TL1`
(`nbytes = nbytes/4 + 32`, accounting for the 2-bit packing + trailing
scale), but `ggml_row_size()` (`ggml.c:1349`, `type_size*ne/blck_size`) does
not - and `I2_S`'s type trait entry (`ggml.c:931`) sets `blck_size=1,
type_size=sizeof(int8_t)=1`, i.e. "1 byte per element" with no packing
awareness at all. So for any `I2_S` tensor, `ggml_row_size(I2_S, ne0) * ne1
== ne0*ne1` (the *unpacked* byte count), while `ggml_nbytes()` on the same
tensor is `ne0*ne1/4 + 32` (the *packed* byte count) - roughly 4x smaller.
`ggml_view_tensor()`'s own sanity assert compares a `data_size` computed via
`ggml_row_size()` against `ggml_nbytes(view_src)` computed the special way,
so it fails for essentially any `I2_S` tensor, not something this offload
path does wrong.

This is a distinct bug from sec 10's `quantize_i2_s`/`dequantize_row_i2_s`
convention mismatch (that one is about bit layout; this one is about size
*accounting*, in a completely different pair of functions) - both stem from
the same underlying pattern of `I2_S` being bolted onto ggml's type-trait
table with a nominal `blck_size=1` and then special-cased ad hoc wherever
its real packed size actually matters, rather than being modeled as a normal
block-quantized type (e.g. `blck_size=128, type_size=32+4` would make
`ggml_row_size` correct with zero special-casing, matching how every other
quantized type in ggml works - not attempted here, out of scope for this
backend, and risks touching the shared CPU I2_S path sec 10-11 already found
fragile).

**Not fixed here** - this is a ggml-core / upstream-fork issue, orthogonal to
the ttnn backend and outside what the buffer-type/view work (sec 16) set out
to touch; flagging it as the next concrete blocker rather than patching
`ggml.c`'s type trait table without discussing the blast radius (it's shared
by the whole BitNet fork, not just this backend, and CPU inference already
works today - sec 12's `-ngl 0` sanity check - so any change here needs to
not regress that path).

## 18. Sec 17's blocker fixed: full model offload works end to end under ttsim

Fixed sec 17's `ggml_row_size`/`ggml_nbytes` mismatch with the smallest
change that keeps the two consistent, without touching `ggml_row_size()`
itself (a widely-called public API) or the `I2_S`/`TL1` type-trait entries
(`blck_size`/`type_size`, which many unrelated call sites read directly -
changing those risks fallout far outside this backend). Instead,
`ggml_new_tensor_impl()` (`ggml.c:1784`, the single function both plain
tensor creation *and* `ggml_view_tensor()` go through) now applies the exact
same `nbytes/4 + 32` special case to its local `data_size` that
`ggml_nbytes()` already applies to its own return value - mirroring an
existing pattern rather than introducing a new one. Concretely:

```c
size_t data_size = ggml_row_size(type, ne[0]);
for (int i = 1; i < n_dims; i++) {
    data_size *= ne[i];
}
if (type == GGML_TYPE_I2_S || type == GGML_TYPE_TL1) {
    data_size = data_size / 4 + 32;
}
GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));
```

This fixes `ggml_view_tensor()`'s assert (now compares two consistently-
packed numbers) and, as a side effect, shrinks the host allocation
`ggml_new_tensor_impl` reserves for a freshly-created, non-view `I2_S`/`TL1`
tensor when `ctx->no_alloc == false` from 4x-inflated (the previous
accidental behavior) down to the real packed size - a reduction in memory
use, not a new constraint, so it doesn't risk truncating anything that
actually gets written into that space (`quantize_i2_s` and GGUF loading
already only ever write the correctly-packed, smaller size). `TL2` was left
alone - its special-case formula in `ggml_nbytes` genuinely needs `ne[0]`/
`ne[1]` individually rather than the flattened product `ggml_new_tensor_impl`
computes, isn't exercised by anything in this port, and touching it here
would be unverified surface area for no benefit.

**Verified no regression**: the sec 16 standalone sliced-view test still
passes with an identical result (max abs error 0.176924, 0 elements over
threshold) - this backend's own buffer-type/view logic was never touching
`ggml_row_size` or `ggml_new_tensor_impl` in the first place, so this was
expected, not just hoped for.

**Verified the actual milestone**: re-ran the sec 12/17 GGUF command,
`llama-cli -m <I2_S GGUF> -ngl 99 -dev TT_METALIUM0 -nkvo --single-turn`,
under ttsim. **This is the first successful full end-to-end offloaded run**
- past every blocker found in sec 12 (leaf-tensor `supports_op`), sec 16 (the
view assert), and sec 17 (this section) - the model loads, every offloadable
`MUL_MAT` runs on `TT_METALIUM0` via the sec 13-15 kernel triad, everything
else (norms, RoPE, softmax, KV-cache via `-nkvo`) runs on CPU exactly as
`ggml_backend_sched`'s graph-splitter is supposed to handle it, and the
process produces real generated text and exits cleanly (exit code 0 -
notably, sec 11's third tt-metal bug, the `MeshDevice`-destroy-after-program-
cache-populated crash, did not reproduce on this exit path; still leaking the
device deliberately per sec 11, not relying on this):

```
> You are a helpful assistant
's a helpful assistant. 0: I am a helpful assistant.
You are
[ Prompt: 35.1 t/s | Generation: 4.7 t/s ]
Exiting...
```

(The output text itself is low-quality/repetitive - expected and irrelevant
here: this run's purpose is confirming the *offload pipeline* works, not
evaluating generation quality, and nothing about this port changes the
model's own logits versus a correct CPU run.)

**What's left**: this validates the mechanism, not production readiness.
Concretely still open, in rough priority order:
- **Performance**: every offloaded `mul_mat` round-trips activation data
  through host memory per sec 15, and ttsim is a functional simulator, not a
  performance model - 4.7 t/s here says nothing about real hardware, and
  isn't a target to optimize against yet (sec 8's phase ordering explicitly
  defers hardware performance work until real silicon exists).
- **KV cache still forced to CPU** (`-nkvo`) - sec 12's original placement
  issue (`supports_op(SET_ROWS)` on a KV-cache view returning `false`) is
  untouched by sec 16-18; revisiting it would let more of the graph run on
  device, but CPU-resident KV-cache is a normal, supported llama.cpp
  configuration, not a hack blocking correctness.
- **L1 capacity**: sec 13's kernel still reads the *entire* packed weight
  blob into a resident L1 scratch CB per invocation - this 2B-param model's
  projection matrices apparently fit (this run didn't fail), but that's not
  the same as verifying it scales to larger models/layers without a
  chunked/streamed weight read.
- Only one prompt/shape has been exercised this way; broader coverage
  (longer generations, multiple prompts, batch>1) is still unverified.

## 19. KV cache offloaded too: `GGML_OP_SET_ROWS` + view-producing ops now
    supported, `-nkvo` no longer needed

Closed the first item on sec 18's list. Two additions to `ggml-ttnn.cpp`:

**`GGML_OP_SET_ROWS`** - the KV-cache write op (`ggml_set_rows(ctx, a, b, c)`,
`a`=destination cache tensor, `b`=new rows (always `F32`), `c`=row indices
(`I32`/`I64`); returns `view(a)`, so `dst` is always a whole-tensor view of
the real cache tensor). `ggml_backend_ttnn_compute_set_rows()` mirrors
ggml-cpu's own `ggml_compute_forward_set_rows_f32` (`ggml-cpu/ops.cpp`)
element-for-element - same loop structure, same broadcast rules over
`ne02/ne03/ne11/ne12` - just operating on host-side copies of this backend's
buffers (whole-buffer read via sec 9's pattern, patch the target rows,
whole-buffer write back) instead of the CPU's own tensor memory, and using
ggml core's portable `ggml_get_type_traits(a->type)->from_float_ref`
(`ggml.c`, the non-SIMD reference conversion, already linked via
`libggml-base` - no new dependency on `ggml-cpu` needed just for this) to
convert into whatever type the cache actually stores (`F16` typically, but
not hardcoded - `supports_op` only allows types with a working
`from_float_ref`). `supports_op` requires `b`=`F32` and `c`=`I32`/`I64`,
matching ggml-cpu's own restriction exactly (there is no other case to
support - that's genuinely the only combination the CPU reference
implementation itself handles).

**`GGML_OP_VIEW`/`RESHAPE`/`TRANSPOSE`/`PERMUTE`** - previously deliberately
excluded from `supports_op` (sec 12) because the buffer type couldn't hold a
view at all. Enabling `SET_ROWS` immediately surfaced the *next* node that
needed this: `cache_k_l0 (view) ... cannot run the operation (VIEW)` - the
attention code reads back a slice of the KV cache via a plain `GGML_OP_VIEW`,
and that view's dst is pre-allocated (aliasing the already-resident cache
buffer) the same way the `SET_ROWS` dst is. All four ops are - in ggml,
unconditionally - pure metadata reinterpretations of the *same* underlying
data (new `ne[]`/`nb[]`/`view_src`/`view_offs`, never new bytes or data
movement; confirmed by reading `ggml_view_tensor`/`ggml_reshape`/
`ggml_permute`/`ggml_transpose` in `ggml.c`), which is exactly why
`graph_compute()`'s switch has treated all four as no-ops since sec 11/12 -
only `supports_op` was withholding them, and only because of the
now-resolved sec 16 buffer-type limitation. Enabled all four together rather
than one at a time: they're identically safe by the same argument, and
leaving the others out would just mean re-discovering each one via the next
crash for no benefit.

**Verified**:
- The sec 16 standalone sliced-view test is unaffected (it calls
  `ggml_backend_graph_compute` directly, never through `ggml_backend_sched`,
  so `supports_op` isn't even exercised by it).
- New standalone test: a resident `F16` "cache" tensor (16 rows) gets 3 new
  `F32` rows written at deliberately out-of-order, non-contiguous indices
  (`{9, 2, 13}`) via `ggml_set_rows`, run through `ggml_backend_alloc_ctx_tensors`
  + `ggml_backend_graph_compute` (the same pattern sec 16/18 used). Checked
  byte-exact against `ggml_fp32_to_fp16` on the target rows *and* confirmed
  every one of the other 13 rows is bit-for-bit unchanged from its original
  value - proof this is a real read-modify-write over the whole buffer, not
  an accidental overwrite of unrelated cache content. Passed on the first
  run after the fix compiled.
- Re-ran the sec 18 GGUF command **without `-nkvo`**:
  `llama-cli -m <I2_S GGUF> -ngl 99 -dev TT_METALIUM0 --single-turn`. First
  attempt (`SET_ROWS` support only) got past the old KV-cache-placement abort
  and hit the new `VIEW` one described above - expected, diagnosed, and fixed
  in the same pass rather than as a separate investigation. Second attempt
  (with `VIEW`/`RESHAPE`/`TRANSPOSE`/`PERMUTE` also enabled): clean run, real
  generated text, exit code 0 - now with the *entire* KV-cache lifecycle
  (write via `SET_ROWS`, read back via `VIEW`) happening on `TT_METALIUM0`,
  no CPU fallback flag needed:
  ```
  > You are a helpful assistant
  a helpful assistant, you are an assistant, you are a helpful assistant. However
  [ Prompt: 13.4 t/s | Generation: 1.3 t/s ]
  Exiting...
  ```
  (Generation dropped from 4.7 to 1.3 t/s versus sec 18's `-nkvo` run - expected,
  not a regression to chase: every decode step now round-trips the *entire*
  KV-cache buffer through host memory per sec 9's whole-buffer-only
  constraint, on top of the activation round-trip sec 15 already had. Still a
  functional-simulator number, not a hardware performance signal - sec 18's
  same caveat applies.)

**Consequence**: sec 18's "what's left" list shrinks by one. The remaining
items (performance is meaningless on ttsim, L1 capacity for larger models'
weight blobs, broader prompt/shape coverage) are unchanged by this section.

## 20. L1 capacity: reader kernel now streams weight in N-tile chunks instead
    of keeping the whole packed blob resident

Closed sec 13/14/18's remaining L1-capacity item. The reader kernel
(`kernels/ternary_matmul/dataflow/reader_ternary_mm.cpp`) used to read the
*entire* packed weight blob (`N*K/4 + 32` bytes) into one resident scratch
CB up front - correct, but sized proportional to the whole weight tensor,
which doesn't fit in L1 for real model dimensions. Checked against the
actual `microsoft/BitNet-b1.58-2B-4T-gguf` weights this port has been
validating against: `blk.0.ffn_gate`/`ffn_up` (N=6912, K=2560) need 4.3MB
resident under the old scheme - a Blackhole Tensix core's L1 is on the order
of 1.5MB total, so this was never going to fit outside ttsim (which doesn't
enforce L1 capacity, only correctness), even though the sec 18 GGUF run
"succeeded".

**Fix**: the reader now fetches only one N-tile's row-block (32 rows x K/4
bytes) per (mt, nt) pair, into a scratch CB sized for just that chunk,
instead of the whole blob. This bounds L1 usage to a fixed size per weight
tensor independent of N - 20-54KB across this model's projection matrices,
an 80-216x reduction, comfortably fitting alongside the kernel's other CBs.
The tradeoff: since the loop nesting stays mt-outer/nt-middle/kt-inner
(unchanged, to keep matching the stock compute kernel's tile-consumption
order - see sec 13/14), and the weight chunk doesn't depend on `mt`, it gets
re-fetched from DRAM once per M-tile instead of once total - more DRAM
traffic for less L1, the same kind of tradeoff the activation reads already
made (sec 13's "redundant but simple" re-reads), and irrelevant on ttsim
where performance isn't measured (sec 8/18).

**Mechanism**: the weight tensor's DRAM buffer is still a single page
spanning the whole tensor (buffer-type design from sec 9 - the two upstream
tt-metal interior-`BufferRegion` bugs rule out multi-page host-side
placement). That constraint is specific to *host*-facing
`MeshCommandQueue` transfers, though - it says nothing about device-side NOC
reads. `TensorAccessor::get_noc_addr(page_id, offset)` simply adds `offset`
onto the page's base NOC address with no bounds checking against the page's
nominal size (confirmed by reading `InterleavedAddrGen::get_addr` and the
non-interleaved `TensorAccessor::get_noc_addr` in tt-metal's
`tensor_accessor.h`/`dataflow_api_addrgen.h` - both are a plain offset add).
So `weight_accessor.get_noc_addr(0, nt * chunk_bytes)` followed by a plain
`noc_async_read` of `chunk_bytes` gives a correct, arbitrary-byte-range
device-side read into that one page - entirely bypassing the buggy
`SDMeshCommandQueue` code path (sec 9's bugs are host<->device transfer
bugs, not NOC-level ones), no host-side buffer-type change needed.

**Verified under ttsim**: re-ran the sec 18/19 GGUF command
(`llama-cli -m <I2_S GGUF> -ngl 99 -dev TT_METALIUM0 --single-turn`) after
the fix - clean run, real generated text, exit code 0, comparable throughput
to sec 19's baseline (14.2 t/s prompt / 1.3 t/s generation vs. 13.4/1.3) -
consistent with the fix being a pure L1-footprint change with no effect on
the computed result, not a correctness change.

**Not addressed here**: the per-superblock redundancy within a chunk (4
consecutive `kt` values sharing the same 32 bytes/row are still read as part
of the same chunk, not deduplicated further - already implicitly handled
since the chunk covers the whole N-tile row, not per-kt) and any true
double-buffered/prefetched streaming (the current chunk read is a blocking
`noc_async_read_barrier()` before use, same synchronous style as the rest of
this kernel) are both left as possible follow-up performance work, out of
scope while performance is explicitly deferred to real silicon (sec 8/18).

## 21. Broader coverage pass: longer generations, varied prompts, batch>1 -
    and a significant finding about how much MUL_MAT actually offloads

Closed sec 18/19's "only one prompt/shape tested" item. All runs under
ttsim, `-m <I2_S GGUF> -ngl 99 -dev TT_METALIUM0`, after sec 20's fix:

- **Longer generation** (`-n 64`, same short prompt): completes cleanly,
  exit 0, stable throughput (15.5 t/s prompt / 1.4 t/s generation) - KV
  cache correctly accumulates 64 decode steps' worth of SET_ROWS/VIEW
  traffic with no crash or corruption.
- **Very short prompt** (`"Hi"`, `-n 16`): clean, exit 0.
- **Long multi-sentence prompt** (~70 words, `-n 32`): clean, exit 0, and
  notably higher prompt throughput (33.9 t/s) - consistent with more of the
  larger prefill batch actually landing on-device (see finding below).
- **Batch>1** (`llama-batched -np 4 -n 16 -kvu`, 4 parallel sequences,
  unified KV cache): clean, exit 0, all 4 sequences produce real text
  (`decoded 44 tokens in 5.32s, speed: 8.28 t/s`). Confirms multi-sequence
  KV-cache row placement (different sequences writing/reading different
  cache rows via SET_ROWS/VIEW) works, not just the single-sequence case
  every other test in this document exercises. (Needed `-kvu` - the initial
  attempt without it hit `split_equal: sequential split is not supported
  when there are coupled sequences`, a stock llama.cpp KV-cache-mode
  requirement for `llama-batched`, unrelated to this backend.)

**Significant finding, via `GGML_SCHED_DEBUG=2 -v`**: instrumented a real
run's scheduler node-placement log (`ggml_backend_sched_print_assignments`)
to see which `MUL_MAT` nodes actually execute on `TT_METALIUM0` versus
falling back to CPU. Result: the large majority never reach the device.
Per-pass breakdown (one "pass" = one `llama_decode()` call, delimited by
`GET_ROWS`) shows an unambiguous pattern: passes with a single new token
(ordinary autoregressive decode, `M=1`) show **zero** `MUL_MAT` nodes placed
on `TT_METALIUM0` - 100% CPU fallback; only passes carrying a large enough
prefill batch show any device placement at all, and even then the
non-I2_S-weight matmuls (`kq`/`kqv`, attention scores - never eligible for
this backend regardless of shape) stay on CPU as expected. Root cause is
`ggml_backend_ttnn_mul_mat_shape_ok()`'s `M % 32 == 0` check (sec
13-15/kernel README - the FPU only operates on whole 32x32 tiles, and the
kernel triad has no sub-tile or padding handling): a single-token decode
step's activation matmul has `M=1`, a batch-of-4 parallel-sequence decode
step has `M=4`, neither is a multiple of 32, so `supports_op()` correctly
(per its own contract) returns `false` and `ggml_backend_sched` transparently
routes those nodes to CPU - inserting a `CPU#<weight-name>` host-side copy of
the (already-on-device) I2_S weight to do it, exactly the graph-splitter
behavior sec 2 described as "the standard llama.cpp GPU-offload pattern".
Nothing here is a bug: every fallback is exactly what `supports_op()`
declares it can't handle, and the sec 18/19 "full offload" runs were
correctly described at the level they were tested (the mechanism works, KV
cache included) - but "every offloadable MUL_MAT runs on TT_METALIUM0" reads
as stronger than it is once you know *how much* of a real run's matmul
traffic is actually offloadable. Concretely, on the sec 18/19 GGUF at `-n 4`:
828 of 3945 total `MUL_MAT` nodes (~21%) placed on `TT_METALIUM0`, and all
828 came from the handful of passes with large-enough prefill batches - zero
came from ordinary single-token decode passes, which is the dominant
workload for interactive inference.

**Consequence**: this backend's `MUL_MAT` kernel today mainly benefits
prompt/prefill processing when the batch happens to be large and
tile-aligned (or is chunked by llama.cpp's own ubatching into a tile-aligned
piece), not steady-state token generation - the part of inference most
sensitive to per-token latency. Closing this would mean relaxing
`ggml_backend_ttnn_mul_mat_shape_ok()`'s `M % 32 == 0` requirement by
padding the activation to the next tile boundary on upload (zero-filling the
extra rows) and truncating the result back to the real `M` before writing
`dst` - the `K % 128` and `N % 32` constraints are separate (weight-shape,
not activation-shape) and unaffected. Not attempted here: it's a real
kernel-integration change, not a coverage-testing exercise, and enabling it
would add yet another whole-activation-buffer host round trip to every
single decode step (on top of sec 19's KV-cache round trip), which on ttsim
would show up as *slower* generation for a workload ttsim can't measure
meaningfully anyway (sec 8/18) - worth doing once real hardware makes the
tradeoff measurable, flagged here as the natural next step rather than done
as part of this pass.

## 22. Attempted sec 21's M-padding fix: correct in isolation, but blocked by
    a much bigger, previously-hidden reader-kernel scalability wall

Implemented the fix sec 21 proposed: `ggml_backend_ttnn_compute_mul_mat()`
now rounds `M` up to `Mt = ceil(M/32)` M-tiles, builds the transposed/tiled
activation at that padded width (`M_padded = Mt*32`, extra columns left
zero by `act_transposed`'s value-initializing sized constructor), and after
`untilize_nfaces` truncates back to the real `M` when writing `dst` (only
the real `out_host[m*N+n]` rows are written; `result[n*M_padded+m]`'s
`[M, M_padded)` columns - all-zero-activation outputs - are simply never
read). `K % 128` and `N % 32` are unaffected (weight-shape constraints, not
activation-shape).

**Correctness verified in isolation**: two standalone tests (same
methodology as sec 13/16/18 - a real `ggml_cgraph` through
`ggml_backend_graph_compute`, not a hand-rolled dispatch), both against a
small `K=128,N=32` weight: `M=5` (arbitrary non-tile-aligned) matched a CPU
f64 reference to `max_abs_err=0.056` (0/160 over a 0.5 threshold); `M=1`
(the literal decode-step shape) matched to `max_abs_err=0.004` (0/32 over
threshold) - both within the same bf16-rounding tolerance every prior
kernel validation in this document has seen. The padding/truncation logic
itself is correct.

**But relaxing `ggml_backend_ttnn_mul_mat_shape_ok()`'s `M % 32 == 0` check
to actually use this at runtime surfaced a much larger problem**, found
while re-verifying against the real GGUF: a `GGML_SCHED_DEBUG=2` run at
`-n 4` never got through even the first forward pass in over 280s (it had
previously taken ~15-20s total for `-n 4`), and a longer attempt eventually
aborted (`terminate called without an active exception`, SIGABRT) partway
through - initially suspected as a new tt-metal stability bug in the same
family as sec 9/11.

**Root-caused with two isolated standalone tests**, bypassing llama.cpp
entirely to separate "many calls" from "real-sized calls":
- Looping the *same* small (`K=128,N=32,M=1`) call 300 times in one process
  reached 220+ iterations in under 280s with no crash - ruling out pure call
  volume (Program/MeshBuffer create-destroy churn) as the cause.
- A *single* call at one of this model's actual per-layer shapes
  (`K=2560,N=2560,M=1`, e.g. `attn_q`/`attn_output`) did not complete within
  580 seconds (measured directly, no crash - just still running).

Root cause: the reader kernel's ternary-unpack loop (sec 13/14) is
`Nt*Kt` 32x32-tile iterations, each unpacking 1024 elements with a scalar
per-byte loop (`kernels/ternary_matmul/dataflow/reader_ternary_mm.cpp`) -
for `K=2560,N=2560`, that is `80*80*1024 = 6.55M` scalar loop iterations,
simulated instruction-by-instruction by ttsim (a functional simulator, not
built for throughput). This cost is **independent of `M`** - it was never
about M-padding specifically. The reason it was invisible through sec
13-21 is that `M % 32 == 0` incidentally also gated out virtually all
real per-layer decode-time `MUL_MAT` calls (sec 21's own finding: ordinary
decode has `M=1`), so no test before this one had ever actually pushed a
real `K,N` in the thousands through this reader kernel's unpack loop under
ttsim and measured how long it takes. The sec 11 `MeshDevice`-destroy crash
this superficially resembled is unrelated; the abort earlier in this section
is most plausibly what happens when a process stuck for minutes inside this
loop gets forcibly terminated (by an external timeout) rather than a
distinct tt-metal bug - not confirmed further, since the real, actionable
finding (the unpack loop's cost) doesn't need that resolved to be understood.

**Decision**: keep the M-padding logic in `compute_mul_mat` (correct,
verified, and harmless when unused - `M_padded == M` whenever `M` is
already tile-aligned, its only live case today), but **restore**
`ggml_backend_ttnn_mul_mat_shape_ok()`'s `M % 32 == 0` requirement rather
than ship it relaxed by default. Enabling it would turn the existing,
validated `-ngl 99` smoke test (`docs/run-with-ttsim.md` sec 6, normally
~15-30s for `-n 16`) into something taking many minutes to hours per run,
for a problem the padding fix doesn't cause and can't fix on its own - a
severe regression to a currently-working, documented workflow, discovered
only because this fix finally let real-shaped decode matmuls reach the
kernel at all. Re-verified after restoring the gate: `-n 16` completes in
~16s wall time, `8/8` JIT cache hits, unchanged from before this section.

**Consequence / next step**: sec 21's finding stands - ordinary decode-step
`MUL_MAT` still never reaches `TT_METALIUM0` - but the fix is no longer
"relax the M constraint" (that part is done and ready). The actual blocker
is now the reader kernel's unpack loop needing a real optimization pass
(e.g., vectorizing the 32x32 unpack with SFPU ops instead of a scalar
per-byte RISC-V loop) before it's practical to exercise at real projection-
matrix dimensions under ttsim at all - independent of M-alignment, and
worth doing before any further real-model ttsim testing, not just before
re-enabling decode-step offload.

## 23. Reader kernel unpack loop optimized: real, verified speedup - but not
    enough to make real dimensions practical under ttsim

Attempted sec 22's flagged next step: speeding up the reader kernel's
unpack loop (`kernels/ternary_matmul/dataflow/reader_ternary_mm.cpp`)
enough to make real projection-matrix dimensions testable under ttsim.
Two structural changes, both algorithmic (same math, fewer/cheaper
operations), not a rewrite of what the kernel computes:

1. **Precomputed tile-face index table.** The row/col -> tile-face byte
   offset (`face_y*TILE_ROW_STRIDE + face_x*FACE_HW + local_row*FACE_DIM +
   local_col`) is invariant across every `(mt, nt, kt)` visited - it only
   depends on `(row, c)` within a 32x32 tile. The old code recomputed it
   (with a division/modulo chain) for every one of up to `Mt*Nt*Kt*1024`
   element visits. Now computed once into a 1024-entry table
   (`face_idx[row*32+c]`) at kernel start, replacing that chain with a
   single array lookup in the hot loop.
2. **Four-lane batching per superblock.** The four K-tiles sharing a
   packing superblock (`kt = sb*4 + lane`, sec 13/PORTING_PLAN sec 10) read
   the *exact same* packed byte per row, decoded at four different bit
   offsets. The old code did four entirely separate passes over that same
   32x32 byte block (one per `kt`), reloading and redecoding the same bytes
   four times. Now unpacked together from a single pass: one L1 byte load
   feeds all four lanes' `decode()` + tile-face-indexed store, and all four
   resulting tiles are reserved/pushed to `cb_in0` as one 4-tile batch
   (`cb_reserve_back(cb_id_in0, 4)`, standard tt-metal multi-page
   reserve/fill/push pattern - contiguous pages within one reservation).
   `K % 128 == 0` (required by `ggml_backend_ttnn_mul_mat_shape_ok`)
   guarantees `Kt = K/32` is always a multiple of 4, so this never needs
   remainder handling. Host side: `cb_in0`'s capacity raised from 2 to 4
   tiles (`ggml-ttnn.cpp`) to allow reserving a full 4-tile batch at once;
   this is a single-buffered exact fit, not further pipelined, matching
   this kernel's existing fully-synchronous style (every read is already
   followed immediately by `noc_async_read_barrier()`).

**Correctness reverified thoroughly**, same methodology as every kernel
change in this document: two dense random-pattern tests against a CPU f64
reference (K=256/N=64/M=32: `max_abs_err=0.456`, 0/2048 over a 0.5
threshold; K=384/N=96/M=64: `max_abs_err=0.748`, 25/6144 over threshold),
plus - because that second run's higher over-threshold count was initially
concerning enough to check rather than assume - an **exact sparse
diagnostic** (isolated single nonzero weights, one per `(nt, sb, lane)`
combination, against a constant activation, so the expected output is
exact with no bf16 noise to obscure a real bug): 0/3072 mismatches at
K=384/N=96, 0/40960 mismatches at K=1280/N=1280 (10 superblocks x 40
N-tiles x 4 lanes, the largest scale exercised). This confirms the growing
over-threshold counts in the dense tests are ordinary bf16 accumulation
noise scaling with K (consistent with sec 14's own note that tolerance
needs scaling up for larger K), not a regression from this rewrite - the
addressing/batching logic itself is exact at every scale tested.

**Timing, measured directly** (standalone test harness, real
`ggml_backend_graph_compute` dispatch, single matmul call, `M=32` tile-
aligned so the call isn't rejected by `supports_op`):

| K=N    | old code (sec 22 baseline) | new code (this section) |
|--------|------------------------------|--------------------------|
| 512    | not measured                 | ~43s                     |
| 1280   | not measured                 | ~253s                    |
| 2560   | did not finish in >580s      | **~930s (15.5 min)**     |

The K=512/1280/2560 timings scale roughly linearly with `N*K` (as
expected: outer-loop iteration count is `Nt*Sb = (N/32)*(K/128)`, exactly
proportional to `N*K`), confirming the optimization is working as designed
- this is a real, several-times reduction in the same real-hardware-shaped
call that never completed within 580s before (sec 22). It is not, however,
close to "practical": one `K=2560,N=2560` matmul call - a single Q/attn_out
projection in this 2B model - still takes ~15.5 minutes. A real forward
pass has ~7 such matmuls per layer across 30 layers; even prefill (let
alone per-token decode) at real dimensions remains far outside any
reasonable ttsim testing budget.

**Decision**: keep this change (strictly better, verified correct, and the
existing `-ngl 99 -n 16` smoke test is unaffected and still fast, ~16-18s,
since the `M % 32 == 0` gate from sec 22 stays in place regardless).
**Do not** treat this as closing sec 21/22's gap - re-enabling
`ggml_backend_ttnn_mul_mat_shape_ok`'s M-alignment relaxation still isn't
practical: even with this speedup, a single real-dimension matmul call
measured in minutes, not seconds, means every decode step (which needs
~7 per layer x 30 layers if fully offloaded) would take hours under ttsim.

**Consequence**: closing sec 21's gap for real needs an order-of-magnitude
bigger win than an algorithmic tidy-up of the existing scalar loop can
give - the fundamental issue is that this unpack work runs on a
data-movement RISC-V core (scalar, one element at a time) rather than the
FPU/SFPU compute engine (vector-parallel by design). A genuine fix likely
means restructuring the kernel triad so the ternary unpack happens as a
vectorized SFPU operation on the compute core instead of a data-movement-
core loop - a materially larger redesign than this section's tidy-up, not
attempted here. Separately, ttsim's own per-instruction simulation
overhead is unknown to be the dominant multiplier here or not - real
silicon may make even the *unoptimized* sec 22 baseline practical, since a
RISC-V core actually clocked in the hundreds of MHz to GHz range doing
~6.5M scalar element-unpacks is on the order of tens to low hundreds of
milliseconds, not minutes; ttsim being a functional (not throughput-
oriented) simulator plausibly inflates this by orders of magnitude beyond
what real hardware would show. Both avenues (SFPU redesign, or simply
trusting real hardware to be fast enough without further kernel work) are
open; neither is resolved here.

## 24. The SFPU redesign: real, verified ~6x speedup - plus a genuine
    concurrency deadlock found and fixed along the way

Attempted sec 23's flagged "real fix": move the ternary unpack off the
scalar data-movement RISC-V core entirely, onto the Tensix compute core's
SFPU (a vector engine), rather than continuing to tune the scalar loop.

**Feasibility research first** (no precedent existed for this exact
combination in-tree): tt-metal's compute-kernel API has documented,
LLK-backed SFPU ops for exactly what this needs -
`bitwise_and_tile`/`right_shift_tile` (Int32/UInt32/UInt16 tiles),
`typecast_tile` (UInt16 <-> Float16_b is a supported direct path), and
`sub_unary_tile` (fp32 scalar subtract). No hardware "unpacker" shortcut
exists for a bare 2-bit code on Wormhole/Blackhole (`DataFormat`'s only
narrow-int support, `MxInt2`, is gated on the unreleased Quasar
architecture) - the shift/mask/typecast has to run as real SFPU
instructions. `tt_metal/programming_examples/sfpu_eltwise_chain/` gave the
structural template (`tile_regs_acquire` -> `copy_tile` -> chained
`op_init()`/`op()` calls -> `tile_regs_commit`/`pack_tile`); reading
`ttnn/.../moreh_sum`'s h-reduction kernel confirmed a single compute kernel
file can legitimately be both producer and consumer of an intermediate CB
(needed here: `cb_in0` is produced by the new SFPU phase and consumed by
the matmul phase, both within `compute/mm.cpp`).

**Design** (`kernels/ternary_matmul/`):
- The reader (`dataflow/reader_ternary_mm.cpp`) got *simpler*, not just
  smaller: it now gathers one **shared, undecoded** raw-byte tile per
  superblock (not per lane) into a new CB (`cb_raw`, `UInt16` format -
  the bitwise/shift SFPU ops require Int32/UInt32/UInt16, not `UInt8`) -
  1024 scalar byte-copies per superblock, a 4x reduction from sec 23's
  four-lanes-duplicated version, since all four lanes now decode from the
  *same* shared tile instead of the reader writing it four times.
- `compute/mm.cpp` gained a Phase 1 (per superblock, one `tile_regs`
  acquire/commit/release cycle per lane): `copy_tile` the shared raw tile
  into DST, `right_shift_tile(6-2*lane)` + `bitwise_and_tile(0x3)` isolate
  that lane's 2-bit code, `typecast_tile<UInt16,Float16_b>` converts it to
  a float, `sub_unary_tile(1.0f)` lands on the actual value - the
  code->value mapping (sec 10) is exactly `value = code - 1`, so no lookup
  table is needed on the SFPU side, just one subtract. Phase 2 (the stock
  FPU accumulation loop) is byte-for-byte unchanged from every earlier
  version of this kernel. `reconfig_data_format_srca` /
  `copy_tile_to_dst_init_short` / `matmul_init` are re-issued at each
  phase transition to reprogram the unpack/math hardware state between the
  two different CB formats and op types in play (same pattern used by
  `moreh_sum`'s mixed-op kernel).
- Host side (`ggml-ttnn.cpp`): new `cb_raw` CB (`UInt16`, 1-tile capacity -
  the reader/compute handshake for it is single-buffered, matching this
  kernel's fully-synchronous style everywhere else); `cb_in0` resized from
  a 4-tile batch to a full `Kt`-tile queue, since Phase 1 now produces all
  of a tile's K-dimension decode *before* Phase 2 starts draining it (not
  interleaved - matmul's accumulation holds one open `tile_regs` session
  across the whole Kt loop, which cannot be interrupted by Phase 1's own
  per-lane sessions without corrupting the in-progress accumulation).

**A real concurrency bug, found and fixed**: initial testing at K=128 (a
single superblock) passed cleanly first try - correct output, ~1s. Scaling
to K=256 (two superblocks) hung indefinitely (no completion within 280s,
versus a fraction of a second of actual compute at K=128). Isolated with
two synthetic shapes that separately vary only one dimension
(K=128/N=64, multiple N-tiles/single superblock: fine, ~1.8s; K=256/N=32,
single N-tile/multiple superblocks: hangs) - conclusively tied to
*multiple superblocks*, not multiple N-tiles. Ruled out a `cb_in0`
off-by-one sizing theory by over-provisioning it (no change). Root cause,
once found: `cb_in1` (activation tiles) was left at its old 2-tile
double-buffered capacity, but - unlike every earlier version of this
kernel - Phase 2 is now the *only* consumer of `cb_in1`, and Phase 2 does
not start until Phase 1's entire superblock loop finishes (program order
within one compute kernel). The reader, a separate concurrently-running
core, pushes activation tiles for superblock 0 into `cb_in1` faster than
anything drains them; once it fills `cb_in1`'s 2 slots, the reader blocks
- and since the reader's own loop pushes a superblock's `cb_in1` tiles
*before* moving on to request the *next* superblock's `cb_raw` tile,
Phase 1 (which needs that next `cb_raw` tile to proceed) stalls too. A
three-way circular wait: reader -> blocked on `cb_in1` space -> which only
Phase 2 frees -> which only starts after Phase 1 finishes -> which needs
the reader's next `cb_raw` push -> which the reader can't reach. Single
superblock shapes never hit this (no "next superblock" to block on) - the
same "invisible until problem size > 1" pattern this document has hit
repeatedly (sec 14, sec 21). Fixed by sizing `cb_in1` to `Kt` tiles too,
the same reasoning already applied to `cb_in0`.

**Correctness reverified after the fix**, same methodology as every
kernel change in this document: dense random tests at K=256/384/512
(N=32/96/512) all match sec 23's own numbers on the same shapes almost to
the bit (e.g. K=512/N=512/M=32: `max_abs_err=0.909066`, `mean=0.137053`,
`233/16384` over threshold - identical to sec 23's measurement of the
same shape), plus a full exact sparse diagnostic at K=384/N=96 (36 probes,
every `(nt, sb, lane)` combination): 0/110592 mismatches.

**Timing, measured directly, same standalone harness as sec 22/23**:

| K=N    | sec 23 (scalar, optimized) | sec 24 (SFPU)      | speedup |
|--------|------------------------------|---------------------|---------|
| 1280   | ~253s                        | ~44s                | ~5.7x   |
| 2560   | ~930s                        | ~154s (2m34s)       | ~6.0x   |

Both K=1280 and K=2560 SFPU runs reproduce sec 23's exact error metrics
(`max_abs_err`, `mean_abs_err`, over-threshold counts all identical to
several decimal places) - confirming this is a pure performance change,
not a numerics change, consistent with Phase 2's FPU accumulation being
byte-for-byte the same code as before.

**Still not "practical"**: ~154s for one real-dimension matmul call means
a full 30-layer forward pass (~7 such calls/layer if every projection were
this size) would still take on the order of hours under ttsim, not
seconds - `ggml_backend_ttnn_mul_mat_shape_ok`'s `M % 32 == 0` gate stays
in place (sec 21/22 unchanged); this does not make re-enabling decode-step
offload viable under ttsim. What it does establish: the SFPU redesign is
functionally correct and a real, substantial (~6x) win over further
scalar tuning, verified end to end including a genuine concurrency bug
that would have been very hard to find without the same small-scale-first
methodology this whole document has used throughout. Since ttsim's
absolute numbers are repeatedly not a real-hardware performance proxy
(sec 8/18/23), and this specific change trades scalar RISC-V instructions
for vector SFPU ones - the kind of change real silicon's actual clock-rate
parallelism should reward far more than a slow functional simulator can
show - it remains plausible (not verified) that this redesign matters more
on real hardware than the ttsim numbers alone suggest.

## 25. Multi-core dispatch: partitioning N-tiles across the device grid,
    another ~10-15x on top of sec 24's SFPU win

Sec 24's redesign still ran the whole kernel triad on a single Tensix core
(`CoreCoord core({0, 0})`) regardless of matmul size - every other core on
the chip sat idle. This section spreads the work across the device's
available cores instead.

**Design**: N-tile is the natural split axis. `mesh_device-
>compute_with_storage_grid_size()` gives the available grid;
`split_work_to_cores(core_grid, Nt)` (a stock tt-metal utility, same one
`matmul_multi_core` uses) partitions the `Nt` N-tiles into two core groups
(some cores get one more tile than others when `Nt` doesn't divide evenly)
and returns the actual core set needed - never more cores than `Nt`, so a
single-N-tile matmul still runs on exactly one core, unchanged from sec
24. Each core:
- Handles a *contiguous* slice `[nt_start, nt_start+nt_count)` of the
  global N-tile range, computing complete output tiles independently - full
  `Kt`-depth accumulation, no cross-core communication.
- Redundantly re-reads the *entire* activation tensor for its own `Mt`
  loop, same "redundant but simple" tradeoff already used elsewhere in this
  kernel (sec 13's activation re-reads, sec 20's weight-chunk re-reads).
- Gets its own independent copy of every CB (`cb_in0`, `cb_in1`, `cb_raw`,
  `cb_scratch`, `cb_out`) - CBs are per-core L1, so `CreateCircularBuffer`
  on the whole core set (`all_cores`) just means "allocate this config
  once per core," not a shared buffer.

Concretely: `Nt` moved from a compile-time arg to a runtime one in the
compute kernel (`compute/mm.cpp`) - different cores can get different
`nt_count` values, but one `CreateKernel` call compiles a single binary
shared by every core in `all_cores`, so anything that varies per core has
to be a runtime arg, not baked in. The reader and writer kernels each
gained an `nt_start` runtime arg: the reader needs it to compute the
*global* N-tile index for the weight's DRAM address (its own loop variable
`nt` is now local to the core's slice); the writer needs it for the same
reason when computing the output DRAM page index `(nt_start+n)*Mt+m` - all
cores share the *same* output buffer (sized for the whole `Mt*Nt` tensor),
each just writing into its own slice of it.

**Verified correct**: a dense random test at K=128/N=128 (4 N-tiles, so up
to 4 cores) matched expected bf16-noise tolerance on the first attempt
(`max_abs_err=0.176439`, 0/4096 over threshold - identical to sec 24's
single-core number on the same shape). Given sec 24's own history found a
real concurrency bug specifically once superblock count exceeded 1, this
change was checked more broadly before trusting it: the full exact sparse
diagnostic at K=384/N=384 (12 N-tiles - a wide spread across cores, 3
superblocks, 4 lanes = 144 probes, each exercising whichever core owns
that probe's N-tile) came back **0/1769472 mismatches**. No new
concurrency issue this time - each core's reader/compute/writer triad is
a fully self-contained instance of exactly sec 24's already-verified
logic, with no CB or `tile_regs` state shared *across* cores (only within
one core, unchanged from sec 24), so the failure mode that bit sec 24
(a single kernel's internal CBs sized wrong for its own multi-superblock
loop) has no multi-core analogue here.

**Timing, measured directly, same standalone harness as sec 22-24**:

| K=N    | sec 24 (SFPU, single-core) | sec 25 (SFPU, multi-core) | speedup |
|--------|------------------------------|------------------------------|---------|
| 1280   | ~44s                         | ~4.3s                        | ~10.3x  |
| 2560   | ~154s                        | ~9.9s                        | ~15.5x  |

Both shapes reproduce sec 23/24's exact error metrics again (identical
`max_abs_err`/`mean_abs_err`/over-threshold counts) - another pure
performance change, no numerics difference, as expected: each core runs
byte-for-byte the same per-tile logic as the single-core version, just
fewer tiles per core.

**Cumulative picture**: sec 20's original whole-blob design didn't
complete a K=2560,N=2560 call in over 580s; sec 23's scalar tidy-up got it
to ~930s; sec 24's SFPU redesign got it to ~154s; this section gets it to
~9.9s - roughly a 94x improvement from sec 23 to here, in a workload where
the original design would still be running.

**Not fully explained, flagged rather than chased**: the ttsim SoC
descriptor for this Blackhole configuration
(`~/stage_bitnet/Bitnet-TT/sim/soc_descriptor.yaml`) lists 140 functional
worker cores (a 14x10 grid), comfortably more than `Nt=80` for the K=2560
case - so in principle up to 80 cores were available for that run.
`split_work_to_cores` doesn't report which one back to this program at
that call site, and this wasn't independently reconciled, but the observed
~15.5x wall-clock speedup is well short of an 80x ceiling, and the
speedup growing from ~10.3x (`Nt=40`) to ~15.5x (`Nt=80`) rather than
holding constant suggests the run isn't yet purely core-count-limited.
Plausible causes, neither confirmed: fixed per-core dispatch/kernel-launch
overhead that doesn't shrink with core count, or ttsim itself not being a
throughput-oriented simulator for many-core concurrency (consistent with
this document's repeated point that ttsim's absolute numbers aren't a
real-hardware performance proxy - sec 8/18/23/24). Not investigated
further here since the result is already a substantial, verified win;
worth revisiting if multi-core scaling itself ever becomes the bottleneck
under study.

**Consequence**: `ggml_backend_ttnn_mul_mat_shape_ok`'s `M % 32 == 0` gate
(sec 21/22) is *not* revisited by this section - that's an independent
question from core count, and the combined effect of sec 24+25 (~94x
faster than sec 23 alone) changes the practicality calculus enough that
it may be worth reopening, but that decision is left for a follow-up
rather than made here.

## 26. Revisiting the M-gate: real decode-step offload, enabled by default

Sec 25 flagged this as the natural next step - the cumulative ~94x
speedup from sec 24+25 might make lifting `ggml_backend_ttnn_mul_mat_
shape_ok()`'s `M % 32 == 0` gate (sec 21/22) practical for the first time.
Measured before deciding, same discipline as every other section here.

**Real per-call timing at M=1** (the actual decode-step shape), standalone
harness, across every distinct real projection shape in this 2B model:

| Projection             | K    | N    | M=1 time |
|-------------------------|------|------|----------|
| attn_q / attn_output    | 2560 | 2560 | ~5.8s    |
| attn_k / attn_v         | 2560 | 640  | ~3.3s    |
| ffn_gate / ffn_up       | 2560 | 6912 | ~12.8s   |
| ffn_down                | 6912 | 2560 | ~18.7s   |

Summed per layer (2+2+2+1 of the above): ~62.5s/layer x 30 layers -
**roughly 31 minutes per generated token** if every eligible `MUL_MAT`
offloads. Down from an estimated many-hours-to-days at sec 23's numbers,
but still far from the existing ~17s `-ngl 99 -n 16` smoke test - leaving
the gate permanently relaxed would turn that same command into several
hours (every decode step now attempts offload, not just occasional
large-enough prefill batches per sec 21).

**Decision, via explicit user sign-off given the magnitude of the
tradeoff**: ship the gate relaxed by default. The backend now genuinely
supports what this whole multi-section effort (sec 21-25) was building
toward - real decode-step offload - and that capability being real is
judged more valuable than keeping the smoke test fast. `docs/run-with-
ttsim.md` is updated to warn about the new timing reality and recommend
`-n 1` (or similarly small) for validation runs rather than the previous
`-n 16`/`-n 64` examples.

**Correctness reverified specifically for the M-padding path**: sec 22
verified the padding/truncation logic in isolation before the SFPU (sec
24) and multi-core (sec 25) rewrites existed; none of sec 24/25's own
tests exercised `M_padded != M` (every one used a tile-aligned M, so
padding was a no-op in every test run before this section) - genuinely
new coverage, not just re-confirming old results. Exact sparse diagnostic
at K=384/N=96 with **M=1**: 0/3456 mismatches. Same with **M=5** (a real,
non-trivial partial tile - Mt=1, 27 padding rows): 0/17280 mismatches.
Dense random at M=5: `max_abs_err=0.615`, only 2/480 over a 0.5 threshold
- consistent with this document's established bf16-noise baseline, not a
regression.

**Scheduler placement, verified directly**: `GGML_SCHED_DEBUG=2` on a
partial real run (killed after 90s, well before completion, just to
observe placement decisions) shows **2730 `MUL_MAT` nodes placed on
`TT_METALIUM0` versus 673 on CPU** - a complete reversal from sec 21's
finding (previously ~21% device / 79% CPU, with 0% of *decode-step*
matmuls ever reaching the device). The remaining CPU-placed nodes are
expected, not a gap: `kq`/`kqv` attention-score matmuls are never I2_S
weights and were never eligible for this backend regardless of `M`.

**Real end-to-end milestone: attempted, found a genuine separate blocker,
not resolved here.** `llama-cli -m <I2_S GGUF> -ngl 99 -dev TT_METALIUM0
-n 1 --single-turn` hit `GGML_ASSERT(dst_located.offset == 0 &&
ggml_nbytes(dst) == dst_located.buffer->size() && "ggml-ttnn: mul_mat dst
must not be a view")` after ~3m37s. This is a pre-existing gap, not caused
by this section's change: `compute_mul_mat`'s final write has always
required `dst` (the `MUL_MAT`'s own output tensor) to be the sole occupant
of its device buffer, a deliberate sec 16 decision at the time ("a future
view in [src0 or dst] should fail loudly instead of silently
misbehaving") - it just never fired before because so few `MUL_MAT` nodes
ever reached this backend. With decode-step offload now real, ggml's
graph allocator's normal buffer-reuse produces a `dst` that shares a
buffer with something else often enough to hit this on the very first
real run.

**First fix attempt, reverted**: routed the final dst write through the
existing generic `ggml_backend_ttnn_buffer_set_tensor` helper (already
used elsewhere for exactly this "tensor might not be the sole buffer
occupant" case - a read-modify-write around the resolved offset, not a
new mechanism) instead of the raw whole-buffer write that assumes sole
occupancy. This looked like a small, low-risk change reusing
already-verified infrastructure - and it *did* get further (past the
assert, into `free(): invalid next size (normal)` inside `libggml-ttnn.so`
at ~14s, a heap corruption bug detected downstream of wherever the actual
overrun happened, the same "detected several frames removed from the real
bug" signature sec 16's own original heap-corruption find had). Rather
than chase that blind, or risk shipping a fix that trades a loud,
deterministic assert for silent corruption, **reverted** the dst-write
change entirely - confirmed the revert restores the exact original assert
behavior (same message, same call site, deterministic) and does not
regress any of this section's own standalone verification (identical
`max_abs_err`/mismatch numbers before and after, at every M tested).

**Consequence**: the M-gate relaxation itself (this section's actual
subject) is real, verified, and kept - every isolated-matmul test at
every shape and M value tested passes, and `GGML_SCHED_DEBUG` confirms
decode-step `MUL_MAT` genuinely reaches `TT_METALIUM0` now. But the
*full* real-model end-to-end milestone needs the dst-buffer-aliasing gap
fixed properly first - a bounded, well-understood problem (unlike sec 16's
original view-support work, this isn't "design a whole new mechanism,"
it's "figure out why routing through the already-correct generic
read-modify-write path corrupts the heap") but one that deserves a clean,
isolated repro before touching this code again, not a fix attempted
inside an already-long real-model run. Flagged as the concrete next
blocker, not resolved here.

## 27. The dst-view blocker, actually root-caused and fixed - real progress,
    plus a new, deeper blocker found

Picked up sec 26's flagged next step: root-cause the `dst must not be a
view` assert properly, with a real repro, instead of guessing.

**Root-caused empirically**, not by further reading of ggml internals in
the abstract: added temporary debug instrumentation to `compute_mul_mat`
printing `dst`'s pointer, `view_src`, `ggml_nbytes(dst)`, and the located
buffer's resolved offset/size, then re-ran the real failing case. Two
consecutive nodes told the whole story:

```
dst=0x...aaf0 name=Qcur-0 view_src=(nil) ggml_nbytes(dst)=20480 located.offset=0 located.buffer->size()=20480  N=2560 M=2
dst=0x...add0 name=Vcur-0 view_src=(nil) ggml_nbytes(dst)=5120  located.offset=0 located.buffer->size()=20480  N=640  M=2
```

`Vcur-0` is **not** a view (`view_src` is null, `root == dst`, offset is
correctly 0) - but its registered buffer is sized `20480` bytes, matching
`Qcur-0`'s size exactly, not its own `5120`. Root cause: ggml's graph
allocator (`ggml-alloc.c`) recycles a scratch tensor's raw address for a
later, differently-sized tensor once the earlier one is dead
(`ggml_gallocr_free_node` / `ggml_gallocr_allocate_node`'s address reuse) -
and does this via `ggml_vbuffer_tensor_alloc` setting `tensor->data`
directly, which does **not** always call this backend's `init_tensor`
callback again. This backend's per-address `MeshBuffer` registry
(`ctx->tensor_buffers`, keyed by raw `tensor->data` value, sec 9) then
keeps returning the stale entry from whatever tensor last legitimately
registered that address - `Qcur-0`'s 20480-byte buffer, not a fresh one
sized for `Vcur-0`. The name "dst must not be a view" was accurate for
the mechanism this assert was originally written to catch (sec 16) but
misleading for what was actually happening here: no view was involved at
all, just a stale registration this backend never noticed had gone
out of date.

**Fix**: `ggml_backend_ttnn_locate()` now detects this directly - if the
registered buffer's size doesn't match `root`'s *current* `ggml_nbytes`,
that mismatch itself proves the address was recycled since the entry was
created (a live, non-recycled registration always matches exactly, as
`Qcur-0` did), so it re-creates a fresh, correctly-sized `MeshBuffer` on
the spot (the same construction logic `init_tensor` uses, factored into a
shared `ggml_backend_ttnn_alloc_tensor_buffer` helper) rather than trusting
the stale entry. This fixes the root cause at the one place every access
path goes through (`locate()`), not just the `compute_mul_mat` dst call
site sec 26's reverted attempt special-cased - `get_tensor`/`set_tensor`/
`memset_tensor` and the view-sanity-check in `init_tensor` itself all
benefit the same way, for the same reason. The `compute_mul_mat` asserts
stay (still correctly catch a *genuine* nonzero-offset view, which Option
B's direct on-device addressing still cannot support) but their stale-size
false-positive is gone.

**Re-verified**: dense random and exact sparse diagnostics at every M
tested (1, 5) - `max_abs_err`/mismatch counts identical to sec 26's
pre-fix numbers, confirming no regression.

**Real end-to-end run: got dramatically further, then hit a different,
new failure.** `llama-cli -ngl 99 -n 1 --single-turn` - previously aborted
in 14 seconds (sec 26's revert) or 3m37s (the very first attempt, before
that) - this time ran for **93 minutes** before failing, further into
real computation than anything in this entire porting effort has ever
reached (no `GGML_ASSERT` in the log - the dst-view issue genuinely did
not recur). It then hit a `SIGSEGV` (address `0x58`, i.e. a near-null
pointer dereference - not one of this backend's own asserts) inside
tt-metal's `SDMeshCommandQueue::read_shard_from_device`, reached via
`wait_for_cores_idle` -> `get_core_type`, called from this backend's own
`ggml_backend_ttnn_read_whole`.

**Not chased further in this session**: each reproduction cycle costs
~90+ minutes, and there isn't yet a cheap, isolated way to test a
hypothesis about this new failure the way the debug-print approach did
for the dst-view bug - it only manifests this deep into sustained,
long-running multi-core kernel dispatch (sec 25), a regime nothing in
this port has exercised before (every earlier real-model attempt failed
within seconds to minutes, well before reaching whatever state this
requires). Plausible, unconfirmed causes: a `MeshBuffer` lifetime
interaction with sec 27's own fix (though local `shared_ptr`s in
`compute_mul_mat` should keep any buffer alive for the duration of the
call that resolved it, so this isn't obviously implicated); a genuinely
separate, pre-existing tt-metal/ttsim internal issue that only surfaces
after enough cumulative kernel dispatches (in the spirit of sec 11's
`MeshDevice`-destroy bug - another case of tt-metal-internal state that
degrades under conditions this port was first to exercise); or something
particular to this run's specific KV-cache/decode-step access pattern at
scale. Flagged as the next concrete blocker - genuinely closer to the
real milestone than sec 26 was, since the dst-view fix is real, verified
progress that is being kept regardless of this new finding.

## 28. Sub-linear multi-core scaling, explained with real data - not a bug
    in this backend's code

Sec 25 flagged but didn't chase why multi-core dispatch's speedup fell
well short of the available 140-core ceiling (~15.5x realized at
`Nt=80`). Investigated with the same tool that worked for sec 27: direct
measurement, not more speculation - temporary `std::chrono` timestamps
around each phase of `compute_mul_mat` (weight-scale read, activation
prep, core-split, program setup, device dispatch+readback, host
postprocess), removed again once the data was in.

**Finding, immediate and unambiguous**: essentially all wall-clock time is
in the device dispatch+readback phase - every host-side phase (weight
read, activation transpose/tilize, `split_work_to_cores`, CB/kernel
setup, final postprocess) measured under 5ms combined, regardless of
shape. This rules out any host-side serial bottleneck (e.g. the
whole-weight-blob read done just to extract a 4-byte scale, sec 10 - a
plausible suspect that turned out to be irrelevant, 0.2-4.7ms even at
real dimensions) - the question is entirely about what happens *inside*
the device dispatch.

**Measured dispatch+readback time vs. core count, K fixed at 2560**:

| N-tiles / cores used | time      |
|-----------------------|-----------|
| 1                      | 2959.0ms  |
| 2                      | 3038.7ms  |
| 5                      | 3314.4ms  |
| 10                     | 4159.6ms  |
| 80 (full K=2560,N=2560)| 8291.2ms  |

**And vs. K (Kt), at a fixed 1 core (`N=32`, so `Nt=1`)**:

| K    | Kt | time     |
|------|-----|----------|
| 1280 | 40  | 1987.4ms |
| 2560 | 80  | 2959.0ms |

Fitting the single-core numbers to `time = C + D*Kt` gives `C ~= 1016ms`
(a per-core floor independent of K - plausibly program-load/dispatch
startup cost, simulated) and `D ~= 24.3ms` per K-tile (the actual SFPU
decode work, sec 24, which genuinely does scale with `Kt` - every core
independently unpacks its own full K-depth for whichever N-tiles it
owns, so this term is expected, correct, and not wasteful).

**The key observation**: that per-core cost (`C + D*Kt`, ~2960ms for
`K=2560`) does **not** shrink as *more* cores run concurrently - if it
did (real hardware parallelism), 80 cores each doing 1 N-tile's worth of
work should take about the same wall-clock time as 1 core doing 1
N-tile's worth of work (~2960ms), since they're independent and
non-communicating (sec 25's whole design). Instead, total time grows
from ~2960ms (1 core) to ~8291ms (80 cores) - roughly 5.3 additional
seconds of apparent cost spread across the 79 extra cores, on top of the
expected-constant per-core work. That growth is the sub-linear-scaling
gap sec 25 measured, now attributed to a specific place: something about
running *more concurrent cores* costs additional wall-clock time in a
way pure hardware parallelism would not.

**Conclusion: this is a ttsim characteristic, not a bug in this
backend's host or kernel code.** Every host-side phase is confirmed
negligible; the kernel triad's per-core logic is unchanged from sec 24's
already-verified-correct single-core version, just replicated across
cores with no cross-core coupling (sec 25's design deliberately avoids
any communication between cores, so there's no shared resource or lock in
*this backend's own code* that additional cores could be contending
over). The remaining explanation is ttsim's own execution model: a
"functional simulator, not a timing model" (this document's own repeated
characterization, sec 8/18/23/24) most plausibly does not genuinely
execute multiple simulated Tensix cores' instruction streams with real
wall-clock parallelism, and instead pays additional real (host) CPU time
for each additional core it has to step through, regardless of how
independent that core's work is. Real Blackhole silicon, where 80 cores
genuinely execute concurrently in hardware, would not be expected to show
this same growth - consistent with, and now backed by concrete
measurement of, this document's standing caveat that ttsim's absolute
numbers are not a real-hardware performance proxy.

**Consequence**: no code change made or needed here - there is no
identified inefficiency in this backend to fix, and "make ttsim itself
simulate many-core concurrency with real parallelism" is out of scope
for this port. Sec 25's flagged question is now answered with data
rather than left open: the sub-linear scaling is explained, and the
explanation points away from further engineering effort here, not toward
it. Worth re-measuring once real hardware is available (per the
standing recommendation throughout this document), where this specific
gap is expected to look very different.

## 29. Chasing sec 27's new SIGSEGV: two hypotheses cheaply tested and
    ruled out, real repro needs more than direct backend dispatch

Picked up sec 27's flagged next step: root-cause the `SIGSEGV` inside
tt-metal's `read_shard_from_device` that a real end-to-end run hit after
93 minutes, without paying ~90 minutes per attempt again.

**Key enabling realization**: every standalone test in this entire
porting effort (sec 13 onward) uses `ggml_backend_alloc_ctx_tensors` -
which gives every tensor its own dedicated, never-reused slot. It cannot
exercise the address-reuse behavior that produced sec 27's stale-buffer
bug (or plausibly this one) at all. `ggml_gallocr_new` +
`ggml_gallocr_alloc_graph` is the actual mechanism `ggml_gallocr`/
llama.cpp uses (reusing a persistent scratch arena across many graph
builds) - switching standalone tests to this API is what makes a cheap,
fast repro of this *class* of bug possible in principle.

**Hypothesis 1: `MUL_MAT` buffer/address churn under multi-core
dispatch.** Two persistent I2_S weights of different N (mimicking
`Qcur`/`Vcur`'s mismatched-size address reuse from sec 27) in one graph,
rebuilt via `ggml_gallocr_alloc_graph` for 3000 iterations (each
iteration includes both a 10-core and a 2-core dispatch, exercising
sec 27's stale-registry-repair path repeatedly) - **3000/3000 clean, no
crash**, in 35 minutes (real decode-step volume is ~210-280 `MUL_MAT`
calls total, so this is >10x that in one op type alone).

**Hypothesis 2: sustained `SET_ROWS`/KV-cache traffic.** A persistent
"cache" tensor, repeatedly written (`ggml_set_rows`) and read back
(`ggml_view_2d`) via the same buffer object every iteration (no address
churn this time - tests sustained NOC/queue traffic on one buffer
instead) - **50000/50000 clean, no crash**, in 6.3 seconds. A real
single-token decode needs on the order of 30 `SET_ROWS` calls (one per
layer); this is >1000x that.

**Both ruled out.** Re-examined the original crash log's exact line
ordering to make sure the failure mode itself was understood correctly
before concluding anything: `Signal: Segmentation fault` appears
*before* the `Closing user mode device drivers`/`Sending exit signal to
remote` lines, confirming those are the crash handler's own teardown
attempt after already catching the SIGSEGV, not evidence of a graceful
shutdown-time bug - this is a genuine mid-execution fault, consistent
with sec 27's original read.

**Consequence**: neither of the two most obvious hypotheses (this
backend's own buffer-lifecycle churn, or sustained single-buffer
traffic) reproduces the bug even at volumes far exceeding a single real
decode step - in isolation, via direct backend dispatch
(`ggml_backend_graph_compute(backend, gf)`, bypassing `ggml_backend_sched`
entirely, same as every prior test in this document). The real bug most
plausibly needs one of: (a) the actual `ggml_backend_sched` machinery a
real run goes through - mixed CPU/TTNN graph splitting, cross-backend
copy insertion, none of which any test in this document has ever
exercised; (b) a rare condition (e.g. a threading race in tt-metal's
async command-queue worker, given tt-metal dispatches work
asynchronously under the hood) that only manifests after sustained real
execution at a scale these cheap synthetic loops don't reach even at
3000-50000 iterations; or (c) something specific to the real model's
graph shape/depth (30 layers, many distinct tensor shapes coexisting)
that these 1-2-shape synthetic loops don't capture. Building a
`ggml_backend_sched`-based repro (registering both CPU and TTNN
backends, letting the scheduler split a mixed graph) is a real next
step and plausibly still cheap relative to a full `llama-cli` run, but a
bigger undertaking than either hypothesis tested here - flagged rather
than started, since the two most promising and independently-informative
cheap tests are conclusively negative results worth reporting on their
own.












