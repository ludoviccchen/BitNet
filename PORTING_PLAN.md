# Porting bitnet.cpp to Tenstorrent Blackhole (tt-metal / TT-NN) — Phase 1 findings & plan

Status: DRAFT, for review. No source changes made yet except the submodule
remediation described in §0 (in progress, explicitly requested).

## 0. Blocking issue found and being fixed: submodule pointed at the wrong commit

`.gitmodules` names `isHuangXin/llama.cpp` branch `release-bitnet-embedding-0.6b-270m`
(tip `390c307`) as the BitNet-patched llama.cpp. Two commits ago (`425fbbc`) the
submodule correctly pointed there. The most recent commit, `d63578d` "Replace
submodule by own fork", repointed the submodule to `20f1789` — a **vanilla
upstream llama.cpp commit from August 2024, 6,279 commits before the BitNet
work**. That commit has none of the I2_S/TL1/TL2 types and none of the
`ggml_bitnet_*` dispatch hooks; the `src/ggml-bitnet-{mad,lut}.cpp` files that
already exist at the BitNet.cpp repo root aren't even wired into the CMake
build (`src/CMakeLists.txt` sets `GGML_SOURCES_BITNET` to one file then
immediately overwrites it with the other, and nothing consumes the variable
from outer scope). Additionally, `ludoviccchen/llama.cpp` (the fork now set as
the submodule's `origin`) only ever contained that same stale commit — it was
forked from the wrong point and has no BitNet history at all.

Net effect: as checked out, the submodule builds plain llama.cpp with **no**
ternary-quant code to port.

Fix in progress, per your direction: fetching `release-bitnet-embedding-0.6b-270m`
from `isHuangXin/llama.cpp` and pushing it into `ludoviccchen/llama.cpp`
(~56k objects, slow link — running in background), then repointing the
submodule + updating `.gitmodules`'s `url` to match the fork it actually now
tracks. Everything below is analyzed against that correct branch (tip `390c307`),
fetched directly for inspection; the local checkout will be updated to match
once the push finishes.

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

