# Marigold native validation

## Canonical model boundary

InferBridge selects `prs-eth/marigold-lcm-v1-0` revision
`04a73502f7fd8fc5e59947b9df3b2266d71d6849`.

| Component | Representation | Bytes | SHA-256 |
|---|---|---:|---|
| UNet | canonical FP16 Safetensors | 1,731,927,776 | `953f1ea06169fc6c358b09ecc96a6ee32515e81540442d16239f82348ea62614` |
| VAE | canonical PyTorch archive | 334,715,313 | `a4302e1efa25f3a47ceb7536bc335715ad9d1f203e90c2d25507600d74006e89` |
| CLIP | canonical FP16 Safetensors | 680,820,392 | `bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f` |

The native runtime directly maps the canonical UNet. It never parses pickle.
The development-only, `weights_only=True` converter validates the exact
canonical VAE hash and 248 FP32 tensors, then emits a 334,643,404-byte safe
derived Safetensors representation. The converted representation is 71,909
bytes smaller than the archive container; its additional storage remains
unavoidable unless the catalog starts shipping the VAE in a safe format.

The production pipeline uses only a two-token empty prompt. An 8,704-byte
`MARPRM01` cache binds that constant to the snapshot revision and all three
canonical hashes, avoiding the 680.8 MB CLIP mapping at runtime.
Both derived objects remain hidden, content-addressed cache entries associated
with the shared canonical model rather than separate downloads or model cards.

## Validated native graph

The correctness-first implementation covers deterministic VAE mean encoding,
FP16 conditional UNet execution, one-step `v_prediction` LCM scheduling, VAE
decoding, clipping, channel reduction, and output resizing.

| Gate | Native vs Python CPU |
|---|---:|
| VAE RGB latent relative L1 | `8.46894e-7` |
| VAE decoder relative L1 | `1.43801e-6` |
| UNet relative L1 | `1.03711e-4` (`0.01037%`) |
| Full 64x64 depth relative L1 | `9.61478e-5` (`0.009615%`) |
| Full maximum absolute depth error | `0.000837743` |
| Non-multiple input | `65x73` passed |
| C ABI smoke test | passed |

`marigold_native.dll` exposes ABI 3 lifecycle and stable seeded inference.
The exact validation entry accepts explicit target noise to remove stochastic
ambiguity.

ABI 3 adds the InferBridge byte-image contract without changing the tensor
ABI. `marigold_inferbridge_image_shape` reports the 768-pixel longest-edge
processing dimensions. `marigold_infer_bgra8_f32[_with_noise]` performs the
Python harness's BGR-to-RGB conversion, separable antialiased bilinear resize,
`match_input_res=false` output sizing, and final
`(depth-min)/(1-min)` normalization. This path remains host-backed and does
not advertise an external GPU-resource capability.

The separable filter was checked against PyTorch CPU for representative
upscales and downscales; the largest float64 discrepancy was
`5.55112e-16`. A full 768x64 explicit-noise image-contract canary compared
the BGRA entry with the already validated tensor graph and normalization on
all adapters. Maximum and mean absolute differences were both exactly zero
on the RX 9070, GTX 1080, and RX 6700 XT.

The additive `marigold_create_vulkan` entry converts the canonical FP16 UNet
weights to FP32 during bounded model upload and consumes the same safe derived
VAE sidecar. It fails instead of falling back to CPU. RGB upload and final
depth download remain at the tensor ABI boundary; VAE mean encoding,
conditional UNet, one-step scheduler conversion, decoding, and intermediates
remain on the selected GPU.

| GPU | Full 64x64 relative L1 | Maximum absolute |
|---|---:|---:|
| Radeon RX 9070 | `9.86589e-5` | `0.000829935` |
| GeForce GTX 1080 | `9.80252e-5` | `0.000833750` |
| Radeon RX 6700 XT | `9.89217e-5` | `0.000844181` |

Five consecutive calls on persistent contexts passed. Concurrent canary
medians were 393.47 ms (RX 9070), 717.99 ms (GTX 1080), and 331.08 ms
(RX 6700 XT); these are stability measurements for the untuned graph.

This is the untuned FP32 execution baseline. Mixed precision and external
GPU-resource import/export are not advertised without separate gates.

## First performance pass

VAE and UNet ResNet, attention, and transformer composites now use bounded
Vulkan command batches. Buffer snapshots within a batch use explicit
transfer/compute barriers. Each batch remains block-sized for watchdog safety.

The isolated RX 9070 64x64 median improved from `364.9 ms` to `177.3 ms`
(`51.4%`) in matched seven-iteration runs. Five-call post-change canaries
passed with unchanged errors on every adapter; observed medians were
`226.7 ms` (RX 9070), `625.9 ms` (GTX 1080), and `207.4 ms`
(RX 6700 XT).

## Wave32 packed-weight pass

The Wave32 executor now uses input-major, output-contiguous packed FP16
storage for linear and 3x3 convolution weights while preserving FP32
activations and accumulation. It also uses bank-safe pointwise and linear
tiles, fuses normalization with SiLU, avoids redundant ResNet residual
copies, and computes the shared timestep activation once per denoising step.

For the LCM 64x64 fixture, isolated persistent-context medians were
`102.817 ms` on Radeon RX 9070, `107.026 ms` on GeForce GTX 1080
(101 calls), and `97.069 ms` on Radeon RX 6700 XT. Relative L1 deviation
remained below `0.000813` (`0.0813%`) on all three adapters, and maximum
absolute deviation remained below `0.00301`.

## Full v1 checkpoint

ABI 4 adds `marigold_create_variant` and
`marigold_create_vulkan_variant`, preserving the original create functions
as LCM aliases. `MARIGOLD_MODEL_FULL_V1` selects InferBridge's second
checkpoint, `prs-eth/marigold-v1-0` at revision
`f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c`. The native loader maps its
canonical FP16 UNet and VAE Safetensors directly; no unsafe archive
conversion or duplicate VAE is needed. The full model's prompt cache is a
separate content-bound 8,704-byte sidecar, reusing the identical text-encoder
result but binding the full checkpoint revision and hashes.

The full graph executes the checkpoint's exact ten DDIM timesteps
`[901, 801, 701, 601, 501, 401, 301, 201, 101, 1]`. RGB latent, target
latent, all ten UNet passes, deterministic `v_prediction` scheduler updates,
and VAE decode remain on Vulkan.

| Executor | Full 64x64 relative L1 | Maximum absolute |
|---|---:|---:|
| CPU | `5.79980e-5` | `0.000190556` |
| Radeon RX 9070 | `5.73731e-5` | `0.000192225` |
| GeForce GTX 1080 | `5.73923e-5` | `0.000188887` |
| Radeon RX 6700 XT | `5.75631e-5` | `0.000190258` |

The 768x64 full-v1 BGRA image-contract equivalence canary passed on all
three GPUs with exactly zero maximum and mean absolute difference versus the
validated tensor path plus InferBridge normalization.

## Embedded InferBridge harness

The same DLL exports `ibrh_get_api` for InferBridge harness ABI 2.0. The host
compatibility path and the Core-owned external-resource path both return FP32
depth at `marigold_inferbridge_image_shape` dimensions and preserve
correlation metadata. InferBridge Core, rather than the harness, owns transfer
resources and output leases.

The single Marigold model entry selects its LCM or full-v1 canonical snapshot
through `Checkpoint`. `model_path` names that selected snapshot, while
`VaeModel` and `PromptCache` identify the safe VAE representation and tiny
content-bound empty-prompt cache associated with it. These are backend cache
artifacts for the shared canonical weight, not duplicate model entries or
downloads. The LCM VAE remains an unavoidable safe derivation from its
canonical PyTorch archive; full-v1 maps its canonical VAE Safetensors
directly.

The Python worker fixes its generator seed to `12345`; the harness uses the
same default and accepts an optional unsigned `Seed` override for validation.
Both output paths preserve the worker's antialiased resize,
`match_input_res=false` dimensions, and `(depth-min)/(1-min)` normalization.

Capability reporting advertises host compatibility plus asynchronous D3D12
GPU resources, external synchronization, cancellation, and GPU-resident
output when exact adapter-local interop is available.

The Windows Release ABI and full-graph harness gates pass for both
`prs-eth/marigold-lcm-v1-0` and `prs-eth/marigold-v1-0` on the RX 9070.
They cover the 768x64 image contract, model/sidecar binding, fixed seed,
normalization, correlation, and output-lease lifetime. The underlying exact
image/tensor comparisons remain validated on all three GPUs as reported
above.

## InferBridge ABI2 external resources

The ABI2 harness consumes Core-owned shared D3D12 BGRA8/RGBA8 textures and
producer fences and writes processed-size normalized depth directly to a
Core-owned shared R32_FLOAT texture before signaling Core's fence. Input,
output, and synchronization handles remain borrowed. A persistent worker
keeps public submit asynchronous and bounds admission to three jobs.

Preprocessing, seeded noise, diffusion, VAE, normalization, and output remain
on Vulkan. Prompt and the eleven fixed LCM/full-v1 timestep embeddings are
uploaded once at model load; the tracked transfer counters remain unchanged
during every frame. The graph uses bounded submissions to recycle diffusion
intermediates, with the external producer wait only on preprocessing and the
external consumer signal only on final output. There is no queue-wide idle or
host fallback under the GPU capability.

| Variant | Relative L1 | Maximum absolute | Submit return | Upload/download delta |
|---|---:|---:|---:|---:|
| LCM v1 | `0.000811183` | `0.00300503` | `0.0097 ms` (3 frames) | `0 / 0` bytes |
| Full v1 | `0.0000580178` | `0.000198424` | `0.0073 ms` | `0 / 0` bytes |

The dedicated 768x64 D3D12/Vulkan canary also proves future-valued producer
fence import (`completed=0`, requested wait `1`), exact adapter selection,
sourceFrameId correlation, finite varying R32 output, repeated frame cleanup,
and clean model/runtime shutdown on the Radeon RX 9070.

## E7 tiled attention follow-up (2026-09-08)

E7 remains enabled with `MARIGOLD_ATTENTION_TILING=1`, now restricted to
requested FP32 precision. Missing/0 retains the scalar implementation.
FP16 and INT8 always use the original path, even with the flag set.

One fixed 1920x1080 real video frame, converted to BT.709 limited I420,
was measured through the application GPU path with one warmup and one
measured inference per configuration. These are focused confirmation timings,
not a new repeated performance sweep. Each original/tiled pair used the same
frame and compared all 66,048 finite output floats. All six pairs passed
relative L1 <= 1e-4 and maximum error / original depth range <= 1e-3.

| Variant | GPU | Original ms | E7 ms | Latency reduction | Relative L1 |
|---|---|---:|---:|---:|---:|
| lcm-v1 | AMD Radeon RX 9070 | 333.525 | 273.085 | 18.1% | 2.05e-06 |
| lcm-v1 | AMD Radeon RX 6700 XT | 606.132 | 398.227 | 34.3% | 1.85e-06 |
| lcm-v1 | NVIDIA GeForce GTX 1080 | 685.727 | 612.401 | 10.7% | 1.87e-06 |
| full-v1 | AMD Radeon RX 9070 | 1405.338 | 1001.686 | 28.7% | 4.58e-06 |
| full-v1 | AMD Radeon RX 6700 XT | 2458.350 | 1316.264 | 46.5% | 1.35e-06 |
| full-v1 | NVIDIA GeForce GTX 1080 | 2304.078 | 1938.482 | 15.9% | 3.78e-06 |

The initial RX 9070 LCM INT8 pair failed: relative L1 0.0245722,
maximum normalized depth error 0.425765. A repeated original run was
bit-identical; FP32 passed with the same frame. The new precision guard
restores bit-identical INT8 output even when the E7 flag is set. All 61
compiled shaders match the earlier candidate byte-for-byte; only host-side
precision eligibility changed.

A 20-second 60 Hz streaming check on RX 9070 LCM returned zero measured
outputs for BOTH original and E7, after successful warmup. Streaming therefore
has an existing failure in this setup; no throughput improvement is claimed.
Further streaming runs were stopped. WDDM sampled dedicated memory was
4550.1 MiB original and 4574.9 MiB E7; these sparse process totals do not
establish a memory improvement or regression.

Raw logs, invocation/environment receipts, input/output hashes and dumps are
under `.BuildComponents/Benchmarks/depth-optimization-20260908/E7/followup`
in the DeepDesktopGodot checkout. The exact frame hash is
`3fd0680a2581e662908ede367c9dcefb23828a001472622110653736255b6505`;
guarded DLL SHA-256 is
`38176293cf412047276525bb43478a2420bd7be4a07e74fefeec5d22a2e75660`.
This validates FP32 accuracy on the tested frame and confirms latency gains;
it does not establish general video accuracy or successful live streaming.
