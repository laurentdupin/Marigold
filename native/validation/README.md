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

`marigold_native.dll` exposes ABI 2 lifecycle and stable seeded inference.
The exact validation entry accepts explicit target noise to remove stochastic
ambiguity.

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
