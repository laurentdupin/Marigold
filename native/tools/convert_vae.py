"""Convert the catalog's trusted PyTorch VAE into a safe native sidecar."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file


CANONICAL_SHA256 = (
    "a4302e1efa25f3a47ceb7536bc335715ad9d1f203e90c2d25507600d74006e89"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_bin")
    parser.add_argument("output_safetensors")
    args = parser.parse_args()
    source = Path(args.canonical_bin)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != CANONICAL_SHA256:
        raise RuntimeError(f"unexpected canonical VAE SHA-256: {digest}")

    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 248:
        raise RuntimeError("unexpected Marigold VAE state dictionary")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("VAE contains a non-tensor entry")
        if value.dtype != torch.float32 or value.layout != torch.strided:
            raise RuntimeError(f"unexpected VAE tensor representation: {name}")
        converted_name = name
        if ".mid_block.attentions." in name:
            converted_name = (
                converted_name.replace(".query.", ".to_q.")
                .replace(".key.", ".to_k.")
                .replace(".value.", ".to_v.")
                .replace(".proj_attn.", ".to_out.0.")
            )
        if converted_name in tensors:
            raise RuntimeError(f"duplicate converted VAE tensor: {converted_name}")
        tensors[converted_name] = value.detach().cpu().contiguous()
    save_file(
        tensors,
        args.output_safetensors,
        metadata={
            "format": "pt",
            "canonical_sha256": CANONICAL_SHA256,
            "converter": "marigold-vae-weights-only-v1",
        },
    )


if __name__ == "__main__":
    main()
