"""Convert the catalog's trusted PyTorch VAE into a safe native sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file


CANONICAL_SHA256 = (
    "a4302e1efa25f3a47ceb7536bc335715ad9d1f203e90c2d25507600d74006e89"
)


def canonicalize_safetensors_header(path: Path) -> None:
    """Replace safetensors' unordered metadata JSON with canonical JSON."""
    temporary = path.with_name(path.name + ".canonical.tmp")
    try:
        with path.open("rb") as source:
            encoded_length = source.read(8)
            if len(encoded_length) != 8:
                raise RuntimeError("truncated safetensors header length")
            header_length = struct.unpack("<Q", encoded_length)[0]
            encoded_header = source.read(header_length)
            if len(encoded_header) != header_length:
                raise RuntimeError("truncated safetensors header")
            header = json.loads(encoded_header)
            canonical = json.dumps(
                header, ensure_ascii=False, separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            canonical += b" " * (-len(canonical) % 8)
            with temporary.open("wb") as destination:
                destination.write(struct.pack("<Q", len(canonical)))
                destination.write(canonical)
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
            "converter": "marigold-vae-weights-only-v2",
        },
    )
    canonicalize_safetensors_header(Path(args.output_safetensors))


if __name__ == "__main__":
    main()
