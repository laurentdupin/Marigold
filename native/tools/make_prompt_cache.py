"""Derive Marigold's constant two-token empty-prompt embedding."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer


REVISION = "04a73502f7fd8fc5e59947b9df3b2266d71d6849"
UNET_SHA = "953f1ea06169fc6c358b09ecc96a6ee32515e81540442d16239f82348ea62614"
VAE_SHA = "a4302e1efa25f3a47ceb7536bc335715ad9d1f203e90c2d25507600d74006e89"
TEXT_SHA = "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f"


def put(header: bytearray, offset: int, size: int, text: str) -> None:
    encoded = text.encode("ascii") + b"\0"
    if len(encoded) > size:
        raise ValueError("metadata field is too long")
    header[offset : offset + len(encoded)] = encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    args = parser.parse_args()
    root = Path(args.snapshot)
    tokenizer = CLIPTokenizer.from_pretrained(
        root, subfolder="tokenizer", local_files_only=True
    )
    encoder = CLIPTextModel.from_pretrained(
        root,
        subfolder="text_encoder",
        variant="fp16",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    token_ids = tokenizer(
        "",
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    with torch.no_grad():
        prompt = encoder(token_ids)[0].contiguous().float()
    if tuple(prompt.shape) != (1, 2, 1024):
        raise RuntimeError(f"unexpected empty-prompt shape: {prompt.shape}")

    header = bytearray(512)
    header[:8] = b"MARPRM01"
    struct.pack_into("<IIII", header, 8, 1, 512, 2, 1024)
    put(header, 24, 41, REVISION)
    put(header, 65, 65, UNET_SHA)
    put(header, 130, 65, VAE_SHA)
    put(header, 195, 65, TEXT_SHA)
    Path(args.output).write_bytes(header + prompt.numpy().tobytes())


if __name__ == "__main__":
    main()
