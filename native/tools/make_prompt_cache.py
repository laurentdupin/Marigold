"""Derive Marigold's constant two-token empty-prompt embedding."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

TEXT_SHA = "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f"
MODELS = {
    "lcm": {
        "revision": "04a73502f7fd8fc5e59947b9df3b2266d71d6849",
        "unet": "953f1ea06169fc6c358b09ecc96a6ee32515e81540442d16239f82348ea62614",
        "vae": "a4302e1efa25f3a47ceb7536bc335715ad9d1f203e90c2d25507600d74006e89",
        "vae_file": "diffusion_pytorch_model.bin",
    },
    "full": {
        "revision": "f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c",
        "unet": "da9c13e214461c2cf82e4a0f125d914976522e53806a54d508e30ea5b8cd67f2",
        "vae": "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a",
        "vae_file": "diffusion_pytorch_model.fp16.safetensors",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def put(header: bytearray, offset: int, size: int, text: str) -> None:
    encoded = text.encode("ascii") + b"\0"
    if len(encoded) > size:
        raise ValueError("metadata field is too long")
    header[offset : offset + len(encoded)] = encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("--variant", choices=MODELS, required=True)
    parser.add_argument("--reuse-prompt-cache", type=Path)
    args = parser.parse_args()
    root = Path(args.snapshot)
    model = MODELS[args.variant]
    files = {
        "UNet": (
            root / "unet" / "diffusion_pytorch_model.fp16.safetensors",
            model["unet"],
        ),
        "text encoder": (
            root / "text_encoder" / "model.fp16.safetensors",
            TEXT_SHA,
        ),
    }
    for label, (path, expected) in files.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )

    # The LCM VAE used by the native harness is a separate canonical artifact,
    # because the creator's fp16 snapshot does not contain the original .bin
    # file. Its dedicated VAE converter verifies that artifact's immutable
    # hash. The prompt cache depends only on the tokenizer/text encoder and
    # UNet snapshot, so requiring the separate VAE inside this snapshot made a
    # clean product installation impossible. The fixed VAE hash remains in the
    # cache header to bind both independently verified derived artifacts to the
    # same official model publication.
    if args.reuse_prompt_cache:
        source = args.reuse_prompt_cache.read_bytes()
        if (
            len(source) != 512 + 2 * 1024 * 4
            or source[:8] != b"MARPRM01"
            or struct.unpack_from("<IIII", source, 8)
            != (1, 512, 2, 1024)
        ):
            raise RuntimeError("invalid source Marigold prompt cache")
        prompt_payload = source[512:]
    else:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer

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
            raise RuntimeError(
                f"unexpected empty-prompt shape: {prompt.shape}"
            )
        prompt_payload = prompt.numpy().tobytes()

    header = bytearray(512)
    header[:8] = b"MARPRM01"
    struct.pack_into("<IIII", header, 8, 1, 512, 2, 1024)
    put(header, 24, 41, model["revision"])
    put(header, 65, 65, model["unet"])
    put(header, 130, 65, model["vae"])
    put(header, 195, 65, TEXT_SHA)
    Path(args.output).write_bytes(header + prompt_payload)


if __name__ == "__main__":
    main()
