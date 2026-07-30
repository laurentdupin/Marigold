"""Generate deterministic Python CPU fixtures for native Marigold."""

from __future__ import annotations

import argparse
import enum
import importlib.machinery
import sys
import types
from pathlib import Path

import torch


_torchvision_modules = {
    name: types.ModuleType(name)
    for name in (
        "torchvision",
        "torchvision.transforms",
        "torchvision.io",
        "torchvision.transforms.v2",
        "torchvision.transforms.v2.functional",
    )
}
for _name, _module in _torchvision_modules.items():
    _module.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
    sys.modules[_name] = _module


class _InterpolationMode(enum.Enum):
    NEAREST = 0
    NEAREST_EXACT = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
    LANCZOS = 6


_torchvision_modules["torchvision.transforms"].InterpolationMode = (
    _InterpolationMode
)
_torchvision_modules["torchvision"].transforms = _torchvision_modules[
    "torchvision.transforms"
]
_torchvision_modules["torchvision"].io = _torchvision_modules[
    "torchvision.io"
]
_torchvision_modules["torchvision.transforms"].v2 = _torchvision_modules[
    "torchvision.transforms.v2"
]
_torchvision_modules["torchvision.transforms.v2"].functional = (
    _torchvision_modules["torchvision.transforms.v2.functional"]
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


def save(path: Path, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous().float()
    path.write_bytes(value.numpy().tobytes())
    print(path.name, tuple(value.shape), float(value.min()), float(value.max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument(
        "--variant", choices=("lcm", "full"), default="lcm"
    )
    args = parser.parse_args()
    root = Path(args.snapshot)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(23)

    size = args.size
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    rgb = torch.stack(
        (
            torch.sin(xx * 0.071),
            torch.cos(yy * 0.053),
            torch.sin((xx + yy) * 0.037),
        ),
        dim=0,
    ).unsqueeze(0)
    if args.variant == "lcm":
        vae = AutoencoderKL.from_pretrained(
            root,
            subfolder="vae",
            local_files_only=True,
            use_safetensors=False,
        ).eval()
    else:
        vae = AutoencoderKL.from_pretrained(
            root,
            subfolder="vae",
            variant="fp16",
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval()
    unet = UNet2DConditionModel.from_pretrained(
        root,
        subfolder="unet",
        variant="fp16",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    tokenizer = CLIPTokenizer.from_pretrained(
        root, subfolder="tokenizer", local_files_only=True
    )
    text = CLIPTextModel.from_pretrained(
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
    prompt = text(token_ids)[0]

    moments = vae.quant_conv(vae.encoder(rgb))
    mean, _ = torch.chunk(moments, 2, dim=1)
    rgb_latent = mean * vae.config.scaling_factor
    target_noise = torch.randn_like(rgb_latent)
    scheduler_type = LCMScheduler if args.variant == "lcm" else DDIMScheduler
    scheduler = scheduler_type.from_pretrained(
        root, subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(1 if args.variant == "lcm" else 10)
    target_latent = target_noise
    prediction = None
    for timestep in scheduler.timesteps:
        prediction = unet(
            torch.cat((rgb_latent, target_latent), dim=1),
            timestep,
            encoder_hidden_states=prompt,
        ).sample
        target_latent = scheduler.step(
            prediction, timestep, target_latent
        ).prev_sample
    decoded = vae.decode(
        target_latent / vae.config.scaling_factor,
        return_dict=False,
    )[0]
    depth = ((decoded.clip(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1)

    save(output / "rgb.bin", rgb)
    save(output / "prompt.bin", prompt)
    save(output / "rgb_latent.bin", rgb_latent)
    save(output / "target_noise.bin", target_noise)
    save(output / "unet_prediction.bin", prediction)
    save(output / "target_latent.bin", target_latent)
    save(output / "decoded.bin", decoded)
    save(output / "depth.bin", depth)
    print("timesteps", [int(value) for value in scheduler.timesteps])


if __name__ == "__main__":
    main()
