#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from native_linear_nf24_prepacked import (
    load_nf24_i16_prepacked_model,
    pack_nf24_i16_checkpoint,
)
from native_linear_rnf8_avx512_patch import apply_triposplat_native_rnf8_avx512_patch


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96, 64, bias=True)
        self.norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 17, bias=False)

    def forward(self, x):
        return self.fc2(torch.tanh(self.norm(self.fc1(x))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(0)
    source = ToyModel().eval()
    x = torch.randn(25, 96, dtype=torch.float32)
    reference = source(x)

    with tempfile.TemporaryDirectory(prefix="triposplat-nf24-smoke-") as temporary:
        root = Path(temporary)
        checkpoint = root / "toy.safetensors"
        packed = root / "packed"
        save_file(source.state_dict(), checkpoint.as_posix())
        manifest = pack_nf24_i16_checkpoint(
            checkpoint,
            packed,
            expected_linear_count=2,
        )
        resumed_manifest = pack_nf24_i16_checkpoint(
            checkpoint,
            packed,
            expected_linear_count=2,
        )
        assert resumed_manifest == manifest
        loaded = load_nf24_i16_prepacked_model(
            packed,
            ToyModel,
            verify_checksums=True,
        ).eval()
        metadata = apply_triposplat_native_rnf8_avx512_patch(
            loaded,
            enabled=True,
            include_regex=r".*",
            library_path=args.library.as_posix(),
            threads=args.threads,
            strict=True,
            stages=3,
            residual_mode="nf24_i16",
        )
        with torch.inference_mode():
            observed = loaded(x)
        diff = observed - reference
        rmse = torch.sqrt(torch.mean(diff.square())).item()
        max_abs = diff.abs().max().item()
        assert manifest["linear_count"] == 2
        assert metadata["selected_count"] == 2
        assert metadata["runtime"]["fallbacks"] == 0
        assert metadata["prepacked_load"]["source_checkpoint_loaded"] is False
        assert rmse < 1.0e-4 and max_abs < 5.0e-4, (rmse, max_abs)
        print({
            "status": "pass",
            "linear_count": 2,
            "rmse": rmse,
            "max_abs": max_abs,
            "packed_bytes": manifest["packed_bytes"],
            "source_checkpoint_loaded": False,
            "resume_verified": True,
            "fallbacks": metadata["runtime"]["fallbacks"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
