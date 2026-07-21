#!/usr/bin/env python3
"""Streaming export helpers for TripoSplat Gaussian objects.

These functions replace only the export boundary. They do not alter inference.
They are designed to produce the same PLY/SPLAT bytes as the official exporter
while avoiding the final full structured-array/materialized-record buffers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((R.shape[0], 4), dtype=R.dtype)
    s = np.sqrt(np.maximum(trace + 1, 0)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / np.where(s != 0, s, 1)
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / np.where(s != 0, s, 1)
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / np.where(s != 0, s, 1)
    m01 = (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2]) & (s == 0)
    s1 = np.sqrt(np.maximum(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], 0)) * 2
    q[m01, 0] = (R[m01, 2, 1] - R[m01, 1, 2]) / s1[m01]
    q[m01, 1] = 0.25 * s1[m01]
    q[m01, 2] = (R[m01, 0, 1] + R[m01, 1, 0]) / s1[m01]
    q[m01, 3] = (R[m01, 0, 2] + R[m01, 2, 0]) / s1[m01]
    m11 = (R[:, 1, 1] > R[:, 0, 0]) & (R[:, 1, 1] >= R[:, 2, 2]) & (s == 0)
    s2 = np.sqrt(np.maximum(1 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], 0)) * 2
    q[m11, 0] = (R[m11, 0, 2] - R[m11, 2, 0]) / s2[m11]
    q[m11, 1] = (R[m11, 0, 1] + R[m11, 1, 0]) / s2[m11]
    q[m11, 2] = 0.25 * s2[m11]
    q[m11, 3] = (R[m11, 1, 2] + R[m11, 2, 1]) / s2[m11]
    m21 = (R[:, 2, 2] > R[:, 0, 0]) & (R[:, 2, 2] > R[:, 1, 1]) & (s == 0)
    s3 = np.sqrt(np.maximum(1 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], 0)) * 2
    q[m21, 0] = (R[m21, 1, 0] - R[m21, 0, 1]) / s3[m21]
    q[m21, 1] = (R[m21, 0, 2] + R[m21, 2, 0]) / s3[m21]
    q[m21, 2] = (R[m21, 1, 2] + R[m21, 2, 1]) / s3[m21]
    q[m21, 3] = 0.25 * s3[m21]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _chunks(n: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, n, chunk_size):
        yield slice(start, min(n, start + chunk_size))


def _as_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _header(attributes: list[str], count: int) -> bytes:
    text = "ply\nformat binary_little_endian 1.0\n"
    text += f"element vertex {count}\n"
    for name in attributes:
        text += f"property float {name}\n"
    text += "end_header\n"
    return text.encode("ascii")


def _dtype(attributes: list[str]) -> np.dtype:
    return np.dtype([(name, "f4") for name in attributes])


def _ply_chunk(gaussian, sl: slice, transform, dtype: np.dtype) -> np.ndarray:
    xyz = _as_numpy(gaussian.get_xyz[sl])
    normals = np.zeros_like(xyz)
    f_dc = _as_numpy(gaussian._features_dc[sl].detach().transpose(1, 2).flatten(start_dim=1).contiguous())
    opacities = _as_numpy(gaussian._inverse_opacity_activation(gaussian.get_opacity[sl]))
    scale = _as_numpy(torch.log(gaussian.get_scaling[sl]))
    rotation = _as_numpy(gaussian._rotation[sl] + gaussian.rots_bias[None, :])
    if transform is not None:
        mat = np.array(transform)
        xyz = np.matmul(xyz, mat.T)
        R_mat = _quat_to_matrix(rotation)
        R_mat = np.matmul(mat, R_mat)
        rotation = _matrix_to_quat(R_mat)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)))
    return elements


def save_ply_lowmem(gaussian, path: str | Path, transform=None, chunk_size: int = 32768) -> None:
    if transform is None:
        transform = gaussian._DEFAULT_TRANSFORM
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attributes = gaussian.construct_list_of_attributes()
    dtype = _dtype(attributes)
    count = int(gaussian.get_xyz.shape[0])
    with path.open("wb") as f:
        f.write(_header(attributes, count))
        for sl in _chunks(count, max(1, int(chunk_size))):
            f.write(_ply_chunk(gaussian, sl, transform, dtype).tobytes())


def _transformed_xyz_rot_chunk(gaussian, indices: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    if transform is None:
        transform = gaussian._DEFAULT_TRANSFORM
    mat = np.array(transform, dtype=np.float32)
    idx = torch.as_tensor(indices, device=gaussian.get_xyz.device, dtype=torch.long)
    xyz = _as_numpy(gaussian.get_xyz.index_select(0, idx)).astype(np.float32)
    rotation = _as_numpy((gaussian._rotation.index_select(0, idx) + gaussian.rots_bias[None, :]))
    xyz = np.matmul(xyz, mat.T)
    R_mat = _quat_to_matrix(rotation)
    R_mat = np.matmul(mat, R_mat)
    rotation = _matrix_to_quat(R_mat)
    return xyz, rotation


def _splat_sort_order(gaussian, chunk_size: int) -> np.ndarray:
    count = int(gaussian.get_xyz.shape[0])
    keys = np.empty(count, dtype=np.float32)
    for sl in _chunks(count, chunk_size):
        opacity = _as_numpy(gaussian.get_opacity[sl])[:, 0]
        scale = _as_numpy(gaussian.get_scaling[sl]).astype(np.float32)
        keys[sl] = -opacity * np.prod(scale, axis=-1)
    return np.argsort(keys)


def _pack_splat_chunk(gaussian, indices: np.ndarray, transform) -> bytes:
    idx = torch.as_tensor(indices, device=gaussian.get_xyz.device, dtype=torch.long)
    xyz, rotation = _transformed_xyz_rot_chunk(gaussian, indices, transform)
    scale = _as_numpy(gaussian.get_scaling.index_select(0, idx)).astype(np.float32)
    opacity = _as_numpy(gaussian.get_opacity.index_select(0, idx))
    f_dc = _as_numpy(gaussian._features_dc.index_select(0, idx))
    rgb = np.clip((f_dc[:, 0, :] * 0.28209479177387814 + 0.5) * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(opacity[:, 0:1] * 255, 0, 255).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=1)
    rot = rotation / np.linalg.norm(rotation, axis=-1, keepdims=True)
    rot_u8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)
    data = np.concatenate([
        xyz.astype(np.float32).view(np.uint8).reshape(-1, 12),
        scale.astype(np.float32).view(np.uint8).reshape(-1, 12),
        rgba.reshape(-1, 4),
        rot_u8.reshape(-1, 4),
    ], axis=1).reshape(-1)
    return data.tobytes()


def save_splat_lowmem(gaussian, path: str | Path, transform=None, chunk_size: int = 32768) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = _splat_sort_order(gaussian, max(1, int(chunk_size)))
    with path.open("wb") as f:
        for start in range(0, order.shape[0], max(1, int(chunk_size))):
            f.write(_pack_splat_chunk(gaussian, order[start:start + chunk_size], transform))
