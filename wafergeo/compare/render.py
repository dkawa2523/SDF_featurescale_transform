from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

LABEL_DIFFERENCE_LEGEND = {
    "mode": "label",
    "background": {"rgb": [245, 245, 245], "meaning": "void/background on both sides"},
    "match": {"rgb": [230, 190, 40], "meaning": "simulation and target both present and equal"},
    "mismatch": {
        "rgb": [70, 110, 220],
        "meaning": "simulation and target both present but label/value differs",
    },
    "simulation_only": {"rgb": [220, 70, 70], "meaning": "present only in simulation"},
    "target_only": {"rgb": [70, 160, 80], "meaning": "present only in target"},
}

BOUNDARY_DIFFERENCE_LEGEND = {
    "mode": "boundary",
    "background": {"rgb": [245, 245, 245], "meaning": "no material boundary on both sides"},
    "match": {
        "rgb": [230, 190, 40],
        "meaning": "simulation and target material-boundary pixels overlap",
    },
    "mismatch": {
        "rgb": [70, 110, 220],
        "meaning": "reserved for label mismatch; boundary mode normally uses only-only colors",
    },
    "simulation_only": {
        "rgb": [220, 70, 70],
        "meaning": "material-boundary pixel present only in simulation",
    },
    "target_only": {
        "rgb": [70, 160, 80],
        "meaning": "material-boundary pixel present only in target",
    },
}


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_rgb_png(path: str | Path, rgb: np.ndarray) -> None:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("rgb image must be shape (H,W,3)")
    height, width, _ = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    payload += _png_chunk(b"IEND", b"")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)


def _display_scale(shape_yx: tuple[int, int], *, min_size: int = 384, max_scale: int = 8) -> int:
    smallest = max(min(shape_yx), 1)
    if smallest >= min_size:
        return 1
    return max(1, min(max_scale, int(np.ceil(min_size / smallest))))


def write_difference_png(
    path: str | Path,
    sim_label: np.ndarray,
    target_label: np.ndarray,
    *,
    sim_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
) -> int:
    sim = np.asarray(sim_label)
    target = np.asarray(target_label)
    if sim.shape != target.shape:
        raise ValueError("sim_label and target_label must have the same shape")

    sim_solid = np.asarray(sim_mask, dtype=bool) if sim_mask is not None else sim != 0
    target_solid = (
        np.asarray(target_mask, dtype=bool) if target_mask is not None else target != 0
    )
    if sim_solid.shape != sim.shape or target_solid.shape != sim.shape:
        raise ValueError("sim_mask and target_mask must match label shape")

    rgb = np.full(sim.shape + (3,), 245, dtype=np.uint8)
    same = sim_solid & target_solid & (sim == target)
    mismatch = sim_solid & target_solid & (sim != target)
    only_sim = sim_solid & ~target_solid
    only_target = target_solid & ~sim_solid

    rgb[same] = np.array([230, 190, 40], dtype=np.uint8)
    rgb[mismatch] = np.array([70, 110, 220], dtype=np.uint8)
    rgb[only_sim] = np.array([220, 70, 70], dtype=np.uint8)
    rgb[only_target] = np.array([70, 160, 80], dtype=np.uint8)
    scale = _display_scale(sim.shape)
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    write_rgb_png(path, rgb)
    return scale


def difference_summary(
    sim_label: np.ndarray,
    target_label: np.ndarray,
    *,
    sim_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    mode: str,
) -> dict[str, int | str]:
    sim = np.asarray(sim_label)
    target = np.asarray(target_label)
    if sim.shape != target.shape:
        raise ValueError("sim_label and target_label must have the same shape")

    sim_solid = np.asarray(sim_mask, dtype=bool) if sim_mask is not None else sim != 0
    target_solid = (
        np.asarray(target_mask, dtype=bool) if target_mask is not None else target != 0
    )
    if sim_solid.shape != sim.shape or target_solid.shape != sim.shape:
        raise ValueError("sim_mask and target_mask must match label shape")

    same = sim_solid & target_solid & (sim == target)
    mismatch = sim_solid & target_solid & (sim != target)
    only_sim = sim_solid & ~target_solid
    only_target = target_solid & ~sim_solid
    background = ~(sim_solid | target_solid)
    changed = mismatch | only_sim | only_target
    return {
        "mode": mode,
        "height": int(sim.shape[0]),
        "width": int(sim.shape[1]),
        "background_pixels": int(background.sum()),
        "match_pixels": int(same.sum()),
        "mismatch_pixels": int(mismatch.sum()),
        "simulation_only_pixels": int(only_sim.sum()),
        "target_only_pixels": int(only_target.sum()),
        "changed_pixels": int(changed.sum()),
    }


def write_difference_summary_json(
    path: str | Path,
    sim_label: np.ndarray,
    target_label: np.ndarray,
    *,
    sim_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    mode: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            difference_summary(
                sim_label,
                target_label,
                sim_mask=sim_mask,
                target_mask=target_mask,
                mode=mode,
            ),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_difference_legend_json(
    path: str | Path,
    *,
    mode: str = "label",
    display_scale: int = 1,
) -> None:
    if mode == "label":
        legend = LABEL_DIFFERENCE_LEGEND
    elif mode == "boundary":
        legend = BOUNDARY_DIFFERENCE_LEGEND
    else:
        raise ValueError(f"unsupported difference legend mode: {mode}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**legend, "display_scale": int(display_scale)}
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
