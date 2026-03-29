from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from wafergeo.sem.readers import read_contours_csv, read_contours_json, read_sem_image


def test_read_contours_json_closed_loop(tmp_path: Path) -> None:
    payload = {
        "coord_system": "nm",
        "units": "nm",
        "loops": [
            {
                "id": "outer_0",
                "role": "outer",
                "is_closed": True,
                "points": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0], [10.0, 10.0]],
            }
        ],
    }
    path = tmp_path / "contours.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    raw = read_contours_json(path)
    assert raw.coord_system == "nm"
    assert raw.units == "nm"
    assert len(raw.loops_raw) == 1
    assert raw.loops_raw[0].points_xy.shape == (5, 2)


def test_read_contours_csv_grouped_by_loop_id(tmp_path: Path) -> None:
    path = tmp_path / "contours.csv"
    path.write_text(
        "loop_id,role,x,y\n"
        "a,outer,10,10\n"
        "a,outer,40,10\n"
        "a,outer,40,40\n"
        "a,outer,10,40\n"
        "a,outer,10,10\n",
        encoding="utf-8",
    )

    raw = read_contours_csv(path)
    assert len(raw.loops_raw) == 1
    assert raw.loops_raw[0].loop_id == "a"
    assert raw.loops_raw[0].role == "outer"


def test_read_sem_image_without_pillow_has_install_hint(tmp_path: Path) -> None:
    if importlib.util.find_spec("PIL") is not None:
        pytest.skip("Pillow is installed; missing dependency path is not applicable")

    fake_path = tmp_path / "image.png"
    fake_path.write_bytes(b"not-an-image")
    with pytest.raises(ImportError) as exc_info:
        read_sem_image(fake_path)
    assert "wafergeo[sem]" in str(exc_info.value)
