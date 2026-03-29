from __future__ import annotations

import numpy as np

from wafergeo.core.geometry import nearest_neighbor_distances_numpy, vtk_polys_to_triangles


class _FakePolys:
    def __init__(self, connectivity, offsets) -> None:
        self._connectivity = connectivity
        self._offsets = offsets

    def GetConnectivityArray(self):
        return self._connectivity

    def GetOffsetsArray(self):
        return self._offsets


def _vtk_to_numpy(arr):
    return arr


def test_vtk_polys_to_triangles_triangulates_polygon_fans() -> None:
    polys = _FakePolys(
        connectivity=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        offsets=np.array([0, 4, 6], dtype=np.int64),
    )

    faces = vtk_polys_to_triangles(polys, _vtk_to_numpy)

    assert faces.dtype == np.int32
    np.testing.assert_array_equal(
        faces,
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )


def test_vtk_polys_to_triangles_handles_missing_or_empty_arrays() -> None:
    empty = vtk_polys_to_triangles(_FakePolys(None, None), _vtk_to_numpy)
    assert empty.shape == (0, 3)

    short = vtk_polys_to_triangles(
        _FakePolys(
            connectivity=np.array([0, 1], dtype=np.int64),
            offsets=np.array([0], dtype=np.int64),
        ),
        _vtk_to_numpy,
    )
    assert short.shape == (0, 3)


def test_nearest_neighbor_distances_numpy_matches_simple_geometry() -> None:
    src = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    dst = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)

    out = nearest_neighbor_distances_numpy(src, dst, chunk_size=1)

    np.testing.assert_allclose(out, np.array([1.0, 1.0], dtype=np.float64))


def test_nearest_neighbor_distances_numpy_handles_empty_inputs() -> None:
    src = np.zeros((0, 3), dtype=np.float64)
    dst = np.zeros((2, 3), dtype=np.float64)

    out = nearest_neighbor_distances_numpy(src, dst)
    assert out.shape == (0,)

    infs = nearest_neighbor_distances_numpy(dst, src)
    np.testing.assert_array_equal(infs, np.array([np.inf, np.inf], dtype=np.float64))
