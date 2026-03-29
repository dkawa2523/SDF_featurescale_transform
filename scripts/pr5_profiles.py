from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["core", "full"]


@dataclass(frozen=True)
class ProfileSpec:
    name: ProfileName
    install_argv: tuple[str, ...]
    test_argv: tuple[str, ...]


def _make_install_argv(extra_spec: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        f".[{extra_spec}]",
    )


PROFILE_SPECS: dict[ProfileName, ProfileSpec] = {
    "core": ProfileSpec(
        name="core",
        install_argv=_make_install_argv("dev"),
        test_argv=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/core",
            "tests/io",
            "tests/label",
            "tests/metrics",
            "tests/reports/test_registry.py",
            "tests/reports/test_heavy_plot_policy.py",
            "tests/pipelines/test_vti_correspondence_audit.py",
            "tests/pipelines/test_vti_sdf_mesh_preview.py",
            "tests/sdf/test_engine_optional_deps.py",
            "tests/mesh/test_vtk_optional.py",
            "tests/observe/test_contour_optional.py",
            "tests/sem/test_readers.py",
        ),
    ),
    "full": ProfileSpec(
        name="full",
        install_argv=_make_install_argv("dev,scipy,vtk,observe,sem,viz,parquet,zarr"),
        test_argv=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ),
    ),
}


def get_profile_spec(profile: ProfileName) -> ProfileSpec:
    return PROFILE_SPECS[profile]


def render_command(argv: tuple[str, ...]) -> str:
    return shlex.join(argv)
