from koopman_lm.config import build_config


def test_50m_production_config():
    c = build_config('50m')
    assert c.d_model == 384
    assert c.n_layers == 17
    assert c.ska_rank == 24
    assert tuple(c.ska_layer_indices) == (3, 7, 11, 15)
    assert c.ska_mode == 'parallel'
    assert c.ska_prefix_scan is True
    assert c.mlp_type == 'swiglu'


def test_180m_production_config():
    c = build_config('180m')
    assert c.d_model == 640
    assert c.n_layers == 25
    assert c.ska_rank == 24
    assert tuple(c.ska_layer_indices) == (3, 7, 11, 15, 19, 23)
    assert c.ska_mode == 'parallel'
    assert c.ska_prefix_scan is True
    assert c.mlp_type == 'swiglu'


def test_package_data_points_at_the_real_cuda_sources():
    """The csrc/*.cu glob must name a package that exists.

    A stale key here excludes the CUDA sources from any built wheel or sdist.
    Editable installs mask it completely, so only this assertion catches it.

    The TOML is scanned textually rather than parsed: this project supports
    Python 3.10 (pyproject requires-python = ">=3.10") and tomllib is 3.11+.
    Depending on the third-party tomli would add a test-only dependency for
    one assertion.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text()

    m = re.search(r"^\[tool\.setuptools\.package-data\]\s*$(.*?)(?=^\[|\Z)",
                  text, re.MULTILINE | re.DOTALL)
    assert m, "pyproject.toml has no [tool.setuptools.package-data] section"

    entries = re.findall(r'^\s*"([^"]+)"\s*=\s*\[([^\]]*)\]',
                         m.group(1), re.MULTILINE)
    assert entries, "package-data section is empty; the .cu sources would not ship"

    for pkg, globs_raw in entries:
        pkg_dir = root / Path(*pkg.split("."))
        assert pkg_dir.is_dir(), (
            f"package-data names {pkg!r}, which is not a directory")
        for g in re.findall(r'"([^"]+)"', globs_raw):
            assert list(pkg_dir.glob(g)), f"{pkg!r} glob {g!r} matches no files"
