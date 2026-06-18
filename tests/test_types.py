import pytest

from pim_ge.utils.types import Grid, SourceLocation


def test_grid_uniform_shape():
    g = Grid.uniform((0, 100), (-50, 50), (0, 10), n=20)
    assert g.x.shape == (20,)
    assert g.y.shape == (20,)
    assert g.z.shape == (20,)


def test_grid_uniform_bounds():
    g = Grid.uniform((0, 100), (-50, 50), (0, 10), n=5)
    assert float(g.x[0]) == pytest.approx(0.0)
    assert float(g.x[-1]) == pytest.approx(100.0)


def test_source_location():
    s = SourceLocation(x=10.0, y=20.0, z=1.5)
    assert s.x == 10.0
    assert s.y == 20.0
    assert s.z == 1.5
