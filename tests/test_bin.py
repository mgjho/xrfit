import numpy as np
import pytest
import xarray as xr


def test_bin_accessor():
    # Create a sample DataArray
    rng = np.random.default_rng()
    data = rng.random((10, 20))
    coords = {"x": np.linspace(0, 9, 10), "y": np.linspace(0, 19, 20)}
    da = xr.DataArray(data, coords=coords, dims=["x", "y"])

    # Apply the bin accessor with a multiplier
    binned_da = da.bin(x=2, y=2)

    # Check the new dimensions
    assert binned_da.sizes["x"] == 5
    assert binned_da.sizes["y"] == 10

    # Check the new coordinates
    np.testing.assert_allclose(binned_da["x"].values, np.linspace(0, 9, 5))
    np.testing.assert_allclose(binned_da["y"].values, np.linspace(0, 19, 10))

    # Check that the data has been interpolated correctly
    assert binned_da.isel(x=0, y=0).values == pytest.approx(
        da.isel(x=0, y=0).values, rel=1e-2
    )
    assert binned_da.isel(x=-1, y=-1).values == pytest.approx(
        da.isel(x=-1, y=-1).values, rel=1e-2
    )
