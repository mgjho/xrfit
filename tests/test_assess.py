import lmfit as lf
import numpy as np
import pytest
import xarray as xr


def test_assess_accessor():
    # Create a sample DataArray
    rng = np.random.default_rng()
    x = np.linspace(0, 10, 100)
    y = 3 * np.sin(x) + rng.normal(size=x.size)
    z = np.linspace(0, 1, 2)

    # Define a model and fit it
    model = lf.models.SineModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)

    # Attach the model result to the DataArray
    xarr = xr.DataArray([result, result], coords={"z": z}, dims=["z"])

    # Apply the assess accessor
    rsquared = xarr.assess.fit_stats(attr_name="rsquared")
    best_fit_max = xarr.assess.best_fit_max()
    best_fit_stat = xarr.assess.best_fit_stat(attr_name="rsquared")

    # Check the new dimensions
    assert rsquared.sizes["z"] == 2

    # Check the values
    assert rsquared[0].values == pytest.approx(result.rsquared, rel=1e-2)
    assert best_fit_max == {"z": 0}
    assert best_fit_stat == {"z": 0}
