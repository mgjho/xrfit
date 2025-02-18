import lmfit as lf
import numpy as np
import xarray as xr


def test_get_arr_accessor():
    # Create a sample DataArray
    rng = np.random.default_rng()
    x = np.linspace(0, 10, 100)
    y = 3 * np.sin(x) + rng.normal(size=x.size)
    z = np.linspace(0, 1, 2)
    # data = xr.DataArray(y, coords={"x": x}, dims=["x"])

    # Define a model and fit it
    model = lf.models.SineModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)

    # Attach the model result to the DataArray
    # data.attrs["model_result"] = result
    xarr = xr.DataArray([result, result], coords={"z": z}, dims=["z"])
    # Apply the get_arr accessor
    init_fit = xarr.get_arr(attr_name="init_fit")
    best_fit = xarr.get_arr(attr_name="best_fit")
    residual = xarr.get_arr(attr_name="residual")
    data = xarr.get_arr(attr_name="data")

    # Check the new dimensions
    assert init_fit.sizes["x"] == 100
    assert best_fit.sizes["x"] == 100
    assert residual.sizes["x"] == 100
    assert data.sizes["x"] == 100

    # Check the new coordinates
    np.testing.assert_allclose(best_fit["x"].values, x)
    np.testing.assert_allclose(residual["x"].values, x)
    np.testing.assert_allclose(data["x"].values, x)
    np.testing.assert_allclose(init_fit["x"].values, x)
    # Check that the data has been processed correctly
    np.testing.assert_allclose(best_fit[0].values.flatten(), result.best_fit, rtol=1e-2)
    np.testing.assert_allclose(residual[0].values.flatten(), result.residual, rtol=1e-2)
    np.testing.assert_allclose(data[0].values.flatten(), result.data, rtol=1e-2)
    np.testing.assert_allclose(init_fit[0].values.flatten(), result.init_fit, rtol=1e-2)
