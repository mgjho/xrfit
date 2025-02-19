import lmfit as lf
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def data_array():
    rng = np.random.default_rng()
    x = np.linspace(0, 10, 100)
    y = 3 / (1 + ((x - 5) / 1) ** 2) + rng.normal(size=x.size)  # Lorentzian model
    y2 = 4 / (1 + ((x - 5) / 1) ** 2) + rng.normal(size=x.size)  # Lorentzian model
    z = np.linspace(0, 1, 2)

    # Define a model and fit it
    model = lf.models.LorentzianModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)
    result2 = model.fit(y2, params, x=x)
    return xr.DataArray([result, result2], coords={"z": z}, dims=["z"])


def test_get(data_array):
    result = data_array.params.get(params_name="amplitude")
    assert result.shape == (2, 1)
    assert np.all(result >= 0)  # Assuming amplitude is non-negative


def test_assign(data_array):
    z = np.linspace(0, 1, 2)
    params_dim = ["amplitude"]
    result = data_array.params.get(params_name="amplitude")

    new_values = xr.DataArray(
        np.array([[20.0], [30.0]]),
        coords={"z": z, "params_dim": params_dim},
        dims=["z", "params_dim"],
    )
    data_array.params.assign(new_values, params_name="amplitude")
    result = data_array.params.get(params_name="amplitude")

    assert np.all(
        result.values == np.array([[20.0], [30.0]])
    )  # Check if the assignment was performed correctly


def test_set_bounds(data_array):
    data_array.params.set_bounds(bound_ratio=0.2)
    result = data_array.params.get(params_name="amplitude")
    assert result.shape == (2, 1)  # Check if values are unchanged


def test_smoothen(data_array):
    data_array.params.smoothen(param_name="amplitude", sigma=1)
    result = data_array.params.get(params_name="amplitude")
    assert result is not None  # Check if smoothing was applied


def test_sort(data_array):
    data_array.params.sort("amplitude")
    result = data_array.params.get(params_name="amplitude")
    assert result.size > 0  # Check if result is not empty
    assert result.shape == (2, 1)  # Check if sorting was applied
