import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel


def get_test_data_2d():
    rng = np.random.default_rng(seed=0)
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 7)
    model = LorentzianModel()
    return xr.DataArray(
        np.stack(
            [
                model.eval(x=x, amplitude=1, center=0, sigma=0.05 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.1 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.15 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.2 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.25 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.3 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.35 * 5),
            ]
        ).T
        + rng.normal(size=(x.size, y.size)) * 0.01,
        coords={"x": x, "y": y},
        dims=("x", "y"),
    )


def get_test_data_3d():
    rng = np.random.default_rng(seed=0)
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 2)
    z = np.linspace(0, 2, 3)
    model = LorentzianModel()
    return xr.DataArray(
        np.stack(
            [
                [
                    model.eval(x=x, amplitude=1, center=0, sigma=0.1 * 5),
                    model.eval(x=x, amplitude=1, center=0, sigma=0.2 * 5),
                ],
                [
                    model.eval(x=x, amplitude=1, center=0, sigma=0.2 * 5),
                    model.eval(x=x, amplitude=1, center=0, sigma=0.3 * 5),
                ],
                [
                    model.eval(x=x, amplitude=1, center=0, sigma=0.3 * 5),
                    model.eval(x=x, amplitude=1, center=0, sigma=0.4 * 5),
                ],
            ],
        ).transpose()
        + rng.normal(size=(x.size, y.size, z.size)) * 0.01,
        coords={"x": x, "y": y, "z": z},
        dims=("x", "y", "z"),
    )


def get_test_result(data):
    model = LorentzianModel()
    guess = data.fit.guess(model=model)
    return data.fit(model=model, params=guess)


def test_fit_3d(qtbot):
    rng = np.random.default_rng(seed=0)
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 2)
    z = np.linspace(0, 2, 3)
    model = LorentzianModel()
    data = xr.DataArray(
        np.stack(
            [
                [
                    model.eval(x=x, amplitude=1, center=0, sigma=0.1),
                    model.eval(x=x, amplitude=1, center=0, sigma=0.2),
                ],
                [
                    model.eval(x=x, amplitude=1, center=0, sigma=0.2),
                    model.eval(x=x, amplitude=1, center=0, sigma=0.3),
                ],
                [
                    model.eval(x=x, amplitude=1, center=0, sigma=0.3),
                    model.eval(x=x, amplitude=1, center=0, sigma=0.4),
                ],
            ],
        ).transpose()
        + rng.normal(size=(x.size, y.size, z.size)) * 0.01,
        coords={"x": x, "y": y, "z": z},
        dims=("x", "y", "z"),
    )

    assert isinstance(data, xr.DataArray)
    assert data.shape == (x.size, y.size, z.size)
    assert data.dims == ("x", "y", "z")

    guess = data.fit.guess(model=model)
    assert isinstance(guess, xr.DataArray)
    assert guess.shape == (y.size, z.size)
    assert guess.dims == ("y", "z")
    assert isinstance(guess[0, 0].item(), lf.Parameters)

    result = data.fit(model=model, params=guess)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (y.size, z.size)
    assert result.dims == ("y", "z")
    assert isinstance(result[0, 0].item(), lf.model.ModelResult)


def test_fit(qtbot):
    rng = np.random.default_rng(seed=0)
    x = np.linspace(-10, 10, 200)
    y = np.linspace(0, 2, 3)
    model = LorentzianModel()
    data = xr.DataArray(
        np.stack(
            [
                model.eval(x=x, amplitude=1, center=0, sigma=0.1 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.2 * 5),
                model.eval(x=x, amplitude=1, center=0, sigma=0.3 * 5),
            ]
        ).T
        + rng.normal(size=(x.size, y.size)) * 0.01,
        coords={"x": x, "y": y},
        dims=("x", "y"),
    )

    assert isinstance(data, xr.DataArray)
    assert data.shape == (x.size, y.size)
    assert data.dims == ("x", "y")

    guess = data.fit.guess(model=model)
    assert isinstance(guess, xr.DataArray)
    assert guess.shape == (y.size,)
    assert guess.dims == ("y",)
    assert isinstance(guess[0].item(), lf.Parameters)

    result = data.fit(model=model, params=guess)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (y.size,)
    assert result.dims == ("y",)
    assert isinstance(result[0].item(), lf.model.ModelResult)

    params = result.params()
    assert isinstance(params, xr.DataArray)
    assert isinstance(params[0].item(), lf.Parameters)
    sorted_result = result.params.sort("center")
    assert isinstance(sorted_result, xr.DataArray)
    assert isinstance(sorted_result[0].item(), lf.model.ModelResult)
    smoothend_result = result.params.smoothen("center", 5)
    assert isinstance(smoothend_result, xr.DataArray)
    assert isinstance(smoothend_result[0].item(), lf.model.ModelResult)
