import dill
import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel


def test_fit_3d():
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
    with open("fit_result_3d.dill", "wb") as f:
        dill.dump(result, f)


def test_fit():
    rng = np.random.default_rng(seed=0)
    x = np.linspace(-10, 10, 200)
    y = np.linspace(0, 2, 3)
    model = LorentzianModel()
    data = xr.DataArray(
        np.stack(
            [
                model.eval(x=x, amplitude=1, center=0, sigma=0.1),
                model.eval(x=x, amplitude=1, center=0, sigma=0.2),
                model.eval(x=x, amplitude=1, center=0, sigma=0.3),
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
    with open("fit_result.dill", "wb") as f:
        dill.dump(result, f)
