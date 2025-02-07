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
                model.eval(x=x, amplitude=1, center=-5, sigma=0.05 * 5)
                + model.eval(x=x, amplitude=1, center=5, sigma=0.05 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.1 * 5)
                + model.eval(x=x, amplitude=1, center=5, sigma=0.1 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.15 * 5)
                + model.eval(x=x, amplitude=1, center=5, sigma=0.15 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.2 * 5)
                + model.eval(x=x, amplitude=1, center=5, sigma=0.2 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.25 * 5)
                + model.eval(x=x, amplitude=1, center=5, sigma=0.25 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.3 * 5)
                + model.eval(x=x, amplitude=1, center=5, sigma=0.3 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.35 * 5)
                + model.eval(x=x, amplitude=1, center=5, sigma=0.35 * 5),
            ]
        ).T
        + rng.normal(size=(x.size, y.size)) * 0.01,
        coords={"x": x, "y": y},
        dims=("x", "y"),
    )


def test_fit_2d():
    data = get_test_data_2d()

    model = LorentzianModel(prefix="p0") + LorentzianModel(prefix="p1")
    guess = data.fit.guess(model=model)
    assert isinstance(guess, xr.DataArray)
    assert isinstance(guess[0].item(), lf.Parameters)

    result = data.fit.fit_with_corr(model=model, params=guess)
    assert isinstance(result, xr.DataArray)
    assert isinstance(result[0].item(), lf.model.ModelResult)

    params = result.params.parse()
    assert isinstance(params, xr.DataArray)
    assert isinstance(params[0].item(), lf.Parameters)
    sorted_result = result.params.sort("center")
    assert isinstance(sorted_result, xr.DataArray)
    assert isinstance(sorted_result[0].item(), lf.model.ModelResult)
    smoothend_result = result.params.smoothen("center", 5)
    assert isinstance(smoothend_result, xr.DataArray)
    assert isinstance(smoothend_result[0].item(), lf.model.ModelResult)
