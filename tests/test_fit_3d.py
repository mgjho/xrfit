import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel


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


def test_fit_3d():
    data = get_test_data_3d()
    model = LorentzianModel()
    guess = data.fit.guess(model=model)
    assert isinstance(guess, xr.DataArray)
    assert isinstance(guess[0, 0].item(), lf.Parameters)

    result = data.fit.fit_with_corr(model=model, params=guess)
    assert isinstance(result, xr.DataArray)
    assert isinstance(result[0, 0].item(), lf.model.ModelResult)
