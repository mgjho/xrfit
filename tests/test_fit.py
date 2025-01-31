import dill
import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel

import xrfit


def test_fit():
    rng = np.random.default_rng(seed=0)
    x = np.linspace(-10, 10, 200)
    model = LorentzianModel()
    data = xr.DataArray(
        np.stack(
            [
                model.eval(x=x, amplitude=1, center=0, sigma=0.1),
                model.eval(x=x, amplitude=1, center=0, sigma=0.2),
                model.eval(x=x, amplitude=1, center=0, sigma=0.3),
            ]
        ).T
        + rng.normal(size=(x.size, 3)) * 0.01,
        coords={"x": x, "y": [0, 1, 2]},
    )

    assert isinstance(data, xr.DataArray)
    assert data.shape == (200, 3)
    assert data.dims == ("x", "y")

    guess = data.fit.guess(model=model)
    assert isinstance(guess, xr.DataArray)
    assert guess.shape == (3,)
    assert guess.dims == ("y",)
    assert isinstance(guess[0].item(), lf.Parameters)

    result = data.fit(model=model, params=guess)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (3,)
    assert result.dims == ("y",)
    assert isinstance(result[0].item(), lf.model.ModelResult)
    with open("fit_result.dill", "wb") as f:
        dill.dump(result, f)
