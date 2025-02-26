import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel

import xrfit


def get_test_data_2d():
    rng = np.random.default_rng(seed=0)
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 7)
    model = LorentzianModel()
    return xr.DataArray(
        np.stack(
            [
                model.eval(x=x, amplitude=1, center=-5, sigma=0.05 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.1 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.15 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.2 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.25 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.3 * 5),
                model.eval(x=x, amplitude=1, center=-5, sigma=0.35 * 5),
            ]
        ).T
        + rng.normal(size=(x.size, y.size)) * 0.1,
        coords={"x": x, "y": y},
        dims=("x", "y"),
    )


def test_bound():
    data = get_test_data_2d()
    model = LorentzianModel(prefix="p0_")
    fit_result = data.fit.fit_with_corr(model=model, bound_ratio=1e-1)
