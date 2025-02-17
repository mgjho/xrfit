import numpy as np
from lmfit.models import LorentzianModel

import xrfit


def get_data_vars():
    rng = np.random.default_rng(seed=0)
    x_real = np.linspace(-1, 1, 30)
    x_imag = np.linspace(-1, 1, 30)
    x = np.meshgrid(x_real, x_imag)
    x = x[0] + 1j * x[1]
    x = x.flatten()
    model = LorentzianModel()

    params = model.make_params(center=0.1, amplitude=1, sigma=0.1)
    data = model.eval(params, x=x)
    data += 1.0 * rng.standard_normal(data.shape)

    return data, x, model, params


def test_modelresult_wrapper():
    data, x, model, params = get_data_vars()
    modelresult = model.fit(data, params, x=x)
    xrfit.ModelResultWrapper(modelresult).display()
