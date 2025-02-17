import numpy as np
import pytest
from lmfit.models import LorentzianModel
from qtpy import QtWidgets

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


@pytest.fixture
def app(qtbot):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_modelresult_wrapper(qtbot, app):
    data, x, model, params = get_data_vars()
    modelresult = model.fit(data, params, x=x)

    # Create and show the window
    window = xrfit.ModelResultWrapper(modelresult)

    # Add window to qtbot, so it's managed during the test
    qtbot.addWidget(window)
    window.gen_plot_fit()
    window.show()

    # Wait for the window to be exposed
    qtbot.waitExposed(window)

    # Close the window automatically after a short wait using QTimer
    qtbot.wait(2000)  # Wait a bit longer to ensure the window closes
