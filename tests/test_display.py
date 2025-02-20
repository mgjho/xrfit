import lmfit as lf
import numpy as np
import pytest
import xarray as xr
from qtpy import QtWidgets


@pytest.fixture
def sample_data():
    rng = np.random.default_rng()
    x = np.linspace(0, 10, 100)
    y = 3 * np.sin(x) + rng.normal(size=x.size)
    z = np.linspace(0, 1, 2)

    model = lf.models.SineModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)

    return xr.DataArray([result, result], coords={"z": z}, dims=["z"])


@pytest.fixture
def app(qtbot):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# def test_display_accessor(qtbot, app, sample_data):
# Create the MainWindow instance using the display accessor
# main_window = sample_data.display(return_window=True)

# # # Ensure main_window is a QWidget
# # assert isinstance(main_window, QtWidgets.QWidget)

# # # Add window to qtbot, so it's managed during the test
# qtbot.addWidget(main_window)

# # Wait for the window to be exposed
# qtbot.waitExposed(main_window)

# # Check initial state
# assert main_window.fit_stat == "rsquared"
# assert main_window.goodness_threshold_lower == 0.8
# assert main_window.goodness_threshold_upper == 1.5

# # Simulate user interaction
# main_window.sliders[0].setValue(1)
# qtbot.wait(100)

# # Check updated state
# assert main_window.slider_values[0] == 1
# assert main_window.slider_labels[0].text() == "z: 1"

# # Check plot data
# index = (1,)
# np.testing.assert_allclose(
#     main_window.curve.getData()[1], sample_data[index].item().best_fit
# )
# np.testing.assert_allclose(
#     main_window.init_curve.getData()[1], sample_data[index].item().init_fit
# )
# np.testing.assert_allclose(
#     main_window.data_curve.getData()[1], sample_data[index].item().data
# )

# # Check parameter labels
# for param_label, (_, param) in zip(
#     main_window.param_labels,
#     sample_data[index].item().params.items(),
#     strict=True,
# ):
#     color = (
#         "green"
#         if param.min + main_window.tolerance
#         < param.value
#         < param.max - main_window.tolerance
#         else "red"
#     )
#     assert f"color:{color}" in param_label.text()

# # Check fit_stat_label
# main_window.update_fit_stat_label(index)
# fit_stat = sample_data.assess.fit_stats(main_window.fit_stat)
# current_fit_stat = fit_stat[index].item()
# assert main_window.fit_stat_label.text() == f"Current Fit Stat: {current_fit_stat}"

# # Check parameter status label
# main_window.update_param_status_label()
# all_within_bounds = all(
#     "green" in label.text() for label in main_window.param_labels
# )
# if all_within_bounds:
#     assert (
#         main_window.param_status_label.text() == "All Parameters Within Bounds ✅"
#     )
#     assert "color: green;" in main_window.param_status_label.styleSheet()
# else:
#     assert (
#         main_window.param_status_label.text() == "Some Parameters Out of Bounds ❌"
#     )
#     assert "color: red;" in main_window.param_status_label.styleSheet()


# # Close the MainWindow instance
# # qtbot.waitSignal(main_window.destroyed, timeout=1000)
# # main_window.close()
# # app.quit()
