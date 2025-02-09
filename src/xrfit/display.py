import sys

import pyqtgraph as pg
import xarray as xr

# from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from qtpy import QtCore, QtWidgets

from xrfit.base import DataArrayAccessor

# os.environ["QT_API"] = "pyqt6"
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class MainWindow(QtWidgets.QWidget):
    def __init__(self, xarr) -> None:
        super().__init__()
        self._obj = xarr
        self.fit_stat = "rsquared"
        self.goodness_threshold_lower = 0.8
        self.goodness_threshold_upper = 1.5
        self.tolerance = 1e-4
        self.setWindowTitle("Display Manager")

        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)

        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout)

        self.win = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.win)
        self.plot = self.win.addPlot(title="Fitting Result")
        initial_index = tuple([0] * (self._obj.ndim))
        x = self._obj[initial_index].item().userkws["x"]
        self.x_range = (x.min(), x.max())  # Store the x-axis range
        self.data_curve = self.plot.plot(
            x=x,
            y=self._obj[initial_index].item().data,
            symbol="o",
            pen=None,
            symbolBrush="k",
        )
        self.init_curve = self.plot.plot(
            x=x,
            y=self._obj[initial_index].item().init_fit,
            pen=pg.mkPen("b", width=4),
        )
        self.curve = self.plot.plot(
            x=x,
            y=self._obj[initial_index].item().best_fit,
            pen=pg.mkPen("r", width=4),
        )

        self.component_curves = []
        colors = [
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "cyan",
            "magenta",
        ]
        for i, component_name in enumerate(
            self._obj[initial_index].item().eval_components()
        ):
            component_curve = self.plot.plot(
                x=x,
                y=self._obj[initial_index].item().eval_components()[component_name],
                pen=pg.mkPen(colors[i], width=2, style=QtCore.Qt.PenStyle.DashLine),
                name=component_name,
            )
            self.component_curves.append(component_curve)

        self.fix_ylim_checkbox = QtWidgets.QCheckBox("Fix Y-Axis Limits")
        self.fix_ylim_checkbox.toggled.connect(self.toggle_ylim)
        left_layout.addWidget(self.fix_ylim_checkbox)

        self.sliders = []
        self.slider_values = []
        self.slider_labels = []

        for dim in range(self._obj.ndim):
            slider_label = QtWidgets.QLabel(f"{self._obj.dims[dim]}: 0")
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(self._obj.shape[dim] - 1)
            slider.valueChanged.connect(self.update_plot)
            self.sliders.append(slider)
            self.slider_values.append(0)
            self.slider_labels.append(slider_label)
            left_layout.addWidget(slider_label)
            left_layout.addWidget(slider)

        # Add dropdown for fit_stat
        self.fit_stat_dropdown = QtWidgets.QComboBox()
        self.fit_stat_dropdown.addItems(
            [
                "aic",
                "bic",
                "chisqr",
                "ci_out",
                "redchi",
                "rsquared",
                "success",
                "aborted",
                "ndata",
                "nfev",
                "nfree",
                "nvarys",
                "ier",
                "message",
            ]
        )
        self.fit_stat_dropdown.setCurrentText(self.fit_stat)
        self.fit_stat_dropdown.currentTextChanged.connect(self.update_fit_stat_label)
        left_layout.addWidget(QtWidgets.QLabel("Fit Statistic:"))
        left_layout.addWidget(self.fit_stat_dropdown)

        # Add a label to display the current fit_stat value
        self.fit_stat_label = QtWidgets.QLabel("Current Fit Stat: N/A")
        left_layout.addWidget(self.fit_stat_label)

        # Add a button to apply the input values
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_inputs)
        left_layout.addWidget(self.apply_button)
        # Add input fields for goodness_threshold_lower and goodness_threshold_upper
        self.goodness_threshold_lower_input = QtWidgets.QLineEdit(
            str(self.goodness_threshold_lower)
        )
        self.goodness_threshold_upper_input = QtWidgets.QLineEdit(
            str(self.goodness_threshold_upper)
        )
        left_layout.addWidget(QtWidgets.QLabel("Goodness Threshold Lower:"))
        left_layout.addWidget(self.goodness_threshold_lower_input)
        left_layout.addWidget(QtWidgets.QLabel("Goodness Threshold Upper:"))
        left_layout.addWidget(self.goodness_threshold_upper_input)

        # Add parameter values at the right of the main window
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_layout)

        self.param_labels = []
        for param_name, param in self._obj[initial_index].item().params.items():
            color = (
                "green"
                if param.min + self.tolerance < param.value < param.max - self.tolerance
                else "red"
            )
            param_label = QtWidgets.QLabel(
                f"<b style='color:{color}'>{param_name}</b><br>Value: {param.value:.3f}<br>Min: {param.min:.3f}<br>Max: {param.max:.3f}<br>"
            )
            self.param_labels.append(param_label)
            right_layout.addWidget(param_label)

        # Add a scroll area for the parameter values
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        for label in self.param_labels:
            scroll_layout.addWidget(label)
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)

        # Add a label to indicate if all parameters are within bounds
        self.param_status_label = QtWidgets.QLabel("All Parameters Within Bounds")
        right_layout.addWidget(self.param_status_label)
        self.update_param_status_label()

    def toggle_ylim(self, checked):
        if checked:
            y_range = self.plot.viewRange()[1]
            self.plot.enableAutoRange("y", False)
            self.plot.setYRange(y_range[0], y_range[1])
        else:
            self.plot.enableAutoRange("y", True)

    def update_plot(self, value):
        sender = self.sender()
        for i, slider in enumerate(self.sliders):
            if slider == sender:
                self.slider_values[i] = value
                self.slider_labels[i].setText(f"{self._obj.dims[i]}: {value}")

        index = tuple(self.slider_values)

        x = self._obj[index].item().userkws["x"]  # Update x data
        self.curve.setData(x, self._obj[index].item().best_fit)
        self.init_curve.setData(x, self._obj[index].item().init_fit)
        self.data_curve.setData(x, self._obj[index].item().data)

        # Update individual component plots
        components = self._obj[index].item().eval_components()
        for component_curve, component_name in zip(
            self.component_curves, components.keys(), strict=False
        ):
            component_curve.setData(x, components[component_name])

        # Ensure the x-axis range remains visible
        # self.plot.setXRange(*self.x_range)

        # Update parameter labels
        for param_label, (param_name, param) in zip(
            self.param_labels, self._obj[index].item().params.items(), strict=False
        ):
            color = (
                "green"
                if param.min + self.tolerance < param.value < param.max - self.tolerance
                else "red"
            )
            param_label.setText(
                f"<b style='color:{color}'>{param_name}</b><br>Value: {param.value:.3f}<br>Min: {param.min:.3f}<br>Max: {param.max:.3f}<br><br>"
            )

        self.update_slider_label_color(index)
        self.update_fit_stat_label(index)
        self.update_param_status_label()

    def update_slider_label_color(self, index):
        fit_stat = self._obj.assess.fit_stats(self.fit_stat)
        goodness_of_fit = fit_stat[index].item()
        for _, label in enumerate(self.slider_labels):
            if isinstance(goodness_of_fit, float):
                if (
                    self.goodness_threshold_lower
                    <= goodness_of_fit
                    <= self.goodness_threshold_upper
                ):
                    label.setStyleSheet("color: green;")
                else:
                    label.setStyleSheet("color: red;")

    def update_fit_stat_label(self, index=None):
        self.fit_stat = self.fit_stat_dropdown.currentText()
        if index is None:
            index = tuple(self.slider_values)
        fit_stat = self._obj.assess.fit_stats(self.fit_stat)
        try:
            current_fit_stat = fit_stat[index].item()
            self.fit_stat_label.setText(f"Current Fit Stat: {current_fit_stat}")
        except KeyError:
            self.fit_stat_label.setText("Current Fit Stat: N/A")

    def update_param_status_label(self):
        all_within_bounds = all("green" in label.text() for label in self.param_labels)
        if all_within_bounds:
            self.param_status_label.setText("All Parameters Within Bounds ✅")
            self.param_status_label.setStyleSheet("color: green;")
        else:
            self.param_status_label.setText("Some Parameters Out of Bounds ❌")
            self.param_status_label.setStyleSheet("color: red;")

    def apply_inputs(self):
        self.fit_stat = self.fit_stat_dropdown.currentText()
        self.goodness_threshold_lower = float(
            self.goodness_threshold_lower_input.text()
        )
        self.goodness_threshold_upper = float(
            self.goodness_threshold_upper_input.text()
        )
        self.update_slider_label_color(tuple(self.slider_values))
        self.update_fit_stat_label(tuple(self.slider_values))
        self.update_param_status_label()


@xr.register_dataarray_accessor("display")
class DisplayAccessor(DataArrayAccessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def __call__(self):
        if not QtWidgets.QApplication.instance():
            qapp = QtWidgets.QApplication(sys.argv)
        else:
            qapp = QtWidgets.QApplication.instance()
        qapp.setStyle("Fusion")
        win = MainWindow(xarr=self._obj)
        win.show()
        win.activateWindow()
        qapp.exec()
