import sys

import pyqtgraph as pg
import xarray as xr
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
        self.setWindowTitle("Display Manager")

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        self.plot = self.win.addPlot(title="Fitting Result")
        initial_index = tuple([0] * (self._obj.ndim))
        x = self._obj[initial_index].item().userkws["x"]
        self.init_curve = self.plot.plot(
            x=x,
            y=self._obj[initial_index].item().init_fit,
            pen=pg.mkPen("b", width=3),
        )
        self.curve = self.plot.plot(
            x=x,
            y=self._obj[initial_index].item().best_fit,
            pen=pg.mkPen("r", width=3),
        )
        self.data_curve = self.plot.plot(
            x=x,
            y=self._obj[initial_index].item().data,
            symbol="o",
            pen=None,
            symbolBrush="k",
        )
        self.fix_ylim_checkbox = QtWidgets.QCheckBox("Fix Y-Axis Limits")
        self.fix_ylim_checkbox.toggled.connect(self.toggle_ylim)
        layout.addWidget(self.fix_ylim_checkbox)

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
            layout.addWidget(slider_label)
            layout.addWidget(slider)

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
        layout.addWidget(QtWidgets.QLabel("Fit Statistic:"))
        layout.addWidget(self.fit_stat_dropdown)

        # Add a label to display the current fit_stat value
        self.fit_stat_label = QtWidgets.QLabel("Current Fit Stat: N/A")
        layout.addWidget(self.fit_stat_label)

        # Add a button to apply the input values
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_inputs)
        layout.addWidget(self.apply_button)
        # Add input fields for goodness_threshold_lower and goodness_threshold_upper
        self.goodness_threshold_lower_input = QtWidgets.QLineEdit(
            str(self.goodness_threshold_lower)
        )
        self.goodness_threshold_upper_input = QtWidgets.QLineEdit(
            str(self.goodness_threshold_upper)
        )
        layout.addWidget(QtWidgets.QLabel("Goodness Threshold Lower:"))
        layout.addWidget(self.goodness_threshold_lower_input)
        layout.addWidget(QtWidgets.QLabel("Goodness Threshold Upper:"))
        layout.addWidget(self.goodness_threshold_upper_input)

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

        self.curve.setData(self._obj[index].item().best_fit)
        self.init_curve.setData(self._obj[index].item().init_fit)
        self.data_curve.setData(self._obj[index].item().data)

        self.update_slider_label_color(index)
        self.update_fit_stat_label(index)

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
