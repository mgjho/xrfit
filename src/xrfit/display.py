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

        # Add the checkbox for fixing/unfixing the y-axis limits
        self.fix_ylim_checkbox = QtWidgets.QCheckBox("Fix Y-Axis Limits")
        self.fix_ylim_checkbox.toggled.connect(self.toggle_ylim)
        layout.addWidget(self.fix_ylim_checkbox)

        self.sliders = []
        self.slider_values = []

        for dim in range(self._obj.ndim):
            label = QtWidgets.QLabel(f"Dimension: {self._obj.dims[dim]}")
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(self._obj.shape[dim] - 1)
            slider.valueChanged.connect(self.update_plot)
            self.sliders.append(slider)
            self.slider_values.append(0)
            layout.addWidget(label)
            layout.addWidget(slider)

    def toggle_ylim(self, checked):
        if checked:
            # When checked, capture the current y-axis limits and disable auto-ranging.
            y_range = self.plot.viewRange()[1]  # [ymin, ymax]
            self.plot.enableAutoRange("y", False)
            self.plot.setYRange(y_range[0], y_range[1])
        else:
            # When unchecked, re-enable auto-ranging.
            self.plot.enableAutoRange("y", True)

    def update_plot(self, value):
        sender = self.sender()
        for i, slider in enumerate(self.sliders):
            if slider == sender:
                self.slider_values[i] = value

        index = tuple(self.slider_values)

        self.curve.setData(self._obj[index].item().best_fit)
        self.init_curve.setData(self._obj[index].item().init_fit)
        self.data_curve.setData(self._obj[index].item().data)


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
