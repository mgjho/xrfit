import sys
from typing import cast

import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtWidgets

from xrfit.base import DataArrayAccessor


@xr.register_dataarray_accessor("display")
class DisplayAccessor(DataArrayAccessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        self.qapp = cast(
            QtWidgets.QApplication | None, QtWidgets.QApplication.instance()
        )
        if not self.qapp:
            self.qapp = QtWidgets.QApplication(sys.argv)
            self._own_qapp = True
        else:
            self._own_qapp = False
        self.qapp.setApplicationDisplayName("Display Manager")

    def __call__(self):
        self.win = pg.GraphicsLayoutWidget(show=True, title="Display Manager")
        self.plot = self.win.addPlot(title="Fitting Result")

        initial_index = tuple([0] * (self._obj.ndim))
        self.curve = self.plot.plot(
            self._obj[initial_index].item().best_fit,
            pen=pg.mkPen("r"),
        )
        self.data_curve = self.plot.plot(
            self._obj[initial_index].item().data,
            symbol="o",
            pen=None,
        )
        self.sliders = []
        self.slider_values = []

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.win)

        for dim in range(self._obj.ndim):
            label = QtWidgets.QLabel(f"Dimension: {self._obj.dims[dim]}")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(self._obj.shape[dim] - 1)
            slider.valueChanged.connect(self.update_plot)
            self.sliders.append(slider)
            self.slider_values.append(0)
            layout.addWidget(label)
            layout.addWidget(slider)

        self.main_widget = QtWidgets.QWidget()
        self.main_widget.setLayout(layout)
        self.main_widget.show()

        if self._own_qapp:
            self.qapp.exec_()
            self.main_widget.close()
            self.qapp.quit()

    def update_plot(self, value):
        sender = self.qapp.sender()
        for i, slider in enumerate(self.sliders):
            if slider == sender:
                self.slider_values[i] = value

        index = tuple(self.slider_values)
        self.curve.setData(self._obj[index].item().best_fit)
        self.data_curve.setData(self._obj[index].item().data)
