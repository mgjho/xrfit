import sys

import numpy as np
import pyqtgraph.opengl as gl
from qtpy import QtWidgets


class ModelResultWrapper(gl.GLViewWidget):
    def __init__(self, xarr) -> None:
        super().__init__()
        self._obj = xarr

    def gen_plot_fit(self):
        self.setWindowTitle("3D Display Manager")
        # self.setCameraPosition(distance=20)
        # Set Background Color
        self.setBackgroundColor((0, 0, 0, 1))  # Black background for contrast
        # Add Grid
        self.addItem(gl.GLGridItem())

        # Add Axis
        self.add_axes()

        # Add Scatter Points
        # rng = np.random.default_rng()
        # pos = rng.random(size=(100, 3))
        # pos *= [10, -10, 10]
        # sp = gl.GLScatterPlotItem(pos=pos, color=(1, 0, 0, 1), size=5.0)
        # self.addItem(sp)
        # self.update()
        x_cmplx = self._obj.userkws["x"]
        x = x_cmplx.real
        y = x_cmplx.imag

        z_list = [self._obj.best_fit, self._obj.data, self._obj.init_fit]
        color_list = [(0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)]
        for z, color in zip(z_list, color_list, strict=True):
            z = z.imag
            pos = np.column_stack((x, y, z))
            scatter_points = gl.GLScatterPlotItem(pos=pos, color=color, size=10.0)
            self.addItem(scatter_points)
            self.update()
        return self

    def add_axes(self):
        """Add X, Y, Z axes to the plot."""
        axis_length = 10  # Length of the axis lines

        # X-axis (Red)
        x_axis = np.array([[0, 0, 0], [axis_length, 0, 0]])  # From (0,0,0) to (10,0,0)
        x_line = gl.GLLinePlotItem(pos=x_axis, color=(1, 0, 0, 1), width=2)  # Red
        self.addItem(x_line)

        # Y-axis (Green)
        y_axis = np.array([[0, 0, 0], [0, axis_length, 0]])  # From (0,0,0) to (0,10,0)
        y_line = gl.GLLinePlotItem(pos=y_axis, color=(0, 1, 0, 1), width=2)  # Green
        self.addItem(y_line)

        # Z-axis (Blue)
        z_axis = np.array([[0, 0, 0], [0, 0, axis_length]])  # From (0,0,0) to (0,0,10)
        z_line = gl.GLLinePlotItem(pos=z_axis, color=(0, 0, 1, 1), width=2)  # Blue
        self.addItem(z_line)

    def display(self):
        if not QtWidgets.QApplication.instance():
            qapp = QtWidgets.QApplication(sys.argv)
        else:
            qapp = QtWidgets.QApplication.instance()
        qapp.setStyle("Fusion")
        win = self.gen_plot_fit()
        win.show()
        win.activateWindow()
        qapp.exec()
