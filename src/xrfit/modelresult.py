# Wrapper for lmfit.model.modelresult
import sys

from qtpy import QtWidgets


class MainWindow(QtWidgets.QWidget):
    def __init__(self, xarr) -> None:
        super().__init__()
        self._obj = xarr


class ModelResultWrapper:
    def __init__(self, modelresult):
        self.modelresult = modelresult

    def display(self):
        if not QtWidgets.QApplication.instance():
            qapp = QtWidgets.QApplication(sys.argv)
        else:
            qapp = QtWidgets.QApplication.instance()
        qapp.setStyle("Fusion")
        win = MainWindow(xarr=self.modelresult)
        win.show()
        win.activateWindow()
        qapp.exec()
