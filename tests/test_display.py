import dill
from qtpy import QtWidgets


def test_display_3d(qtbot):
    with open("fit_result_3d.dill", "rb") as f:
        result_3d = dill.load(f)
    result_3d.display()
    qtbot.addWidget(result_3d.display.main_widget)
    assert isinstance(result_3d.display.main_widget, QtWidgets.QWidget)
    result_3d.display.main_widget.close()


def test_display_2d(qtbot):
    with open("fit_result.dill", "rb") as f:
        result_2d = dill.load(f)
    result_2d.display()
    qtbot.addWidget(result_2d.display.main_widget)
    assert isinstance(result_2d.display.main_widget, QtWidgets.QWidget)
    result_2d.display.main_widget.close()
