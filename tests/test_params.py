import dill
import lmfit as lf
import xarray as xr


def test_params():
    with open("fit_result.dill", "rb") as f:
        result = dill.load(f)
    params = result.params()
    assert isinstance(params, xr.DataArray)
    assert isinstance(params[0].item(), lf.Parameters)
    sorted_result = result.params.sort("center")
    assert isinstance(sorted_result, xr.DataArray)
    assert isinstance(sorted_result[0].item(), lf.model.ModelResult)
    smoothend_result = result.params.smoothen("center", 5)
    assert isinstance(smoothend_result, xr.DataArray)
    assert isinstance(smoothend_result[0].item(), lf.model.ModelResult)
