from typing import Literal

import lmfit as lf
import numpy as np
import numpy.typing as npt
import xarray as xr

from xrfit.base import DataArrayAccessor


def _get_arr(
    model_result: lf.model.ModelResult,
    attr_name: Literal["best_fit", "init_fit", "residual", "data"] = "best_fit",
) -> npt.NDArray[np.float64]:
    # since parameter value might have changed, we re-evaluate the model just in case
    best_fit = model_result.eval()
    data = model_result.data
    if attr_name == "best_fit":
        return best_fit
    if attr_name == "init_fit":
        return model_result.init_fit
    if attr_name == "residual":
        return data.data - best_fit
    if attr_name == "data":
        return data
    raise ValueError(
        f"Invalid attr_name: {attr_name} (must be one of 'best_fit', 'init_fit', 'residual', 'data')"
    )


@xr.register_dataarray_accessor("get_arr")
class ArrAccessor(DataArrayAccessor):
    def _get_x(self):
        return self._obj[0].item().userkws["x"]

    def __call__(
        self,
        attr_name: Literal["best_fit", "init_fit", "residual", "data"] = "best_fit",
        new_dim_name: str = "x",
    ) -> xr.DataArray:
        return xr.apply_ufunc(
            _get_arr,
            self._obj,
            output_core_dims=[[new_dim_name]],
            kwargs={"attr_name": attr_name},
            vectorize=True,
            dask="parallelized",
        ).assign_coords(coords={new_dim_name: self._get_x()})
