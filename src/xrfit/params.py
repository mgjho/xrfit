import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from xrfit.base import DataArrayAccessor


@xr.register_dataarray_accessor("params")
class ParamsAccessor(DataArrayAccessor):
    def __call__(
        self,
    ) -> xr.DataArray:
        return xr.apply_ufunc(
            lambda x: x.params,
            self._obj,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[object],
        )

    def smoothen(
        self,
        param_name: str = "center",
        sigma: int = 5,
        # params_name: list | None = None,
    ) -> xr.DataArray:
        # if params_name is None:
        # params_name = ["center"]
        param = self._obj.params.get(param_name)
        smoothing_sigma = [
            sigma if i < param.ndim - 1 else 0 for i in range(param.ndim)
        ]

        param_smooth = gaussian_filter(param, sigma=smoothing_sigma)
        self._obj.params.set(param_smooth, param_name)
        return self._obj

    def sort(
        self,
        target_param_name: str = "center",
        params_name: list | None = None,
    ) -> xr.DataArray:
        if params_name is None:
            params_name = ["center"]
        param_to_sortby = self._obj.params.get(target_param_name)
        sorted_indices = param_to_sortby.argsort(axis=-1)
        for param_name in params_name:
            param = self._obj.params.get(param_name)
            sorted_param = param.isel(params_dim=sorted_indices)
            self._obj.params.set(sorted_param, param_name)
        return self._obj

    def get(
        self,
        params_name: str = "center",
        params_attr: str = "value",
    ) -> xr.DataArray:
        return xr.apply_ufunc(
            self._get,
            self._obj,
            kwargs={
                "params_name": params_name,
                "params_attr": params_attr,
            },
            input_core_dims=[[]],
            output_core_dims=[["params_dim"]],
            vectorize=True,
        )

    def _get(
        self,
        data: xr.DataArray,
        params_name: str = "center",
        params_attr: str = "value",
    ):
        params = data.params
        return np.array(
            [
                getattr(params[key], params_attr)
                for key in params
                if key.endswith(params_name)
            ]
        )

    def set(
        self,
        params_value_new: xr.DataArray,
        params_name: str = "center",
        # params_attr: str = "value",
    ) -> xr.DataArray:
        return xr.apply_ufunc(
            self._set,
            self._obj,
            params_value_new,
            kwargs={
                "params_name": params_name,
                # "params_attr": params_attr,
            },
            input_core_dims=[[], ["params_dim"]],
            vectorize=True,
        )

    # TODO - Currently only sets values
    def _set(
        self,
        data: xr.DataArray,
        params_value_new: xr.DataArray,
        params_name: str = "center",
        # params_attr: str = "value",
    ):
        params = data.params
        pars = [key for key in params if key.endswith(params_name)]
        for i, par in enumerate(pars):
            data.params[par].set(value=params_value_new[i])
        return data
