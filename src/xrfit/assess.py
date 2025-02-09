from typing import Literal

import numpy as np
import xarray as xr

from xrfit.base import DataArrayAccessor


@xr.register_dataarray_accessor("assess")
class AccessAccessor(DataArrayAccessor):
    def fit_stats(
        self,
        attr_name: Literal[
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
        ] = "rsquared",
    ) -> xr.DataArray:
        def get_attr_safe(x, attr_name):
            try:
                return getattr(x, attr_name)
            except AttributeError:
                return np.nan

        return xr.apply_ufunc(
            get_attr_safe,
            self._obj,
            input_core_dims=[[]],
            kwargs={"attr_name": attr_name},
            vectorize=True,
            dask="parallelized",
        )

    def fit_max(
        self,
    ):
        def get_fit_safe(x):
            try:
                return x.best_fit.max()
            except AttributeError:
                return np.nan

        return xr.apply_ufunc(
            get_fit_safe,
            self._obj,
            input_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
        )

    def best_fit_max(
        self,
    ) -> dict:
        darr = self.fit_max()
        idx = np.unravel_index(darr.values.argmax(), darr.shape)
        return dict(zip(darr.dims, idx, strict=False))

    def best_fit_stat(
        self,
        attr_name: Literal[
            "aic",
            "bic",
            "chisqr",
            "redchi",
            "rsquared",
        ] = "rsquared",
    ) -> dict:
        darr = self.fit_stats(attr_name)
        if attr_name in ["aic", "bic", "chisqr", "redchi"]:
            idx = np.unravel_index(darr.values.argmin(), darr.shape)
        if attr_name in ["rsquared"]:
            idx = np.unravel_index(darr.values.argmax(), darr.shape)
        return dict(zip(darr.dims, idx, strict=False))
