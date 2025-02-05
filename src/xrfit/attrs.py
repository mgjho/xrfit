from typing import Literal

import xarray as xr

from xrfit.base import DataArrayAccessor


@xr.register_dataarray_accessor("get_attrs")
class AttrsAccessor(DataArrayAccessor):
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
        ],
    ) -> xr.DataArray:
        return xr.apply_ufunc(
            lambda x: getattr(x, attr_name),
            self._obj,
            vectorize=True,
            dask="parallelized",
            # output_dtypes=[object],
        )
