import xarray as xr


class DataArrayAccessor:
    def __init__(self, xarr: xr.DataArray) -> None:
        self._obj = xarr
