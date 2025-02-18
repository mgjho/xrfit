import numpy as np
import xarray as xr

from xrfit.base import DataArrayAccessor


@xr.register_dataarray_accessor("bin")
class BinAccessor(DataArrayAccessor):
    def __call__(self, **dim_multipliers):
        dim_dict = {}

        # Loop over the passed dimensions and apply the interpolation multiplier
        for dim, multiplier in dim_multipliers.items():
            if dim not in self._obj.dims:
                raise ValueError(f"Dimension '{dim}' not found in the DataArray.")

            # Get the current coordinate values for this dimension
            current_coord = self._obj[dim].values

            # Create the new coordinate using np.linspace
            new_dim = np.linspace(
                current_coord[0],
                current_coord[-1],
                int(
                    self._obj.sizes[dim] / multiplier
                ),  # Adjust the size based on multiplier
            )
            dim_dict[dim] = new_dim

        # Perform the interpolation with the constructed dim_dict
        return self._obj.interp(**dim_dict)
