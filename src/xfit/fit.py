import lmfit as lf
import xarray as xr

from xfit.base import DataArrayAccessor


@xr.register_dataarray_accessor("fit")
class FitAccessor(DataArrayAccessor):
    def guess(
        self,
        model: lf.model.Model,
        input_core_dims: str = "x",
    ) -> xr.DataArray:
        """
        Generate initial guess for the model parameters.

        model : lf.model.Model
            The model for which to generate the initial guess.
        input_core_dims : str, optional
            The dimension name in the xarray object to be used as input for the model's guess function. Default is "x".

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the initial guess for the model parameters.

        Notes
        -----
        This method uses `xr.apply_ufunc` to apply the model's guess function to the data
        """
        return xr.apply_ufunc(
            model.guess,
            self._obj,
            input_core_dims=[[input_core_dims]],
            kwargs={
                "x": getattr(self._obj, input_core_dims).values,
            },
            vectorize=True,
            dask="parallelized",
            output_dtypes=[object],
        )

    def _update(
        self,
        params: xr.DataArray,
        params_new: xr.DataArray,
    ) -> xr.DataArray:
        """
        Update the parameters with new values.

        This method takes two xarray DataArray objects, `params` and `params_new`,
        and updates the values in `params` with the corresponding values from
        `params_new`.

        Parameters
        ----------
        params : xr.DataArray
            The original parameters to be updated.
        params_new : xr.DataArray
            The new parameters to update the original parameters with.

        Returns
        -------
        xr.DataArray
            The updated parameters as an xarray DataArray.
        """
        return xr.apply_ufunc(
            lambda x, y: x.update(y),
            params,
            params_new,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[object],
        )

    def __call__(
        self,
        model: lf.model.Model,
        params: xr.DataArray | None = None,
        input_core_dims: str = "x",
        **kws,
    ) -> lf.model.ModelResult:
        """
        Call method to fit a model to the data.

        Parameters
        ----------
        model : lf.model.Model
            The model to be fitted.
        params : xr.DataArray or None, optional
            The parameters for the model. If None, parameters will be guessed.
        input_core_dims : str, optional
            The dimension name for the input data, by default "x".

        Returns
        -------
        xr.DataArray
            The result of the model fitting.

        """
        guesses = self.guess(model, input_core_dims)
        guesses = self._update(guesses, params) if params is not None else guesses

        args = [kws.pop(name) for name in ["weights"] if name in kws]
        input_core_dims_new = [
            [input_core_dims],
            [],
            *[[input_core_dims] for _ in args],
        ]
        return xr.apply_ufunc(
            model.fit,
            self._obj,
            guesses,
            *args,
            input_core_dims=input_core_dims_new,
            kwargs={
                "x": getattr(self._obj, input_core_dims).values,
                **kws,
            },
            vectorize=True,
            dask="parallelized",
        )
