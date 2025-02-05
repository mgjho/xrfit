import lmfit as lf
import numpy as np
import xarray as xr

from xrfit.base import DataArrayAccessor


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

    # def _get_attrs(
    #     self,
    #     fit_results: xr.DataArray,

    # ):
    #     return xr.apply_ufunc(
    #         lambda x: x.,
    #         fit_results,
    #         vectorize=True,
    #         dask="parallelized",
    #         output_dtypes=[object],
    #     )

    def fit_with_corr(
        self,
        model: lf.model.Model,
        input_core_dims: str = "x",
        start_dict: dict | None = None,
        **kws,
    ) -> xr.DataArray:
        """
        Fit the model starting from a certain index and use the resulting parameters for the next fit.

        Parameters
        ----------
        model : lf.model.Model
            The model to be fitted.
        start_index : tuple
            The starting index for the fit.
        input_core_dims : str, optional
            The dimension name for the input data, by default "x".

        Returns
        -------
        xr.DataArray
            The result of the model fitting with correlated parameters.
        """
        fit_results = self.__call__(model=model, input_core_dims=input_core_dims)
        dims = fit_results.dims
        if start_dict is None:
            start_dict = fit_results.assess.best_fit_stat()
            print("⚡️ No initial coords provided for fit_with_corr")
            print("⚡️ Estimate used :", start_dict)
        start_tuple = tuple(start_dict.values())
        dims_tuple = tuple(fit_results.sizes[dim] for dim in dims)
        start_idx = np.ravel_multi_index(start_tuple, dims_tuple)
        total_idx = np.prod(dims_tuple)
        previous_params = fit_results.params.__call__().isel(start_dict).item()

        for idx in range(start_idx - 1, -1, -1):
            indices = np.unravel_index(idx, dims_tuple)
            index_dict = dict(zip(dims, indices, strict=False))
            single_fit_result = fit_results.isel(index_dict).item()
            single_fit_result.fit(params=previous_params)
            fit_results[index_dict] = single_fit_result
            previous_params = single_fit_result.params

        previous_params = fit_results.params.__call__().isel(start_dict).item()

        for idx in range(start_idx + 1, total_idx):
            indices = np.unravel_index(idx, dims_tuple)
            index_dict = dict(zip(dims, indices, strict=False))
            single_fit_result = fit_results.isel(index_dict).item()
            single_fit_result.fit(params=previous_params)
            fit_results[index_dict] = single_fit_result
            previous_params = single_fit_result.params

        return fit_results
