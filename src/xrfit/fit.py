from typing import Literal

import lmfit as lf
import numpy as np
import xarray as xr

from xrfit.base import DataArrayAccessor

# from xrfit.params import _set_bounds


def _generalized_guess(model, data, x):
    """Recursively generates initial parameter guesses for lmfit models, including composite and convolved models."""
    params = lf.Parameters()

    # Handle CompositeModel
    if isinstance(model, lf.model.CompositeModel):
        if model.left is not None:
            params.update(
                _generalized_guess(model.left, data, x)
            )  # Recursively call on left component
        if model.right is not None:
            params.update(
                _generalized_guess(model.right, data, x)
            )  # Recursively call on right component
        return params

    # Handle ConvolutionModel
    if hasattr(model, "model") and hasattr(model, "op"):
        params.update(
            _generalized_guess(model.model, data, x)
        )  # Recursively call on convolved model
        return params

    # Handle models with a `guess` method
    if hasattr(model, "guess"):
        return model.guess(data, x=x)  # Call the guess method of the model

    raise ValueError(f"Model {model} does not support guess().")


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
            lambda data, x: _generalized_guess(model, data, x),
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

    def fit_with_corr(
        self,
        model: lf.model.Model,
        params: xr.DataArray | None = None,
        input_core_dims: str = "x",
        start_dict: dict | Literal["stat", "max"] = "max",
        bound_ratio: float = 0.05,
        bound_ratio_inc: float = 0.05,
        bound_tol: float = 1e-3,
        iter_max: int = 100,
        iter_crit: Literal["rsquared", "chisqr", "redchi"] = "rsquared",
        iter_tol: float = 0.99,
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
        fit_results = self.__call__(
            model=model,
            params=params,
            input_core_dims=input_core_dims,
            **kws,
        )
        dims = fit_results.dims
        if not isinstance(start_dict, dict):
            if start_dict == "stat":
                start_dict = fit_results.assess.best_fit_stat()
            elif start_dict == "max":
                start_dict = fit_results.assess.best_fit_max()
            else:
                raise ValueError("Invalid value for start_dict.")
            print("⚡️ No initial coords provided for fit_with_corr")
            print("⚡️ Estimate used :", start_dict)
        if isinstance(start_dict, dict):
            start_tuple = tuple(start_dict.values())
        else:
            raise TypeError("start_dict must be a dictionary.")
        dims_tuple = tuple(fit_results.sizes[dim] for dim in dims)
        start_idx = np.ravel_multi_index(start_tuple, dims_tuple)
        total_idx = np.prod(dims_tuple)
        # if bound_ratio is not None:
        # fit_results = fit_results.params.set_bounds(bound_ratio=bound_ratio)
        previous_params = fit_results.params.parse().isel(start_dict).item()

        for idx in range(start_idx, -1, -1):
            indices = np.unravel_index(idx, dims_tuple)
            index_dict = dict(zip(dims, indices, strict=False))
            single_fit_result = fit_results.isel(index_dict).item()
            for iter_idx in range(iter_max):
                single_fit_result.fit(params=previous_params, **kws)
                iter_crit_val = getattr(single_fit_result, iter_crit)
                # single_fit_result = _set_bounds(
                #     single_fit_result,
                #     bound_ratio=bound_ratio + bound_ratio_inc * iter_idx,
                #     bound_tol=bound_tol,
                # )
                fit_results[index_dict] = single_fit_result
                fit_results.params.set_bounds(
                    bound_ratio=bound_ratio + bound_ratio_inc * iter_idx,
                    bound_tol=bound_tol,
                    index_dict=index_dict,
                )
                previous_params = fit_results.params.parse().isel(index_dict).item()
                if iter_crit_val > iter_tol:
                    print(
                        "⚡️ iter_bound tol reached at iter : ",
                        iter_idx,
                        "iter_crit : ",
                        iter_crit_val,
                        "iter_tol : ",
                        iter_tol,
                    )
                    break
                if iter_idx == iter_max - 1:
                    print(
                        "⚠️ iter_max reached at iter : ",
                        iter_idx,
                        "for idx : ",
                        index_dict,
                        "iter_crit : ",
                        iter_crit_val,
                        "iter_tol : ",
                        iter_tol,
                    )

        previous_params = fit_results.params.parse().isel(start_dict).item()
        for idx in range(start_idx + 1, total_idx):
            indices = np.unravel_index(idx, dims_tuple)
            index_dict = dict(zip(dims, indices, strict=False))
            single_fit_result = fit_results.isel(index_dict).item()
            for iter_idx in range(iter_max):
                single_fit_result.fit(params=previous_params, **kws)
                fit_results[index_dict] = single_fit_result
                fit_results.params.set_bounds(
                    bound_ratio=bound_ratio + bound_ratio_inc * iter_idx,
                    bound_tol=bound_tol,
                    index_dict=index_dict,
                )
                previous_params = fit_results.params.parse().isel(index_dict).item()
                if iter_crit_val > iter_tol:
                    print(
                        "⚡️ iter_bound tol reached at iter : ",
                        iter_idx,
                        "iter_crit : ",
                        iter_crit_val,
                        "iter_tol : ",
                        iter_tol,
                    )
                    break
                if iter_idx == iter_max - 1:
                    print(
                        "⚠️ iter_max reached at iter : ",
                        iter_idx,
                        "for idx : ",
                        index_dict,
                        "iter_crit : ",
                        iter_crit_val,
                        "iter_tol : ",
                        iter_tol,
                    )
        return fit_results
