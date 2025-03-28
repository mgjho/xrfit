from typing import Literal

import lmfit as lf
import numpy as np
import xarray as xr

from xrfit.base import DataArrayAccessor
from xrfit.params import _set_bounds


def fit_with_iter_bound(
    result,
    init_params,
    iter_max,
    iter_crit,
    iter_tol,
    bound_ratio,
    bound_ratio_inc,
    bound_tol,
    max_bound_ratio,
    log,
    index_dict,
    **kws,
):
    init_val = getattr(result, iter_crit)

    for i in range(iter_max):
        ratio = min(bound_ratio + bound_ratio_inc * i, max_bound_ratio)
        result = _set_bounds(result, bound_ratio=ratio, bound_tol=bound_tol)
        result.fit(params=init_params, **kws)

        crit_val = getattr(result, iter_crit)
        crit_delta = abs((crit_val - init_val) / crit_val)

        if crit_delta < iter_tol:
            log(f"⚡️ tol reached at {i=} for {index_dict=}, delta={crit_delta:.3g}")
            break

        if i == iter_max - 1:
            log(f"⚠️ max iter reached at {index_dict=}, final delta={crit_delta:.3g}")

    return result


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
        set_bound: bool = False,
        bound_ratio: float = 0.05,
        bound_ratio_inc: float = 0.05,
        bound_tol: float = 1e-2,
        iter_max: int = 20,
        iter_crit: Literal["rsquared", "chisqr", "redchi"] = "rsquared",
        iter_tol: float = 0.01,
        max_bound_ratio: float = 0.3,
        verbose: bool = True,
        **kws,
    ) -> xr.DataArray:
        """Correlated fit with early exit, efficient bounds, and directional propagation."""
        import time

        def log(*args):
            if verbose:
                print(*args)

        fit_results = self.__call__(
            model=model,
            params=params,
            input_core_dims=input_core_dims,
            **kws,
        )

        dims = fit_results.dims
        shape = tuple(fit_results.sizes[d] for d in dims)

        if not isinstance(start_dict, dict):
            start_dict = (
                fit_results.assess.best_fit_stat()
                if start_dict == "stat"
                else fit_results.assess.best_fit_max()
            )
            log("⚡️ Start index estimated:", start_dict)

        start_tuple = tuple(start_dict[dim] for dim in dims)
        start_idx = np.ravel_multi_index(start_tuple, shape)
        total_idx = np.prod(shape)

        bound_ratios = np.clip(
            bound_ratio + bound_ratio_inc * np.arange(iter_max), None, max_bound_ratio
        )

        def process_voxel(flat_idx, initial_params):
            coord = np.unravel_index(flat_idx, shape)
            index_dict = dict(zip(dims, coord, strict=True))
            result = fit_results.isel(index_dict).item()

            t0 = time.time()

            if not set_bound:
                result.fit(params=initial_params, **kws)
                fit_results[index_dict] = result
                log(f"✅ Fit (no bound) at {index_dict} in {time.time() - t0:.2f}s")
                return result.params

            # Initial trial fit
            result.fit(params=initial_params, **kws)
            fit_results[index_dict] = result
            init_val = getattr(result, iter_crit)
            if iter_crit == "rsquared" and init_val > 0.99:
                log(f"✅ Early exit at {index_dict} — R² > 0.99")
                return result.params

            prev_delta = float("inf")
            for i, ratio in enumerate(bound_ratios):
                result = _set_bounds(result, bound_ratio=ratio, bound_tol=bound_tol)
                result.fit(params=initial_params, **kws)
                fit_results[index_dict] = result

                val = getattr(result, iter_crit)
                delta = abs((val - init_val) / val)

                if delta < iter_tol:
                    log(f"⚡️ Tol hit at {i=} for {index_dict}, Δ={delta:.3g}")
                    break
                if i > 1 and delta > prev_delta:
                    log(f"🛑 No gain at {i=} for {index_dict}, breaking.")
                    break
                prev_delta = delta

            log(f"⏱️ Fit at {index_dict} in {time.time() - t0:.2f}s")
            return result.params

        def propagate(index_iter):
            prev_params = fit_results.params.parse().isel(start_dict).item()
            for idx in index_iter:
                prev_params = process_voxel(idx, prev_params)

        propagate(range(start_idx - 1, -1, -1))  # Backward
        propagate(range(start_idx + 1, total_idx))  # Forward

        return fit_results
