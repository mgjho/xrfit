import lmfit as lf
import numpy as np
from lmfit import CompositeModel
from lmfit.lineshapes import gaussian
from lmfit.models import LorentzianModel, StepModel
from scipy.signal import convolve

from xrfit.fit import _generalized_guess


class GaussianConvolveModel(lf.Model):
    """Model for Gaussian convolution."""

    def __init__(
        self,
        independent_vars: list = ["x"],
        prefix: str = "",
        missing: str = "drop",
        name: None = None,
        sigma: float | None = None,
        **kwargs,
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        self.sigma = sigma
        super().__init__(gaussian, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        pars[f"{self.prefix}center"].set(value=0.0, vary=False)
        if self.sigma is not None:
            pars[f"{self.prefix}sigma"].set(value=self.sigma, vary=False)
        else:
            pars[f"{self.prefix}sigma"].set(value=1.0, min=1e-4, max=100.0, vary=True)
        pars[f"{self.prefix}amplitude"].set(
            expr=f"1 / ({self.prefix}sigma * sqrt(2 * pi))"
        )

        return lf.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


def test_generalized_guess():
    x = np.linspace(-10, 10, 100)
    data = np.exp(-(x**2))  # Example Gaussian data

    # Multiple sum of Lorentzian models
    lorentz1 = LorentzianModel(prefix="l1_")
    lorentz2 = LorentzianModel(prefix="l2_")
    lorentz_sum = lorentz1 + lorentz2

    params = _generalized_guess(lorentz_sum, data, x)
    assert isinstance(params, lf.Parameters)
    assert "l1_center" in params
    assert "l2_center" in params

    # Multiplying by a step function
    step = StepModel(prefix="s_")
    step_mult_model = lorentz_sum * step

    params = _generalized_guess(step_mult_model, data, x)
    assert isinstance(params, lf.Parameters)
    assert "l1_center" in params
    assert "l2_center" in params
    assert "s_center" in params

    # Convolving sum of Lorentzian models with a Gaussian
    gauss = GaussianConvolveModel(prefix="g_")
    convolved_model = CompositeModel(step_mult_model, gauss, convolve)

    params = _generalized_guess(convolved_model, data, x)
    print("params", params)
    assert isinstance(params, lf.Parameters)
    assert "l1_center" in params
    assert "l2_center" in params
    assert "s_center" in params
    assert "g_center" in params

    # print("All tests passed.")
