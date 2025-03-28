import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel

import xrfit

__all__ = [
    "test_bound",
    "xrfit",
]


def get_test_data_2d():
    # x, y 좌표 생성
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y, indexing="ij")
    rng = np.random.default_rng(42)
    model = LorentzianModel()

    # 피크의 파라볼릭 중심과 폭 설정
    center1 = -2 + 0.1 * y**2
    center2 = 2 + 0.3 * y**2
    sigma1 = 0.5 + 0.01 * y**2
    sigma2 = 0.5 + 0.01 * y**2
    amplitude1 = 1.0
    amplitude2 = 0.8

    # 데이터 생성
    data = np.zeros_like(X)
    for i in range(len(y)):
        peak1 = model.eval(
            x=x, amplitude=amplitude1, center=center1[i], sigma=sigma1[i]
        )
        peak2 = model.eval(
            x=x, amplitude=amplitude2, center=center2[i], sigma=sigma2[i]
        )
        data[:, i] = peak1 + peak2

    # 노이즈 추가
    data += rng.normal(scale=0.05, size=data.shape)

    # xarray 포맷
    return xr.DataArray(
        data,
        coords={"x": x, "y": y},
        dims=("x", "y"),
    )


def test_bound():
    data = get_test_data_2d()
    # data.plot()
    # plt.show()
    model = LorentzianModel(prefix="p0_") + LorentzianModel(prefix="p1_")
    # data.fit.fit_with_corr(model=model, bound_ratio=1e-1)
    fit_result = data.fit.fit_with_corr(model=model, method="least_squares")
    fit_result.display()
