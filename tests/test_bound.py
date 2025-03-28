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
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 30)
    X, Y = np.meshgrid(x, y, indexing="ij")
    rng = np.random.default_rng(42)
    model = LorentzianModel()

    # 피크의 파라볼릭 중심과 폭 설정
    center1 = -2 + 0.1 * y**2
    center2 = 2 + 0.3 * y**2
    center3 = -1 + 0.5 * y**2
    sigma1 = 0.5 + 0.01 * y**2
    sigma2 = 0.5 + 0.01 * y**2
    sigma3 = 0.5 + 0.01 * y**2
    amplitude1 = 1.0 + y * 0.0
    amplitude2 = 0.1 + 0.1 * y**2
    amplitude3 = 0.2 + 0.3 * y**2

    # 데이터 생성
    data = np.zeros_like(X)
    for i in range(len(y)):
        peak1 = model.eval(
            x=x, amplitude=amplitude1[i], center=center1[i], sigma=sigma1[i]
        )
        peak2 = model.eval(
            x=x, amplitude=amplitude2[i], center=center2[i], sigma=sigma2[i]
        )
        peak3 = model.eval(
            x=x, amplitude=amplitude3[i], center=center3[i], sigma=sigma3[i]
        )
        data[:, i] = peak1 + peak2 + peak3

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
    model = (
        LorentzianModel(prefix="p0_")
        + LorentzianModel(prefix="p1_")
        + LorentzianModel(prefix="p2_")
    )
    # data.fit.fit_with_corr(model=model, bound_ratio=1e-1)
    # fit_result = data.fit.fit_with_corr(model=model, method="least_squares")
    fit_result = data.fit.fit_with_corr(
        model=model,
        method="least_squares",
        set_bound=True,
        bound_ratio=1e0,
    )
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221)
    fit_result.get_arr("data").T.plot(ax=ax, cmap="viridis")
    ax = fig.add_subplot(222)
    fit_result.get_arr("best_fit").T.plot(ax=ax, cmap="viridis")
    fit_result.params.get("center").plot(ax=ax, hue="params_dim")
    ax = fig.add_subplot(223)
    fit_result.get_arr("residual").T.plot(ax=ax)
    plt.show()
    # fit_result.display()
