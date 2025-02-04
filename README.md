# xrfit

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/xrfit?logo=python&logoColor=white)](https://pypi.org/project/xrfit/)
[![PyPi](https://img.shields.io/pypi/v/xrfit?logo=pypi&logoColor=white)](https://pypi.org/project/xrfit/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/mgjho/xrfit/main.svg)](https://results.pre-commit.ci/latest/github/mgjho/xrfit/main)
[![Codecov Coverage](https://img.shields.io/codecov/c/github/mgjho/xrfit?logo=codecov&logoColor=white)](https://codecov.io/gh/mgjho/xrfit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview 🎯

[*xrfit*](https://github.com/mgjho/xrfit) provides Model-based, Multidimensional regression/analysis methods. By Integrating,

1. [*xarray*](https://xarray.pydata.org/) to handle multivariate data by utilizing coordinate-aware structure of DataArray.

2. [*lmfit*](https://lmfit.github.io/lmfit-py/) to deal complex regression problems using the composite-model based approach.

## Features ✨
1. __Model based regression__
    ```python
    model = Model_1() + Model_2()
    guess = data.fit.guess(model=model)
    result = data.fit(model=model, params=guess)
    ```
2. __Parameter handling__ (e.g. Sorting and Smoothing parameters)
    ```python
    sorted_result = result.params.sort("params_name")
    smoothened_result = sorted_result.params.smoothen("params_name")
    new_result = data.fit(model=model, params=smoothened_result.parmams)
    ```
3. __Interactive visualization__
    ```python
    result.display()
    ```
