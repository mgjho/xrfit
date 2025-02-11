## v0.9.0 (2025-02-11)

### ✨ Features

- improve fit_with_corr to update bounds iteratively (#26)
- add max option in start_dict kws for fit_with_corr method

### 🐞 Bug Fixes

- solve set method bound issue #28
- add bound_tol parameter to set_bounds for improved parameter limit handling (#27)
- update data plotting to use x values

## v0.8.0 (2025-02-08)

### ✨ Features

- enhance _set_bounds function to conditionally set parameter limits (#25)
- add set_bound method for ParamsAccessor (#23)
- improve parameter and component visualization (#22)
- add distinguishable colors to components plot
feat: add parameter status label
- add tolerance to parameter value checks in DisplayAccessor
- add parameter display for DisplayAccessor (#21)

## v0.7.0 (2025-02-05)

### ✨ Features

- add component fit and plot (#20)
- enhance FitAccessor to accept additional keyword arguments in fit method
- add ArrAccessor for getting fit_arr values (#18)
- add visualization of fit stat to DisplayAccessor (#17)

## v0.5.0 (2025-02-05)

### ✨ Features

- implement auto-estimating initial coords for fit_with_corr method (#14)

## v0.4.0 (2025-02-05)

### ✨ Features

- add y-axis fix toggle for display accessor  (#13)

## v0.3.0 (2025-02-04)

### ✨ Features

- add fit_with_correlation method for FitAccessor (#12)

## v0.2.0 (2025-02-03)

### ✨ Features

- add params and display accessor (#10)

## v0.1.0 (2025-01-30)
