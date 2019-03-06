import numpy as np
from scipy.optimize import differential_evolution as de


def stability_test(model, P, T, z, monitor=False):
    n_components = model.number_of_components
    search_space = [(0, 1)] * n_components
    f_z = model.fugacity(P, T, z)

    result = de(
        _reduced_tpd, 
        bounds=search_space, 
        args=[model, P, T, z, f_z],
        popsize=10,
        recombination=0.9,
        maxiter=50,
        disp=monitor,
        polish=False
    )

    x = result.x / result.x.sum()

    return x, result.fun


def _reduced_tpd(x, model, P, T, z, f_z):
    tol = 1e-2
    x = x / x.sum()
    condition1 = 1 - tol <= x.sum()
    condition2 = x.sum() <= 1 + tol
    condition3 = x.max() <= 1.0
    condition4 = x.min() >= 0.0
    feasible_conditions = condition1 and condition2 and condition3 and condition4

    f_x = model.fugacity(P, T, x)
    tpd = np.dot(x, (np.log(f_x) - np.log(f_z)))

    if not feasible_conditions:
        # Penalization if candidate is not in feasible space
        penalty_parameter = 10.
        tpd += penalty_parameter * _penalty_function(x, 1)

    return tpd


def _penalty_function(x, feasible_limit):
    feasible_region_marker = np.abs(x.sum() - feasible_limit)
    return feasible_region_marker * feasible_region_marker
