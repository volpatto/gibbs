import numpy as np
import attr
from scipy.optimize import differential_evolution as de
from scipy.optimize import minimize


@attr.s(auto_attribs=True)
class StabilityResult:
    phase_split: bool
    x: np.ndarray
    reduced_tpd: float


def stability_test(model, P, T, z, monitor=False, ctol=1e-5, rtol=1e-3, polish=False):
    n_components = model.number_of_components
    search_space = [(0, 1)] * n_components
    f_z = model.fugacity(P, T, z)

    result = de(
        _reduced_tpd, 
        bounds=search_space, 
        args=[model, P, T, f_z, ctol],
        popsize=50,
        recombination=0.95,
        mutation=0.6,
        tol=1e-8,
        disp=monitor,
        polish=False
    )

    x = result.x / result.x.sum()

    if polish:

        cons = (
            {'type': 'ineq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)}
        )

        result = minimize(
            _reduced_tpd,
            x,
            args=(model, P, T, f_z, ctol),
            bounds=search_space,
            method='SLSQP',
            constraints=cons,
            jac=False,
            tol=1e-10,
        )
        x = result.x

    reduced_tpd = result.fun

    if np.allclose(x, z, rtol=rtol):  # This criterium could be separated in a function itself with a norm
        phase_split = False
    elif reduced_tpd < 0.:
        phase_split = True
    elif np.abs(reduced_tpd) < 1e-10:
        phase_split = False
    else:
        phase_split = False

    stability_test_result = StabilityResult(
        phase_split=phase_split,
        x=x,
        reduced_tpd=reduced_tpd
    )

    return stability_test_result


def _reduced_tpd(x, model, P, T, f_z, tol=1e-5):
    try:
        f_x = model.fugacity(P, T, x)
        tpd = np.sum(x * (np.log(f_x / f_z)))

        if not 1 - tol <= np.sum(x) <= 1 + tol:
            # Penalization if candidate is not in feasible space
            penalty_parameter = 1e3
            tpd += penalty_parameter * _penalty_function(x, 1)
    except TypeError:
        tpd = np.inf

    return tpd


def _penalty_function(x, feasible_limit):
    feasible_region_marker = np.abs(x.sum() - feasible_limit)
    return feasible_region_marker * feasible_region_marker
