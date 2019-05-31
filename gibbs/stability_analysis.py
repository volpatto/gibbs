import numpy as np
import attr
from scipy.optimize import differential_evolution as de


@attr.s(auto_attribs=True)
class StabilityResult:
    phase_split: bool
    x: np.ndarray
    reduced_tpd: float


def stability_test(model, P, T, z, monitor=False, rtol=1e-3, polish=True):
    n_components = model.number_of_components
    search_space = [(0, 1)] * n_components
    f_z = model.fugacity(P, T, z)

    result = de(
        _reduced_tpd,
        bounds=search_space, 
        args=[model, P, T, f_z],
        popsize=30,
        recombination=0.95,
        mutation=0.6,
        tol=1e-3,
        disp=monitor,
        polish=polish
    )

    x = result.x / result.x.sum()
    reduced_tpd = result.fun

    if np.allclose(x, z, rtol=rtol):  # This criterion could be separated in a function itself
        # with a norm
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


def _reduced_tpd(n, model, P, T, f_z):
    try:
        x = n / n.sum()
        f_x = model.fugacity(P, T, x)
        tpd = np.sum(x * (np.log(f_x / f_z)))
    except TypeError:
        tpd = np.inf

    return tpd
