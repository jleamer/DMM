import numpy as np
from scipy.io import mmread, mmwrite
from scipy import linalg
from scipy import sparse
from numba import njit
import matplotlib.pyplot as plt
from pyscf import gto, dft
import numpy.ma as ma

def aitkens(rho, nsteps, single_step_func, func_args):
    """
    function for performing the Aitken's delta-squared convergence method
    :param rho:                 the density matrix to start the convergence with
    :param nsteps:              the number of steps to try converging for
    :param single_step_func:    the function that generates the next density matrix
    :param func_args:           the extra arguments for the single step function
    :return:                    the converged density matrix and the norm differences
    """
    norm_diff = []
    rho_0 = rho.copy()
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step_func(rho_0, func_args)
        rho_2 = single_step_func(rho_1, func_args)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

    return aitken_rho, norm_diff
