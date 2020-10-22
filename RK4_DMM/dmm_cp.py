import numpy as np
from scipy.io import mmread, mmwrite
from scipy import linalg
from scipy import sparse
from numba import njit
import matplotlib.pyplot as plt
from pyscf import gto, dft
import numpy.ma as ma


def rhs(rho, H, inv_ovlp, identity):
    """
    this function implements the rhs of the derivative for minimizing rho
    :param rho:
    :param h:
    :param inv_ovlp:
    :param identity:
    :return:
    """
    c = rho.dot(identity-inv_ovlp.dot(rho))
    d = H.dot(c)
    alpha = np.sum(inv_ovlp*d.T)/c.trace()
    scaledH = -0.5*(inv_ovlp.dot(H) - alpha*identity)
    K = (identity - inv_ovlp.dot(rho)).dot(scaledH)
    f = rho.dot(K) + K.conj().T.dot(rho)
    return f


def rk4(rhs, rho, dbeta, h, inv_ovlp, identity, nsteps):
    """
    this function implements an RK4 method for calculating the final rho using the rhs
    :param rhs:         the rhs function to use
    :param rho:         the current density matrix
    :param dbeta:       the change in beta at each step
    :param h:           the hamiltonian of the system
    :param inv_ovlp:    the inverse of the overlap matrix; for orthonormal systems, this is just the identity matrix
    :param identity:    the identity matrix
    :param mu:          the chemical potential
    :param nsteps:      the number of steps to propagate through; final beta will be dbeta*nsteps
    :return:            the final density matrix
    """

    for i in range(nsteps):
        rhocopy = rho.copy()

        k1 = rhs(rhocopy, h, inv_ovlp, identity).copy()

        temp_rho = rhocopy + 0.5*dbeta*k1
        k2 = rhs(temp_rho, h, inv_ovlp, identity).copy()

        temp_rho = rhocopy + 0.5*dbeta*k2
        k3 = rhs(temp_rho, h, inv_ovlp, identity).copy()

        temp_rho = rhocopy + dbeta*k3
        k4 = rhs(temp_rho, h, inv_ovlp, identity).copy()

        rho += (1/6)*dbeta*(k1 + 2*k2 + 2*k3 + k4)

    return rho


def single_step(rho_, *, h1e, mf, dbeta, inv_ovlp, rk4steps, **kwargs):
    """
    function that implements a single step for the Aitken delta-sq process
    :param rho_:        the density matrix to be iterated over
    :param h1e:         the core hamiltonian of the system
    :param hexc:        the exchange hamiltonian
    :param dbeta:      the amount to change beta by in each iteration for calculating the new density matrix
    :param rk4steps:    the number of steps to iterate the RK4 method for calculating the different density matrix
    :return:            the next density matrix
    """
    identity = np.identity(rho_.shape[0])
    h = h1e + mf.get_veff(mf.mol, rho_)
    rho = rk4(rhs, rho_, dbeta, h, inv_ovlp, identity, rk4steps)
    return rho

def exact_single_step(rho_, *, h1e, mf, beta, inv_ovlp, ovlp, mu, **kwargs):
    h = h1e + mf.get_veff(mf.mol, rho_)
    mu = get_mu(rho_, h, inv_ovlp)
    rho = ovlp @ linalg.funm(inv_ovlp @ h, lambda _: 1/(1+np.exp(beta*(_ - mu))))
    return rho

def exact0_single_step(rho_, *, h1e, mf, ovlp, inv_ovlp, mu, **kwargs):
    h = h1e + mf.get_veff(mf.mol, rho_)
    mu = get_mu(rho_, h, inv_ovlp)
    rho = ovlp @ linalg.funm(inv_ovlp @ h, lambda _: _ <= mu)
    return rho

def get_mu(rho, h, inv_ovlp):
    identity = np.identity(rho.shape[0])
    temp = rho @ (identity - inv_ovlp @ rho)
    return np.sum(inv_ovlp @ h * temp.T)/temp.trace()

def steady_linear_single_step(rho_, *, h, mu, inv_ovlp, **kwargs):
    mu = get_mu(rho_, h, inv_ovlp)
    rho = (rho_ @ inv_ovlp @ h + mu*rho_ @ inv_ovlp @ rho_ - rho_ @ inv_ovlp @ rho_ @ inv_ovlp @ h)/(mu)
    rho += rho.conj().T
    rho /= 2
    return rho

def aitkens(rho, nsteps, single_step_func, **func_args):
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
        rho_1 = single_step_func(rho_0, **func_args)
        rho_2 = single_step_func(rho_1, **func_args)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        func_args['mu'] = get_mu(rho_0, func_args['h1e'], func_args['inv_ovlp'])

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho) and i > 5:
            print("Iterations converged!")
            break

    return aitken_rho, norm_diff
