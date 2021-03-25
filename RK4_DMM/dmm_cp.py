import numpy as np
from scipy.io import mmread, mmwrite
from scipy import linalg
from scipy import sparse
from numba import njit
import matplotlib.pyplot as plt
from pyscf import gto, dft
import numpy.ma as ma


def rhs(rho, H, inv_ovlp, identity, num_electrons, ovlp, mu, beta):
    """
    this function implements the rhs of the derivative for minimizing rho
    :param rho:
    :param h:
    :param inv_ovlp:
    :param identity:
    :return:
    """
    # calculate the prefactor
    n = 2 * num_electrons / ovlp.trace()

    if beta == 0:
        dmu = 0
        a = rho @ (identity - inv_ovlp @ rho / n) @ inv_ovlp @ H
        a += a.conj().T
        b = rho @ (identity - inv_ovlp @ rho / n)
        b += b.conj().T
    else:
        # Calculate dmu/dbeta
        a = rho @ (identity - inv_ovlp @ rho / n) @ inv_ovlp @ H
        a += a.conj().T
        b = rho @ (identity - inv_ovlp @ rho / n)
        b += b.conj().T
        dmu = (a.trace() / b.trace() - mu) / beta

    '''
    # Calculate dmu/dbeta
    a = rho @ (identity - inv_ovlp @ rho / n) @ inv_ovlp @ H
    a += a.conj().T
    b = rho @ (identity - inv_ovlp @ rho / n)
    b += b.conj().T
    dmu = (a.trace() / b.trace() - mu) / beta
    '''
    # calculate dP/dbeta
    scaledH = -0.5 * (inv_ovlp @ H - a.trace() / b.trace() * identity)

    k = rho @ (identity - inv_ovlp @ rho / n) @ scaledH
    dP = k + k.conj().T
    return dP, dmu

def zvode_rhs(beta, x,  H, inv_ovlp, identity, num_electrons, ovlp):
    mu = x[-1]
    rho = x[:-1].reshape(*H.shape)
    return np.append(*rhs(rho, H, inv_ovlp, identity, num_electrons, ovlp, mu, beta))


def non_linear_rhs(rho, h1e, inv_ovlp, identity, mf, num_electrons, ovlp, mu, beta):
    # update H with new value of rho
    H = h1e + mf.get_veff(mf.mol, rho)

    # calculate the prefactor
    n = 2*num_electrons/ovlp.trace()

    if beta == 0:
        dmu = 0
        a = rho @ (identity - inv_ovlp @ rho / n) @ inv_ovlp @ H
        a += a.conj().T
        b = rho @ (identity - inv_ovlp @ rho / n)
        b += b.conj().T
    else:
        # Calculate dmu/dbeta
        a = rho @ (identity - inv_ovlp @ rho/n) @ inv_ovlp @ H
        a += a.conj().T
        b = rho @ (identity - inv_ovlp @ rho/n)
        b += b.conj().T
        dmu = (a.trace()/b.trace() - mu)/beta

    '''
    a = rho @ (identity - inv_ovlp @ rho / n) @ inv_ovlp @ H
    a += a.conj().T
    b = rho @ (identity - inv_ovlp @ rho / n)
    b += b.conj().T
    dmu = (a.trace() / b.trace() - mu) / beta
    '''
    # calculate dP/dbeta
    scaledH = -0.5*(inv_ovlp @ H - a.trace() / b.trace() * identity)
    #scaledH = -0.5*(inv_ovlp @ H - (beta*dmu + mu)*identity)

    k = rho @ (identity - inv_ovlp @ rho/n) @ scaledH
    dP = k + k.conj().T

    return dP, dmu

def non_linear_rk4(rhs, rho, dbeta, h, inv_ovlp, identity, nsteps, mf, num_electrons, ovlp, mu, beta):
    list_H = []
    rhocopy = rho.copy()
    curr_H = h + mf.get_veff(mf.mol, rhocopy)
    list_H.append(curr_H)
    for i in range(nsteps):
        rhocopy = rho.copy()
        k1, l1 = rhs(rhocopy, h, inv_ovlp, identity, mf, num_electrons, ovlp, mu, beta)

        temp_rho = rhocopy + 0.5*dbeta*k1
        temp_mu = mu + 0.5*dbeta*l1
        k2, l2 = rhs(temp_rho, h, inv_ovlp, identity, mf, num_electrons, ovlp, temp_mu, beta)

        temp_rho = rhocopy + 0.5*dbeta*k2
        temp_mu = mu + 0.5*dbeta*l2
        k3, l3 = rhs(temp_rho, h, inv_ovlp, identity, mf, num_electrons, ovlp, temp_mu, beta)

        temp_rho = rhocopy + dbeta*k3
        temp_mu = mu + dbeta*l3
        k4, l4 = rhs(temp_rho, h, inv_ovlp, identity, mf, num_electrons, ovlp, temp_mu, beta)

        rho += (1/6)*dbeta*(k1 + 2*k2 + 2*k3 + k4)
        mu += (1/6)*dbeta*(l1 + 2*l2 + 2*l3 + l4)
        beta += dbeta
        rhocopy = rho.copy()
        list_H.append(h + mf.get_veff(mf.mol, rhocopy))

    return rho, mu, list_H

def rk4(rhs, rho, dbeta, h, inv_ovlp, identity, nsteps, num_electrons, ovlp, mu, beta):
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
        k1, l1 = rhs(rhocopy, h, inv_ovlp, identity, num_electrons, ovlp, mu, beta)

        temp_rho = rhocopy + 0.5 * dbeta * k1
        k2, l2 = rhs(temp_rho, h, inv_ovlp, identity, num_electrons, ovlp, mu, beta)

        temp_rho = rhocopy + 0.5 * dbeta * k2
        k3, l3 = rhs(temp_rho, h, inv_ovlp, identity, num_electrons, ovlp, mu, beta)

        temp_rho = rhocopy + dbeta * k3
        k4, l4 = rhs(temp_rho, h, inv_ovlp, identity, num_electrons, ovlp, mu, beta)

        rho += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)
        mu += (1 / 6) * dbeta * (l1 + 2 * l2 + 2 * l3 + l4)
        beta += dbeta

    return rho, mu


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

def exact_single_step(rho_, *, h1e, mf, beta, inv_ovlp, ovlp, mu, num_electrons, **kwargs):
    h = h1e + mf.get_veff(mf.mol, rho_)
    #mu = get_mu(rho_, h, inv_ovlp, num_electrons, ovlp)
    #print(mu)
    #rho = 2 * num_electrons * ovlp / ovlp.trace() @ linalg.funm(inv_ovlp @ h, lambda _: np.exp(-beta*(_ - mu))/(1+np.exp(-beta*(_ - mu))))
    rho = ovlp @ linalg.funm(inv_ovlp @ h, lambda _: np.exp(-beta*(_ - mu))/(1+np.exp(-beta*(_ - mu))))
    rho *= num_electrons/rho.trace()
    return rho

def exact0_single_step(rho_, *, h1e, mf, ovlp, inv_ovlp, mu, **kwargs):
    h = h1e + mf.get_veff(mf.mol, rho_)
    mu = get_mu(rho_, h, inv_ovlp)
    rho = ovlp @ linalg.funm(inv_ovlp @ h, lambda _: _ <= mu)
    return rho

def get_mu(rho, h, inv_ovlp, num_electrons, ovlp):
    identity = np.identity(rho.shape[0])
    n = num_electrons / rho.trace()
    c = rho @ (identity - inv_ovlp @ rho / n)
    d = h @ c
    alpha = np.sum(inv_ovlp * d.T) / c.trace()
    return alpha

def steady_linear_single_step(rho_, *, h, inv_ovlp, **kwargs):
    mu = get_mu(rho_, h, inv_ovlp)
    rho = (rho_ @ inv_ovlp @ h + mu*rho_ @ inv_ovlp @ rho_ - rho_ @ inv_ovlp @ rho_ @ inv_ovlp @ h)/(mu)
    rho += rho.conj().T
    rho /= 2
    return rho

def steady_single_step(rho_, *, h, inv_ovlp, mf, **kwargs):
    H = h + mf.get_veff(mf.mol, rho_)
    mu = get_mu(rho_, H, inv_ovlp)
    rho = (rho_ @ inv_ovlp @ H + mu*rho_ @ inv_ovlp @ rho_ - rho_ @ inv_ovlp @ rho_ @ inv_ovlp @ H)/(mu)
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
        #func_args['mu'] = get_mu(rho_0, func_args['h1e'], func_args['inv_ovlp'], func_args['num_electrons'], func_args['ovlp'])

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho) and i > 5:
            print("Iterations converged!")
            break

    return aitken_rho, norm_diff
