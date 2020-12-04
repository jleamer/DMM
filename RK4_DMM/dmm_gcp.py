import numpy as np
from scipy.io import mmread, mmwrite
from scipy import linalg
from scipy import sparse
from numba import njit
import matplotlib.pyplot as plt
from pyscf import gto, dft
import numpy.ma as ma


def rhs(rho, h, inv_ovlp, identity, mu):
    """
    this function implements the rhs of the derivative expression for minimizing rho
    :param rho:         the current density matrix
    :param h:           the hamiltonian of the system
    :param inv_ovlp:    the inverse of the overlap matrix; for orthonormal systems, this is just the identity matrix
    :param identity:    the identity matrix
    :param mu:          the chemical potential
    :return:            the density matrix at the next step
    """
    scaledH = -0.5 * (inv_ovlp.dot(h) - mu * identity)
    k = (identity - inv_ovlp.dot(rho)).dot(scaledH)
    f = rho.dot(k) + k.conj().T.dot(rho)
    return f

def non_linear_rhs(rho, h1e, inv_ovlp, identity, mu, mf):
    h = h1e + mf.get_veff(mf.mol, rho)
    scaledH = -0.5 * (inv_ovlp.dot(h) - mu * identity)
    k = (identity - inv_ovlp.dot(rho)).dot(scaledH)
    f = rho.dot(k) + k.conj().T.dot(rho)
    return f

def non_linear_rk4(rhs, rho, dbeta, h, inv_ovlp, identity, mu, nsteps, mf):
    for i in range(nsteps):
        rhocopy = rho.copy()
        k1 = rhs(rhocopy, h, inv_ovlp, identity, mu, mf).copy()

        temp_rho = rhocopy + 0.5*dbeta*k1
        k2 = rhs(temp_rho, h, inv_ovlp, identity, mu, mf).copy()

        temp_rho = rhocopy + 0.5*dbeta*k2
        k3 = rhs(temp_rho, h, inv_ovlp, identity, mu, mf).copy()

        temp_rho = rhocopy + dbeta*k3
        k4 = rhs(temp_rho, h, inv_ovlp, identity, mu, mf).copy()

        rho += (1/6)*dbeta*(k1 + 2*k2 + 2*k3 + k4)

    return rho

def rk4(rhs, rho, dbeta, h, inv_ovlp, identity, mu, nsteps):
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
        k1 = rhs(rhocopy, h, inv_ovlp, identity, mu).copy()

        temp_rho = rhocopy + 0.5*dbeta*k1
        k2 = rhs(temp_rho, h, inv_ovlp, identity, mu).copy()

        temp_rho = rhocopy + 0.5*dbeta*k2
        k3 = rhs(temp_rho, h, inv_ovlp, identity, mu).copy()

        temp_rho = rhocopy + dbeta*k3
        k4 = rhs(temp_rho, h, inv_ovlp, identity, mu).copy()

        rho += (1/6)*dbeta*(k1 + 2*k2 + 2*k3 + k4)

    return rho


def single_step_purify(rho_, kwargs):
    """
    function to implement a single step of the purification algorithm where the function f(x) = 3x^2 - 2x^3 is applied
    to the density matrix to make it idempotent
    :param rho_:        the density matrix
    :param inv_ovlp:    the inverse of the overlap matrix; for orthonormal systems, this is just the identity matrix
    :return:            a slightly more idempotent density matrix
    """
    inv_ovlp = kwargs['inv_ovlp']
    rho_sq = rho_ @ inv_ovlp @ rho_
    rho_cu = rho_ @ inv_ovlp @ rho_sq
    return 3*rho_sq + 2*rho_cu


def single_step(rho_, *, h1e, mf, dbeta, inv_ovlp, mu, rk4steps, **kwargs):
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
    rho = rk4(rhs, rho_, dbeta, h, inv_ovlp, identity, mu, rk4steps)
    return rho


def exact_single_step(rho_, *, h1e, mf, beta, inv_ovlp, ovlp, mu, **kwargs):
    h = h1e + mf.get_veff(mf.mol, rho_)
    rho = ovlp @ linalg.funm(inv_ovlp @ h, lambda _: 1/(1+np.exp(beta*(_ - mu))))
    return rho

def exact0_single_step(rho_, *, h1e, mf, ovlp, inv_ovlp, mu, **kwargs):
    h = h1e + mf.get_veff(mf.mol, rho_)
    rho = ovlp @ linalg.funm(inv_ovlp @ h, lambda _: _ <= mu)
    return rho


def steady_single_step(rho_, *, h1e, mf, inv_ovlp, mu, **kwargs):
    h = h1e + mf.get_veff(mf.mol, rho_)
    rho = (rho_ @ inv_ovlp @ h + mu*rho_ @ inv_ovlp @ rho_ - rho_ @ inv_ovlp @ rho_ @ inv_ovlp @ h)/(mu)
    rho += rho.conj().T
    rho /= 2
    return rho


def steady_linear_single_step(rho_, *, h, mu, inv_ovlp, **kwargs):
    rho = (rho_ @ inv_ovlp @ h + mu*rho_ @ inv_ovlp @ rho_ - rho_ @ inv_ovlp @ rho_ @ inv_ovlp @ h)/(mu)
    rho += rho.conj().T
    rho /= 2
    return rho


def linear_single_step(rho_, *, h, mu, inv_ovlp, rk4steps, dbeta, **kwargs):
    identity = np.identity(rho_.shape[0])
    rho = rk4(rhs, rho_, dbeta, h, inv_ovlp, identity, mu, rk4steps)
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

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho) and i > 5:
            print("Iterations converged!")
            break

    return aitken_rho, norm_diff

if __name__ == '__main__':
    '''
    A simple example to run DFT calculation.
    '''

    # PBE Hydrogen flouride in a 6-31G basis set.
    mol = gto.Mole()
    mol.build(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '631g',
        symmetry = True,
    )

    mf = dft.RKS(mol)
    #mf.xc = 'blyp' # shorthand for b88,lyp
    mf.xc = 'pbe' # shorthand for pbe,pbe
    #mf.xc = 'lda,vwn_rpa'
    #mf.xc = 'pbe0'
    #mf.xc = 'b3lyp'

    # this where self-content diagonalization happens
    mf.kernel()

    # Orbital energies, Mulliken population etc.
    mf.analyze()

    # Get the converged density matrix (it generates the density matrix)
    dm = mf.make_rdm1()
    mmwrite('dft_density.mtx', sparse.coo_matrix(dm))


    # Get the nuclear-nuclear repulsion energy
    e_nuc = mf.energy_nuc()

    # Get the 'core' hamiltonian, corresponding to kinetic energy and e-nuclear repulsion terms
    h1e = mf.get_hcore()

    # Compute the kinetic + e-nuclear repulsion energy
    e1 = np.einsum('ij,ji', h1e, dm)

    # Get the kohn-sham potential, including the Hartree coulomb repulsion and exchange-correlation potential, integrated on a grid
    vhf = mf.get_veff(mf.mol, dm)

    # Total energy
    tot_e = e1 + vhf.ecoul + vhf.exc + e_nuc    # Total energy is sum of terms
    print('Total dft energy: {}'.format(tot_e))

    # chemical potential
    index = int(mol.nelectron/2)
    mu = (mf.mo_energy[index] + mf.mo_energy[index - 1]) / 2.
    print('Chemical Potential: ', str(mu))
    f = open('dft_mu.txt', 'w+')
    f.write(str(mu))
    f.close()

    # get the overlap matrix and print to file
    ovlp = mf.get_ovlp()
    mmwrite('dft_overlap.mtx', sparse.coo_matrix(ovlp))

    print(linalg.eigvalsh(h1e, ovlp))

    inv_ovlp = np.linalg.inv(ovlp)
    better_mu = -4
    num_electrons = 10
    init_rho = num_electrons/ovlp.trace() * ovlp
    test = init_rho.copy()
    #init_rho = ovlp
    dbeta = 0.003

    rho = rk4(rhs, init_rho, dbeta, h1e, inv_ovlp, np.identity(init_rho.shape[0]), mu, 1000)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    im = ax1.imshow(rho.real, origin='lower')
    ax1.set_xlabel("j")
    ax1.set_ylabel("i")
    ax1.set_title("RK4")
    fig1.colorbar(im, ax=ax1)

    # Now try Aitken's convergence
    func_args = {'h1e': h1e, 'mf': mf, 'mu': better_mu, 'inv_ovlp': inv_ovlp, 'dbeta': dbeta, 'rk4steps': 1000}
    aitkens_rho, norm_diff = aitkens(rho, 50, single_step, func_args)

    fig2 = plt.figure(2)
    ax21 = fig2.add_subplot(121)
    ax21.semilogy(norm_diff, '*-')
    ax21.set_xlabel("Iteration #")
    ax21.set_ylabel("||P_n+1 - P_n||")
    ax21.set_title("Aitken's Convergence GCP")

    ax22 = fig2.add_subplot(122)
    im = ax22.imshow(aitkens_rho, origin='lower')
    ax22.set_xlabel("j")
    ax22.set_ylabel("i")
    ax22.set_title("Final Aitken's Rho")
    fig2.colorbar(im, ax=ax22)

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    im = ax3.imshow(inv_ovlp.real)
    ax3.set_title("Inv Overlap")
    ax3.set_xlabel("i")
    ax3.set_ylabel("j")
    fig3.colorbar(im, ax=ax3)
    plt.show()
