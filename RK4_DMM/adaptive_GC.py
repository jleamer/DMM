import numpy as np
from numba import njit
from scipy import sparse, linalg
from adaptive_step_dmm import CAdaptiveDMM
from pyscf import gto, dft
import numpy.ma as ma
import time

# Class for linear GC RK4 method
class CAdaptive_GC_RK4(CAdaptiveDMM):
    """
    DMM GCP via RK4.
    Using Jacob's code
    """

    def __init__(self, *, ovlp, H, mu, **kwargs):
        """
        :param inv_ovlp: the overlap matrix
        :param H: Hamiltonian
        :param mu: chemical potential
        """
        assert ovlp.shape == H.shape

        # saving arguments
        self.ovlp = ovlp
        self.H = H
        self.mu = mu

        # the inverse overlap matrix
        self.inv_ovlp = linalg.inv(ovlp)

        self.identity = np.identity(self.H.shape[0])

        self.scaledH = -0.5 * (self.inv_ovlp @ self.H - self.mu * self.identity)

        # initial density matrix
        self.rho = 0.5 * self.ovlp

        # call the parent's constructor
        CAdaptiveDMM.__init__(self, **kwargs)


    def rhs(self, rho):
        """
        Method implements the rhs of the derivative expression for minimizing rho
        """
        k = (self.identity - self.inv_ovlp @ rho) @ self.scaledH
        return rho @ k + k.conj().T @ rho


    def single_step_propagation(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """
        # alias
        rho = self.rho_next

        k1 = self.rhs(rho)

        k2 = self.rhs(rho + 0.5 * dbeta * k1)

        k3 = self.rhs(rho + 0.5 * dbeta * k2)

        k4 = self.rhs(rho + dbeta * k3)

        rho += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)


# Class for linear GC RK4 method using sparse matrices
class CAdaptive_GC_RK4_S(CAdaptiveDMM):
    """
    DMM GCP via RK4.
    Using Jacob's code
    """

    def __init__(self, *, ovlp, H, mu, **kwargs):
        """
        :param inv_ovlp: the overlap matrix
        :param H: Hamiltonian
        :param mu: chemical potential
        """
        assert ovlp.shape == H.shape

        # saving arguments
        self.ovlp = sparse.csr_matrix(ovlp)
        self.H = sparse.csr_matrix(H)
        self.mu = mu

        # the inverse overlap matrix
        self.inv_ovlp = sparse.csr_matrix(linalg.inv(ovlp))

        self.identity = sparse.csr_matrix(np.identity(self.H.shape[0]))

        self.scaledH = -0.5 * (self.inv_ovlp @ self.H - self.mu * self.identity)

        # initial density matrix
        self.rho = 0.5 * self.ovlp

        """
        # Save sparse copies of matrices for testing
        self.sparse_ovlp = sparse.csr_matrix(self.ovlp)
        self.sparse_scaledH = sparse.csr_matrix(self.scaledH)
        self.sparse_inv_ovlp = sparse.csr_matrix(self.inv_ovlp)
        self.sparse_rho = sparse.csr_matrix(self.rho)
        self.sparse_id = sparse.csr_matrix(self.identity)
        """
        # call the parent's constructor
        CAdaptiveDMM.__init__(self, **kwargs)

        # override rho_next from parent constructor
        self.rho_next = sparse.csr_matrix(self.rho, copy=True)

    def rhs(self, rho):
        k = (self.identity - self.inv_ovlp * rho) * self.scaledH
        return rho * k + k.conj().T * rho

    def single_step_propagation(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """
        # alias
        rho = self.rho_next
        k1 = self.rhs(rho)

        k2 = self.rhs(rho + 0.5 * dbeta * k1)

        k3 = self.rhs(rho + 0.5 * dbeta * k2)

        k4 = self.rhs(rho + dbeta * k3)

        self.rho_next += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)

# Class for non linear GC RK4 method
class CAdaptive_GC_RK4_NL(CAdaptiveDMM):
    """
    DMM GC with adaptive step approach using RK4 method for systems where
        H = H_core + H_exc(P)
    """
    def __init__(self, *, ovlp, H, mu, mf, **kwargs):
        """
        :param inv_ovlp: the overlap matrix
        :param H: Hamiltonian
        :param mu: chemical potential
        :param mf: df.RKS class from pyscf for performing self-consistent calculations
        """
        assert ovlp.shape == H.shape

        # saving arguments
        self.ovlp = ovlp
        self.H = H
        self.mu = mu
        self.mf = mf

        # the inverse overlap matrix
        self.inv_ovlp = linalg.inv(ovlp)

        self.identity = np.identity(self.H.shape[0])

        self.scaledH = -0.5 * (self.inv_ovlp @ self.H - self.mu * self.identity)

        # initial density matrix
        self.rho = 0.5 * self.ovlp

        # call the parent's constructor
        CAdaptiveDMM.__init__(self, **kwargs)

    def rhs(self, rho):
        """
        Function that implements right-hand side of derivative for minimizing density matrix
        :param rho: local copy of density matrix to avoid overwriting data by mistake
        :return:    rhs of derivative expression
        """
        h = self.H + self.mf.get_veff(self.mf.mol, rho)
        self.scaledH = -0.5 * (self.inv_ovlp @ h - self.mu * self.identity)
        k = (self.identity - self.inv_ovlp @ rho) @ self.scaledH
        return rho @ k + k.conj().T @ rho

    def single_step_propagation(self, dbeta):
        """
        Function that propagates self.rho_next by a single step of size dbeta
        :param dbeta:   step size of inverse temperature
        :return:
        """
        rho = self.rho_next

        k1 = self.rhs(rho)

        k2 = self.rhs(rho + 0.5 * k1 * dbeta)

        k3 = self.rhs(rho + 0.5 * k2 * dbeta)

        k4 = self.rhs(rho + k3 * dbeta)

        rho += (1/6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ##############################################################################################
    #
    #   Use Pyscf to obtain HF parameters for testing classes
    #
    ##############################################################################################

    # PBE Hydrogen flouride in a 6-31G basis set.
    mol = gto.Mole()
    mol.build(
        atom='H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis='631g',
        symmetry=True,
    )

    mf = dft.RKS(mol)
    # mf.xc = 'blyp' # shorthand for b88,lyp
    mf.xc = 'pbe'  # shorthand for pbe,pbe
    # mf.xc = 'lda,vwn_rpa'
    # mf.xc = 'pbe0'
    # mf.xc = 'b3lyp'

    # this where self-content diagonalization happens
    mf.kernel()

    # Orbital energies, Mulliken population etc.
    mf.analyze()

    # Get the converged density matrix (it generates the density matrix)
    dm = mf.make_rdm1()

    # Get the nuclear-nuclear repulsion energy
    e_nuc = mf.energy_nuc()

    # Get the 'core' hamiltonian, corresponding to kinetic energy and e-nuclear repulsion terms
    h1e = mf.get_hcore()

    # Compute the kinetic + e-nuclear repulsion energy
    e1 = np.einsum('ij,ji', h1e, dm)

    # Get the kohn-sham potential, including the Hartree coulomb repulsion and exchange-correlation potential, integrated on a grid
    vhf = mf.get_veff(mf.mol, dm)

    # Total energy
    tot_e = e1 + vhf.ecoul + vhf.exc + e_nuc  # Total energy is sum of terms
    print('Total dft energy: {}'.format(tot_e))

    # chemical potential
    index = int(mol.nelectron / 2)
    mu = (mf.mo_energy[index] + mf.mo_energy[index - 1]) / 2.
    print('Chemical Potential: ', str(mu))

    # get the overlap matrix and invert
    ovlp = mf.get_ovlp()
    inv_ovlp = linalg.inv(ovlp)

    ##############################################################################################
    #
    #   Linear HF model (only core H)
    #
    ##############################################################################################

    core_spect = linalg.eigvalsh(h1e, ovlp)
    num_electrons = 10
    index = int(num_electrons / 2)
    mu = (core_spect[index] + core_spect[index - 1]) / 2
    print(core_spect)
    print(mu)
    beta = 1
    dbeta = beta / 10000
    ferm_exact = ovlp @ linalg.funm(inv_ovlp @ h1e, lambda _: np.exp(-beta * (_ - mu)) / (1 + np.exp(-beta * (_ - mu))))

    gc = CAdaptive_GC_RK4(ovlp=ovlp, H=h1e, mu=mu, dbeta=dbeta, epsilon=1e-1)

    gc.propagate(beta)

    plt.title("Populations")
    plt.plot(linalg.eigvalsh(gc.rho, ovlp)[::-1], '*-', label="GC")
    plt.plot(linalg.eigvalsh(ferm_exact, ovlp)[::-1], '*-', label="FD")
    plt.legend(numpoints=1)
    plt.show()

    plt.title("Variable step method in action")
    plt.plot(gc.beta_increments, '*-')
    plt.xlabel('steps')
    plt.ylabel('dbeta')
    plt.show()

    ##############################################################################################
    #
    #   Full HF model
    #
    #############################################################################################

    def exact_single_step(rho_, *, h1e, mf, beta, inv_ovlp, ovlp, mu, **kwargs):
        h = h1e + mf.get_veff(mf.mol, rho_)
        rho = ovlp @ linalg.funm(inv_ovlp @ h, lambda _: np.exp(-beta * (_ - mu)) / (1 + np.exp(-beta * (_ - mu))))
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

            aitken_rho = rho_2 - (rho_2 - rho_1) ** 2 / ma.array(rho_2 - 2 * rho_1 + rho_0)
            aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

            rho_0 = aitken_rho

            norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

            if np.allclose(aitken_rho, prev_aitken_rho) and i > 5:
                print("Iterations converged!")
                break

        return aitken_rho, norm_diff


    beta = 3
    dbeta = beta / 10000

    # Calculate exact fermi dirac distribution
    func_args = {'h1e': h1e, 'mf': mf, 'mu': mu, 'ovlp': ovlp, 'inv_ovlp': inv_ovlp, 'beta': beta}
    ferm_exact, norm_diff = aitkens(ovlp / 2, 50, exact_single_step, **func_args)

    # Perform our method
    gcf = CAdaptive_GC_RK4_NL(ovlp=ovlp, H=h1e, mu=mu, dbeta=dbeta, epsilon=1e-1, mf=mf)
    gcf.propagate(beta)

    plt.title("Populations")
    plt.plot(linalg.eigvalsh(gcf.rho, ovlp)[::-1], '*-', label="GC")
    plt.plot(linalg.eigvalsh(ferm_exact, ovlp)[::-1], '*-', label="FD")
    plt.legend(numpoints=1)
    plt.show()

    plt.title("Variable step method in action")
    plt.plot(gcf.beta_increments, '*-')
    plt.xlabel('steps')
    plt.ylabel('dbeta')
    plt.show()

    plt.title("Aitken's Convergence")
    plt.plot(norm_diff, '*-')
    plt.xlabel('steps')
    plt.ylabel('norm')
    plt.show()