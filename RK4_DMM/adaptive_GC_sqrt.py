import numpy as np
from numba import njit
from scipy import sparse, linalg
from adaptive_step_DMM_sqrt import CAdaptiveDMMsqrt
from pyscf import gto, dft
import numpy.ma as ma
import time


class CAdaptive_GC_RK4_sqrt(CAdaptiveDMMsqrt):
    def __init__(self, *, ovlp, H, mu, rho=None, omega=None, **kwargs):
        assert ovlp.shape == H.shape

        # Save variables
        self.ovlp = ovlp
        self.sqrt_ovlp = linalg.sqrtm(self.ovlp)
        self.H = H
        self.mu = mu

        if rho is None:
            self.rho = ovlp / 2
        else:
            self.rho = rho

        if omega is None:
            self.omega = self.sqrt_ovlp / np.sqrt(2)
        else:
            self.omega = omega

        self.omega = self.omega.astype(complex)

        assert np.allclose(self.omega.conj().T @ self.omega, self.rho)

        # Get additional matrices for later use
        self.inv_ovlp = linalg.inv(ovlp)
        self.inv_sqrt_ovlp = linalg.sqrtm(self.inv_ovlp)
        self.identity = np.eye(ovlp.shape[0])
        self.scaledH = -0.5 * (self.inv_ovlp @ self.H - self.mu * self.identity)

        self.num_electrons = self.rho.trace()
        self.num_electron_list = [self.rho.trace()]

        self.rho_list = [self.rho]
        self.omega_list = [self.omega]

        # We use S^-1 @ H so much, so we define it here
        self.A = self.inv_ovlp @ self.H

        # call the parent's constructor
        CAdaptiveDMMsqrt.__init__(self, **kwargs)

        self.omega_next = np.empty_like(self.omega)

    def rhs(self, q):
        """
        Method implements the rhs of the derivative expression for minimizing rho
        """
        #k = (self.identity - q @ q) @ self.scaledH
        #return q @ k + k.conj().T @ q.conj().T
        # 4 matrix multiplications by my count
        temp = self.inv_sqrt_ovlp @ q
        self.count += 1
        domega = q @ (self.identity - (temp @ temp)) @ self.scaledH

        #indx = np.abs(domega) < self.tol
        #domega[indx] = 0

        #self.sparsity.append(np.sum(indx) / self.H.shape[0] ** 2)
        return domega

    def single_step_rk2(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """
        # alias
        omega = self.omega_next

        # RK2
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + 0.5 * dbeta * k1)

        omega += dbeta * k2

        # Enforce sparsity on omega
        indx = np.abs(omega) < self.tol
        omega[indx] = 0

        # Update rho and the energy
        rho = omega.conj().T @ omega
        self.energy_next = np.sum(rho.conj() * self.A)

    def single_step_rk4(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """
        # alias
        omega = self.omega_next

        # rk4 approach
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + 0.5 * dbeta * k1)
        k3 = self.rhs(omega + 0.5 * dbeta * k2)
        k4 = self.rhs(omega + dbeta * k3)

        omega += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce sparsity on omega
        indx = np.abs(omega) < self.tol
        omega[indx] = 0

        # Update rho and the energy
        rho = omega.conj().T @ omega
        self.energy_next = np.sum(rho.conj() * self.A)

    def single_step_predictor_corrector(self, dbeta):
        """
        Propagate self.omega_next by a single step using predictor corrector method
        :param dbeta: a step size of inverse temperature
        :return:
        """
        # alias
        omega = self.omega_next

        # Calculate first step using euler approximation
        k1 = self.rhs(omega)
        eta = omega.copy() + dbeta*k1

        # Calculate next part
        l1 = self.rhs(eta) + k1

        # Update omega
        omega += dbeta / 2 * l1

        # Enforce sparsity
        indx = np.abs(omega) < self.tol
        omega[indx] = 0

        # Update rho and the energy
        rho = omega.conj().T @ omega
        self.energy_next = np.sum(rho.conj() * self.A)


    def single_step_propagation(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """

        # RK4 approach
        # alias
        omega = self.omega_next

        """
        k1 = self.rhs(omega)

        k2 = self.rhs(omega + 0.5 * dbeta * k1)

        k3 = self.rhs(omega + 0.5 * dbeta * k2)

        k4 = self.rhs(omega + dbeta * k3)

        omega += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)
        """
        # RK2 approach
        k1 = self.rhs(omega)

        k2 = self.rhs(omega + 0.5 * dbeta * k1)

        omega += dbeta * k2
        indx = np.abs(omega) < self.tol
        omega[indx] = 0
        #"""
        rho = omega.conj().T @ omega
        self.energy_next = np.trace(rho @ self.inv_ovlp @ self.H)


class CAdaptive_GC_RK4_NL_sqrt(CAdaptiveDMMsqrt):
    """
    DMM GC with adaptive step approach using RK4 method for systems where
        H = H_core + H_exc(P)
    """
    def __init__(self, *, ovlp, H, mu, mf, rho=None, omega=None, **kwargs):
        """
        :param inv_ovlp: the overlap matrix
        :param H: Hamiltonian
        :param mu: chemical potential
        :param mf: df.RKS class from pyscf for performing self-consistent calculations
        """
        assert ovlp.shape == H.shape

        # saving arguments
        self.ovlp = ovlp
        self.H = H.astype(complex)
        self.mu = mu
        self.mf = mf

        # the inverse overlap matrix
        self.sqrt_ovlp = linalg.sqrtm(ovlp)
        self.inv_ovlp = linalg.inv(ovlp)
        self.inv_sqrt_ovlp = linalg.sqrtm(self.inv_ovlp)
        self.identity = np.identity(self.H.shape[0])
        self.scaledH = -0.5 * (self.inv_ovlp @ self.H - self.mu * self.identity)

        if rho is None:
            self.rho = ovlp / 2
        else:
            self.rho = rho

        if omega is None:
            self.omega = self.sqrt_ovlp / np.sqrt(2)
        else:
            self.omega = omega

        self.omega = self.omega.astype(complex)

        assert np.allclose(self.omega.conj().T @ self.omega, self.rho)

        self.num_electrons = self.rho.trace()
        # call the parent's constructor
        CAdaptiveDMMsqrt.__init__(self, **kwargs)

    def rhs(self, omega):
        """
        Function that implements right-hand side of derivative for minimizing density matrix
        :param omega: local copy of sqrt of density matrix to avoid overwriting data by mistake
        :return:    rhs of derivative expression
        """
        # Update the value of H given omega, then update scaledH for calculation
        h = self.H + self.mf.get_veff(self.mf.mol, omega.conj().T @ omega)
        self.scaledH = -0.5 * (self.inv_ovlp @ h - self.mu * self.identity)

        # Calculate next value of omega
        temp = self.inv_sqrt_ovlp @ omega
        return omega @ (self.identity - (temp @ temp)) @ self.scaledH
        #k = (self.identity - self.inv_ovlp @ rho) @ self.scaledH
        #return rho @ k + k.conj().T @ rho

    def single_step_propagation(self, dbeta):
        """
        Function that propagates self.rho_next by a single step of size dbeta
        :param dbeta:   step size of inverse temperature
        :return:
        """
        omega = self.omega_next

        k1 = self.rhs(omega)

        k2 = self.rhs(omega + 0.5 * k1 * dbeta)

        k3 = self.rhs(omega + 0.5 * k2 * dbeta)

        k4 = self.rhs(omega + k3 * dbeta)

        omega += (1/6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)
        rho = omega.conj().T @ omega
        self.energy_next = np.trace(rho @ self.inv_ovlp @ self.H)
