import numpy as np
from numba import njit
from scipy import sparse, linalg
from adaptive_step_DMM_sqrt import CAdaptiveDMMsqrt
from pyscf import gto, dft
import numpy.ma as ma
import time


class CAdaptive_C_RK4_sqrt(CAdaptiveDMMsqrt):
    """
    DMM CP via RK4
    """
    def __init__(self, *, ovlp, H, num_electrons, rho=None, omega=None, **kwargs):
        """
        initializing function for class
        :param ovlp:            overlap matrix
        :param H:               hamiltonian
        :param init_mu:         initial chemical potential
        :param num_electrons:   number of electrons in system
        """
        assert(ovlp.shape == H.shape)

        # Save arguments
        self.ovlp = ovlp
        self.H = H

        # Sqrt of overlap matrix
        self.sqrt_ovlp = linalg.sqrtm(self.ovlp)

        # Inverse overlap matrix
        self.inv_ovlp = linalg.inv(ovlp)

        # Inverse sqrt overlap matrix
        self.inv_sqrt_ovlp = linalg.sqrtm(self.inv_ovlp)

        # Store identity matrix
        self.identity = np.identity(H.shape[0])

        # Calculate coefficient: 2 * num_electrons / trace(ovlp)
        self.coeff = np.sqrt(num_electrons / self.identity.trace())

        # We use inv_S @ H so often, we'll just define it here
        self.A = self.inv_ovlp @ self.H

        # Create initial density matrix
        if rho is None:
            self.rho = self.coeff ** 2 * self.ovlp / 2
        else:
            self.rho = rho

        if omega is None:
            self.omega = self.coeff * self.sqrt_ovlp / np.sqrt(2)
        else:
            self.omega = omega

        # Initialize mu at beta = 0
        #self.mu = H.trace() / ovlp.trace()
        self.mu = np.trace(self.inv_ovlp @ self.H) / np.trace(self.identity)

        self.num_electrons = 2*np.trace(self.ovlp @ self.rho)
        self.omega = self.omega.astype(complex)

        assert np.allclose(self.rho, self.omega.conj().T @ self.omega)

        # Call parent constructor on remanining kwargs
        CAdaptiveDMMsqrt.__init__(self, **kwargs)

    def rhs(self, beta, omega, mu):
        """
        Right hand side of the derivative expression for propagating rho
        :param omega: the sqrt of density matrix
        """
        # 8 matrix multiplications by my count

        # Need to update value of mu and dmu/dbeta
        # Numerator: a
        B = self.inv_sqrt_ovlp @ omega
        x = omega @ (self.identity - B @ B)
        C = x @ self.A
        D = C @ self.inv_ovlp
        E = x @ self.inv_ovlp

        #num = np.sum(D.conj() * omega) + np.sum(omega.conj() * D)
        num = np.trace(omega.conj().T @ D + D.conj().T @ omega)
        #den = np.sum(E.conj() * omega) + np.sum(omega.conj() * E)
        den = np.trace(omega.conj().T @ E + E.conj().T @ omega)

        # Calculate dmu/dbeta
        if beta == 0:
            dmu = 0
        else:
            dmu = 0*(num / den - mu) / (beta)

        domega = -0.5 * (C - x * num / den)

        #indx = np.abs(domega) < self.tol
        #domega[indx] = 0

        #self.sparsity.append(np.sum(indx) / self.H.shape[0] ** 2)
        self.count += 1
        return domega, dmu

    def single_step_rk2(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """
        # alias
        omega = self.omega_next
        mu = self.mu_next

        # RK2
        k1, l1 = self.rhs(self.beta, omega, mu)
        k2, l2 = self.rhs(self.beta + 0.5 * dbeta, omega + 0.5 * dbeta * k1, mu + 0.5 * dbeta * l1)

        omega += dbeta * k2
        self.mu_next += dbeta * l2

        # Update rho and the energy
        rho = omega.conj().T @ omega
        self.energy_next = (self.inv_ovlp @ self.H @ self.inv_ovlp @ rho.conj()).trace()

        temp = k1
        trace_arg = temp.conj().T @ self.omega
        trace_arg += trace_arg.conj().T
        self.cv_next = -2*(self.beta + self.dbeta) ** 2 * np.trace(self.inv_ovlp @ trace_arg @ self.A)

    def single_step_rk4(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """
        # alias
        omega = self.omega_next
        mu = self.mu_next

        k1, l1 = self.rhs(self.beta, omega, mu)

        k2, l2 = self.rhs(self.beta + 0.5 * dbeta, omega + 0.5 * dbeta * k1, mu + 0.5 * dbeta * l1)

        k3, l3 = self.rhs(self.beta + 0.5 * dbeta, omega + 0.5 * dbeta * k2, mu + 0.5 * dbeta * l2)

        k4, l4 = self.rhs(self.beta + dbeta, omega + dbeta * k3, mu + dbeta * l3)

        omega += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)
        self.mu_next += (1 / 6) * dbeta * (l1 + 2 * l2 + 2 * l3 + l4)

        # Update rho and the energy
        rho = omega.conj().T @ omega
        self.energy_next = np.sum(rho.conj() * self.A)
        temp = k1
        trace_arg = temp.conj().T @ self.omega
        trace_arg += trace_arg.conj().T
        self.cv_next = -(self.beta+self.dbeta) ** 2 * np.sum(trace_arg.conj() * self.A)

    def single_step_predictor_corrector(self, dbeta):
        """
        Propagate self.omega_next by a single step using predictor corrector method
        :param dbeta: a step size of inverse temperature
        :return:
        """
        # alias
        omega = self.omega_next
        mu = self.mu

        # Calculate first step using euler approximation
        k1, l1 = self.rhs(omega)
        eta = omega.copy() + dbeta * k1

        # Calculate next part
        k2, l2 = self.rhs(eta)

        # Update omega
        omega += dbeta / 2 * (k1 + k2)
        self.mu += dbeta / 2 * (l1 + l2)

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

        # alias
        omega = self.omega_next
        mu = self.mu
        """
        k1, l1 = self.rhs(omega)

        k2, l2 = self.rhs(omega + 0.5 * dbeta * k1)

        k3, l3 = self.rhs(omega + 0.5 * dbeta * k2)

        k4, l4 = self.rhs(omega + dbeta * k3)

        omega += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)
        self.mu += (1 / 6) * dbeta * (l1 + 2 * l2 + 2 * l3 + l4)

        """
        k1, l1 = self.rhs(omega)

        k2, l2 = self.rhs(omega + 0.5 * dbeta * k1)

        omega += dbeta * k2
        indx = np.abs(omega) < self.tol
        omega[indx] = 0
        self.mu += dbeta * l2
        #"""
        rho = omega.conj().T @ omega
        temp = self.rhs(omega)[0]
        trace_arg = temp.conj().T @ self.omega
        trace_arg += trace_arg.conj().T
        self.cv_next = -(self.beta + self.dbeta) ** 2 * np.trace(trace_arg @ self.inv_ovlp @ self.H)
        #self.cv_next = -(self.beta + self.dbeta) ** 2 * np.trace(rho @ self.inv_ovlp @ self.H)


        self.energy_next = np.trace(rho @ self.inv_ovlp @ self.H)


class CAdaptive_C_RK4_NL_sqrt(CAdaptiveDMMsqrt):
    """
    DMM GC with adaptive step approach using RK4 method for systems where
        H = H_core + H_exc(P)
    """
    def __init__(self, *, ovlp, H, num_electrons, mf, rho=None, omega=None, **kwargs):
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
        self.num_electrons = num_electrons
        self.mf = mf

        # the inverse overlap matrix
        self.mu = H.trace() / ovlp.trace()
        self.sqrt_ovlp = linalg.sqrtm(ovlp)
        self.inv_ovlp = linalg.inv(ovlp)
        self.inv_sqrt_ovlp = linalg.sqrtm(self.inv_ovlp)
        self.identity = np.identity(self.H.shape[0])
        self.scaledH = -0.5 * (self.inv_ovlp @ self.H - self.mu * self.identity)
        self.coeff = np.sqrt(2 * num_electrons / ovlp.trace())

        if rho is None:
            self.rho = self.coeff ** 2 * ovlp / 2
        else:
            self.rho = rho

        if omega is None:
            self.omega = self.coeff * self.sqrt_ovlp / np.sqrt(2)
        else:
            self.omega = omega

        self.omega = self.omega.astype(complex)

        assert np.allclose(self.omega.conj().T @ self.omega, self.rho)

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

        # Need to update value of mu and dmu/dbeta
        # Numerator: a
        x = omega.conj().T @ self.inv_sqrt_ovlp @ omega.conj().T @ self.inv_sqrt_ovlp / self.coeff ** 2
        a = h @ self.inv_ovlp @ (self.identity - x) @ omega.conj().T @ omega
        a += a.conj().T

        # Denominator: b
        b = (self.identity - x) @ omega.conj().T @ omega
        b += b.conj().T

        # Calculate dmu/dbeta
        if self.beta == 0:
            dmu = 0
        else:
            dmu = (a.trace() / b.trace() - self.mu) / self.beta

        self.scaledH = -0.5 * (self.inv_ovlp @ h - a.trace() / b.trace() * self.identity)

        domega = omega @ (self.identity - x.conj().T) @ self.scaledH
        return domega, dmu

    def single_step_propagation(self, dbeta):
        """
        Function that propagates self.rho_next by a single step of size dbeta
        :param dbeta:   step size of inverse temperature
        :return:
        """
        # alias
        omega = self.omega_next
        mu = self.mu

        k1, l1 = self.rhs(omega)

        k2, l2 = self.rhs(omega + 0.5 * dbeta * k1)

        k3, l3 = self.rhs(omega + 0.5 * dbeta * k2)

        k4, l4 = self.rhs(omega + dbeta * k3)

        omega += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)
        self.mu += (1 / 6) * dbeta * (l1 + 2 * l2 + 2 * l3 + l4)
        rho = omega.conj().T @ omega
        temp = self.rhs(omega)[0]
        trace_arg = temp.conj().T @ self.omega
        trace_arg += trace_arg.conj().T
        #self.cv_next = -(self.beta + self.dbeta) ** 2 * np.trace(trace_arg @ self.inv_ovlp @ self.H)
        self.cv_next = -(self.beta + self.dbeta) ** 2 * np.trace(rho @ self.inv_ovlp @ self.H)
        self.energy_next = np.trace(rho @ self.inv_ovlp @ self.H)