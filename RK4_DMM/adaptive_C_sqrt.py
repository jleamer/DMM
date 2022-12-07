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
        self.num_electrons = num_electrons

        # Sqrt of overlap matrix
        self.sqrt_ovlp = linalg.sqrtm(self.ovlp)

        # Initialize mu at beta = 0
        self.mu = H.trace() / ovlp.trace()

        # Inverse overlap matrix
        self.inv_ovlp = linalg.inv(ovlp)

        # Inverse sqrt overlap matrix
        self.inv_sqrt_ovlp = linalg.sqrtm(self.inv_ovlp)

        # Store identity matrix
        self.identity = np.identity(H.shape[0])

        # Calculate coefficient: 2 * num_electrons / trace(ovlp)
        self.coeff = np.sqrt(2 * num_electrons / ovlp.trace())

        # Create initial density matrix
        if rho is None:
            self.rho = self.coeff ** 2 * ovlp / 2
        else:
            self.rho = rho

        if omega is None:
            self.omega = self.coeff * self.sqrt_ovlp / np.sqrt(2)
        else:
            self.omega = omega

        self.omega = self.omega.astype(complex)

        assert np.allclose(self.rho, self.omega.conj().T @ self.omega)

        # Call parent constructor on remanining kwargs
        CAdaptiveDMMsqrt.__init__(self, **kwargs)

    def rhs(self, omega):
        """
        Right hand side of the derivative expression for propagating rho
        :param omega: the sqrt of density matrix
        """
        # Need to update value of mu and dmu/dbeta
        # Numerator: a
        x = omega.conj().T @ self.inv_sqrt_ovlp @ omega.conj().T @ self.inv_sqrt_ovlp / self.coeff ** 2
        #y = self.inv_sqrt_ovlp @ omega @ self.inv_sqrt_ovlp @ omega / self.coeff ** 2
        a = self.H @ self.inv_ovlp @ (self.identity - x) @ omega.conj().T @ omega
        a += a.conj().T

        # Denominator: b
        b = (self.identity - x) @ omega.conj().T @ omega
        b += b.conj().T

        # Calculate dmu/dbeta
        if self.beta == 0:
            dmu = 0
        else:
            dmu = (a.trace() / b.trace() - self.mu) / self.beta

        scaledH = -0.5 * (self.inv_ovlp @ self.H - a.trace() / b.trace() * self.identity)

        domega = omega @ (self.identity - x.conj().T) @ scaledH
        return domega, dmu

    def single_step_propagation(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
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
        self.cv_next = -(self.beta + self.dbeta) ** 2 * np.trace(trace_arg @ self.inv_ovlp @ self.H)
        """
        self.rho_next = self.pos_pres_rhs(self.rho_next)
        """
        self.energy_next = np.trace(rho @ self.inv_ovlp @ self.H)

    def pos_pres_rhs(self, rho):
        """
        right-hand side of the derivative expression for propagating rho that preserves positivity
        P_n+1 = (1 + K_n) P_n (1+K_n)^\dag
        :param rho: the density matrix
        """
        # Calculate K_n

        n = self.num_electrons
        """
        # Calculate coefficient for mu sq term for preserving trace
        a = 2 * np.trace(rho @ (self.identity - rho @ self.inv_ovlp / n))

        # Calculate coefficient for mu term for preserving trace
        l_n = self.dbeta / 2 * rho @ self.inv_ovlp @ self.H
        j_n = self.dbeta * rho @ self.inv_ovlp @ rho @ self.inv_ovlp @ self.H / n

        b = 2 * rho @ (self.identity - rho @ self.inv_ovlp / n)
        b -= l_n
        b += j_n
        b -= l_n.conj().T
        b += j_n.conj().T
        b += self.dbeta / 2 * self.H @ self.inv_ovlp @ rho @ self.inv_ovlp @ rho @ self.inv_ovlp @ rho / n
        b = np.trace(b)

        # Calculate coefficient for mu^0 term
        m_n = (self.identity - self.inv_ovlp @ rho / n) @ self.H @ self.inv_ovlp
        o_n = (1 / 2 - self.inv_ovlp @ rho / n - self.inv_ovlp @ rho @ self.inv_ovlp @ rho / (2 * n))

        c = -rho @ m_n - m_n.conj().T @ rho
        c += self.dbeta * self.H @ self.inv_ovlp @ rho @ o_n @ self.inv_ovlp @ self.H
        c = np.trace(c)

        if self.beta == 0:
            self.dmu = 0
        else:
            self.dmu = (- b + np.sqrt(b ** 2 - 4 * a * c) / (2 * a) - self.mu) / self.beta

        """
        """
        l = self.identity - self.inv_ovlp @ rho / self.coeff
        m = l @ self.inv_ovlp @ self.H

        a = (l.conj().T @ rho @ l).trace().real * self.dbeta / 2
        b = ((- self.dbeta * m.conj().T + 2 * self.identity) @ rho @ l).trace().real
        c = ((-2 * self.identity + self.dbeta / 2 * m.conj().T) @ rho @ m).trace().real

        self.mu = (- b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        print(b ** 2 - 4 * a * c)
        K_n = self.dbeta / 2 * (m - self.mu * l)
        """
        """
        a = rho @ (self.identity - self.inv_ovlp @ rho / self.coeff) @ self.inv_ovlp @ self.H
        a += a.conj().T

        b = rho @ (self.identity - self.inv_ovlp @ rho / self.coeff)
        b += b.conj().T
        """
        # Define alpha and gamma - matrices that define K
        alpha = (self.identity - self.inv_ovlp @ rho / self.coeff)
        gamma = alpha @ self.inv_ovlp @ self.H

        # Define coefficients for quadratic formula to calculate mu
        A = self.dbeta / 2 * (self.rho @ alpha @ alpha.conj().T).trace()
        x = alpha @ gamma.conj().T
        B = (self.rho @ (alpha.conj().T + alpha - self.dbeta / 2 * (x + x.conj().T))).trace()
        C = (self.rho @ (gamma @ gamma.conj().T - gamma.conj().T - gamma)).trace()

        # Calculat mu using quadratic formula
        self.mu = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        k = gamma - alpha * self.mu
        K = self.identity - self.dbeta / 2 * k
        return K.conj().T @ rho @ K
        # Calculate K_n
        #K_n = -self.dbeta / 2 * (self.H @ self.inv_ovlp - self.mu * self.identity) @ (self.identity - rho @ self.inv_ovlp / self.coeff)

        #return (self.identity + K_n) @ rho @ (self.identity + K_n).conj().T