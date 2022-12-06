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
        # num_electrons list
        self.num_electron_list = [self.rho.trace()]

        self.rho_list = [self.rho]
        self.omega_list = [self.omega]

        # call the parent's constructor
        CAdaptiveDMMsqrt.__init__(self, **kwargs)

        self.omega_next = np.empty_like(self.omega)

    def rhs(self, q):
        """
        Method implements the rhs of the derivative expression for minimizing rho
        """
        #k = (self.identity - q @ q) @ self.scaledH
        #return q @ k + k.conj().T @ q.conj().T
        temp = self.inv_sqrt_ovlp @ q
        return q @ (self.identity - (temp @ temp)) @ self.scaledH

    def single_step_propagation(self, dbeta):
        """
        Propagate self.rho_next by a single step using RK4
        :param dbeta: a step size in inverse temperature
        :return: None
        """

        # RK4 approach
        # alias
        omega = self.omega_next

        k1 = self.rhs(omega)

        k2 = self.rhs(omega + 0.5 * dbeta * k1)

        k3 = self.rhs(omega + 0.5 * dbeta * k2)

        k4 = self.rhs(omega + dbeta * k3)

        omega += (1 / 6) * dbeta * (k1 + 2 * k2 + 2 * k3 + k4)
        rho = omega.conj().T @ omega
        self.energy_next = np.trace(rho @ self.inv_ovlp @ self.H)
        """
        self.rho_next = self.pos_pres_rhs(self.rho_next)
        self.energy_next = np.trace(self.rho_next @ self.inv_ovlp @ self.H)
        """