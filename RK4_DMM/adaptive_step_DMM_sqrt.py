import numpy as np
from numba import njit
from scipy import linalg, sparse


########################################################################################################################
#
#   Abstract base class for adaptive step method using sqrt(P)
#
########################################################################################################################
@njit
def relative_diff(psi_next, psi):
    """
    Efficiently calculate the relative difference of two arrays. (Used in thea adaptive scheme)
    :param psi_next: numpy.array
    :param psi: numpy.array
    :return: float
    """
    #return np.linalg.norm(psi_next - psi, ord=np.inf) / np.linalg.norm(psi_next, ord=np.inf)
    return np.abs(psi_next - psi).max()
    #return np.linalg.norm(psi_next - psi)
    #return np.abs(np.trace(psi_next) - np.trace(psi))


class CAdaptiveDMMsqrt():
    """
    Abstract base class for adaptive spet-method implementation for DMM.

    Child classes must implement tge method single_step_propagation(self, dbeta),
    where a method for the single step propagation is implemented.
    """
    def __init__(self, *, dbeta, beta=0, epsilon=1e-3):
        """
        :param dbeta:    initial step size of inverse temperature
        :param beta:     initial value of inverse temperature
        :param epsilon:  error tolerance
        """
        # Store variables passed to the constructor
        self.dbeta = dbeta
        self.beta = beta
        self.epsilon = epsilon

        # the relative change estimators for the time adaptive scheme
        self.e_n = self.e_n_1 = self.e_n_2 = 0

        self.previous_dbeta = 0

        # list of self.dbeta to monitor how the adaptive step method is working
        self.beta_increments = []

        # copy of the density matrix sqrt
        self.omega_next = np.empty_like(self.omega)

        # Store variables to be tracked as system propagates
        self.energy = np.trace(self.rho @ self.inv_ovlp @ self.H)
        self.energy_vals = [self.energy, ]
        self.eigenvalues = [np.linalg.eigvalsh(self.rho)]
        self.num_electron_list = [self.num_electrons]
        self.mu_list = [self.mu]
        self.dmu = 0
        self.cv = []
        self.cv_next = 0

    def propagate(self, beta_final):
        """
        Inverse temperature propagation of the density matrix saved in self.rho
        :param beta_final: until what beta to propagate the rho
        :return: self.rho
        """
        e_n = self.e_n
        e_n_1 = self.e_n_1
        e_n_2 = self.e_n_2
        previous_dbeta = self.previous_dbeta

        # copy the initial condition into the propagation array self.rho_next
        np.copyto(self.omega_next, self.omega)

        while self.beta < beta_final:

            ############################################################################################################
            #
            #   Adaptive scheme propagator
            #
            ############################################################################################################

            # propagate the rho by a single dbeta
            self.single_step_propagation(self.dbeta)
            #temp = self.q_next.conj().T @ self.q_next
            #temp2 = self.q.conj().T @ self.q
            #e_n = relative_diff(temp, temp2)
            e_n = relative_diff(self.omega_next, self.omega)

            while e_n > self.epsilon:
                # the error is too high, decrease the time step and propagate with the new step

                self.dbeta *= self.epsilon / e_n

                np.copyto(self.omega_next, self.omega)
                self.single_step_propagation(self.dbeta)
                #temp = self.q_next.conj().T @ self.q_next
                #temp2 = self.q.conj().T @ self.q
                #e_n = relative_diff(temp, temp2)
                #print(e_n)
                e_n = relative_diff(self.omega_next, self.omega)

            # if the energy is increasing stop and only keep last iteration
            if self.energy_next > self.energy:
                print("Energy_next > energy")
                break
            print("=========================")

            # Accept new value of omega and update tracked variables
            np.copyto(self.omega, self.omega_next)
            self.rho = self.omega.conj().T @ self.omega
            self.energy = self.energy_next
            self.energy_vals.append(self.energy)
            #self.eigenvalues.append(np.linalg.eigvalsh(self.rho))
            self.num_electron_list.append(self.rho.trace())
            self.mu += self.dmu
            self.mu_list.append(self.mu)
            self.cv.append(self.cv_next)

            # save self.dbeta for monitoring purpose
            self.beta_increments.append(self.dbeta)

            # increment time
            self.beta += self.dbeta


            ############################################################################################################
            #
            #   Update step via the Evolutionary PID controller
            #
            ############################################################################################################

            # overwrite the zero values of e_n_1 and e_n_2
            previous_dbeta = (previous_dbeta if previous_dbeta else self.dbeta)
            e_n_1 = (e_n_1 if e_n_1 else e_n)
            e_n_2 = (e_n_2 if e_n_2 else e_n)

            # the adaptive time stepping method from
            #   http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture8.pdf
            #self.dbeta *= (e_n_1 / e_n) ** 0.075 * (self.epsilon / e_n) ** 0.175 * (e_n_1 ** 2 / e_n / e_n_2) ** 0.01

            # the adaptive time stepping method from
            #   https://linkinghub.elsevier.com/retrieve/pii/S0377042705001123
            self.dbeta *= (self.epsilon ** 2 / e_n / e_n_1 * previous_dbeta / self.dbeta) ** (1 / 12.)

            # update the error estimates in order to go next to the next step
            e_n_2, e_n_1 = e_n_1, e_n

            if e_n > self.epsilon:
                print("e_n > eps")
                break

        # save the error estimates
        self.previous_dbeta = previous_dbeta
        self.e_n = e_n
        self.e_n_1 = e_n_1
        self.e_n_2 = e_n_2

        return self.omega