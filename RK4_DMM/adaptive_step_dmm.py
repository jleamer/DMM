import numpy as np
from numba import njit
from scipy import linalg, sparse


########################################################################################################################
#
#   Abstract base class for adaptive step method
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


class CAdaptiveDMM(object):
    """
    Abstract base class for adaptive spet-method implementation for DMM.

    Child classes must implement tge method single_step_propagation(self, dbeta),
    where a method for the single step propagation is implemented.
    """

    def __init__(self, *, dbeta, beta=0, epsilon=1e-3):
        """
        :param dbeta: Initial inverse temperature step
        :param epsilon: relative error tolerance
        """
        self.beta = beta
        self.dbeta = dbeta
        self.epsilon = epsilon

        # the relative change estimators for the time adaptive scheme
        self.e_n = self.e_n_1 = self.e_n_2 = 0

        self.previous_dbeta = 0

        # list of self.dbeta to monitor how the adaptive step method is working
        self.beta_increments = []

        # copy of the density matrix
        self.rho_next = np.empty_like(self.rho)
        
        # energy
        self.energy = np.trace(self.rho @ self.inv_ovlp @ self.H)
        self.energy_vals = [self.energy,]
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
        np.copyto(self.rho_next, self.rho)

        while self.beta < beta_final:

            ############################################################################################################
            #
            #   Adaptive scheme propagator
            #
            ############################################################################################################

            # propagate the rho by a single dbeta
            self.single_step_propagation(self.dbeta)

            e_n = relative_diff(self.rho_next, self.rho)
            #e_n = abs(self.energy_next / self.energy - 1)

            while e_n > self.epsilon:
                # the error is to high, decrease the time step and propagate with the new step

                self.dbeta *= self.epsilon / e_n

                np.copyto(self.rho_next, self.rho)
                self.single_step_propagation(self.dbeta)

                e_n = relative_diff(self.rho_next, self.rho)
                #e_n = abs(self.energy_next / self.energy - 1)
                #print(e_n)

            # accept the current density matrix
            #if self.energy_next > self.energy:
            #    print("Energy_next > energy")
            #    break
            #print("=========================")

            np.copyto(self.rho, self.rho_next)
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

        return self.rho


########################################################################################################################
#
#   Specific DMM method
#
########################################################################################################################


class CAdaptive_GCP_RK4(CAdaptiveDMM):
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


########################################################################################################################
#
#   Test using the Huckel model (Using Jacob's code)
#
########################################################################################################################


if __name__ == '__main__':
    from scipy import sparse
    import matplotlib.pyplot as plt

    # First generate a random hamiltonian of the Huckel model
    # i.e. alpha on the diagonal, and gamma on the off-diagonals to represent nearest neighbors
    #    for now use random alpha and gamma
    np.random.seed(75)
    alpha = np.random.random()
    gamma = np.random.random()

    # define the dimensions of the Hamiltonian and how many elements to consider before cutoff
    size = 50


    def huckel_hamiltonian(alpha, gamma, size):
        H = sparse.diags([gamma, alpha, gamma], [-1, 0, 1], shape=(size, size)).toarray()
        H[0][size - 1] = gamma
        H[size - 1][0] = gamma
        return H


    H = huckel_hamiltonian(alpha, gamma, size)

    # define a chemical potential mu
    mu = 0.45

    # Fermi-Dirac - this simulates finite temperature
    beta = 300
    ferm_exact = linalg.funm(H, lambda _: np.exp(-beta * (_ - mu)) / (1 + np.exp(-beta * (_ - mu))))

    numsteps = 10000
    dbeta = beta / numsteps
    ovlp = np.identity(H.shape[0])

    # GCP propagation
    gcp = CAdaptive_GCP_RK4(ovlp=ovlp, H=H, mu=mu, dbeta=dbeta, epsilon=1e-1)

    gcp.propagate(beta)

    plt.title('Populations')
    plt.plot(linalg.eigvalsh(gcp.rho), '*-', label='GCP')
    plt.plot(linalg.eigvalsh(ferm_exact), '*-', label='FD')
    plt.legend()
    plt.show()

    plt.title("Variable step method in action")
    plt.plot(gcp.beta_increments, '*-')
    plt.xlabel('steps')
    plt.ylabel('dbeta')
    plt.show()
