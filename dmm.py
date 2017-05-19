import numpy as np
from scipy import linalg
from types import MethodType, FunctionType


class DMMK:
    """
    Implementation of the Density Matrix Minimization (DMM) Method
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            H -- the Hamiltonian of the system
            beta (optional) -- the initial value of the inverse temperature
            dbeta -- the inverse temperature increment
            mu (optional) -- the chemical potential, default is zero
        """

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.H
        except ArithmeticError:
            raise ArithmeticError("The Hamiltonian (H) was not specified")

        try:
            self.dbeta
        except ArithmeticError:
            raise ArithmeticError("The inverse temperature increment (dbeta) was not specified")

        try:
            self.mu
        except AttributeError:
            print("Warning: Chemical potential was not specified, thus it is set to zero.")
            self.mu = 0

        try:
            self.beta
        except AttributeError:
            self.beta = 0.

        # save the identity matrix
        self.identity = np.identity(self.H.shape[0], dtype=self.H.dtype)

        # First step: symmetrized Hamiltonian
        self.H += self.H.conj().T
        self.H *= 0.5

        # substruct the chemical energy
        self.H -= self.mu * self.identity

        # find eigenvalues of the Hamiltonian for comparision with the exact expression self.get_exact_pop
        self.E = linalg.eigvalsh(self.H)

        # Set the initial condition for the matrix
        self.rho = 0.5 * self.identity

    def propagate(self, nsteps=1):
        """
        :param nsteps: number of steps in the inverse temperature to take
        :return: self.rho (the density matrix) 
        """
        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       rho(beta + dbeta) = K rho(beta) K^{\dagger}
            #       rho(beta = 0) = 1/2
            #
            #   where
            #
            #       K = 1 - 0.5 * dbeta * H * (1 - rho(beta))
            #
            ###########################################################################

            # Construct the maps
            K = self.identity - 0.5 * self.dbeta * self.H.dot(self.identity - self.rho)

            self.rho = K.dot(self.rho).dot(K.conj().T)

            self.beta += self.dbeta

        print(
            "\nThe norm of the commutator of the obtained density matrix and the Hamiltonian: %.2e\n\n"
            % linalg.norm(
                self.rho.dot(self.H) - self.H.dot(self.rho)
            )
        )

        return self.rho

    def get_exact_DF(self):
        """
        :return: The exact Fermi-Dirac density matrix 
        """
        self.rho_exact = linalg.inv(self.identity + linalg.expm(self.beta * self.H))
        return self.rho_exact

    def get_exact_pop(self):
        """
        :return: The exact Fermi-Dirac population distributions
        """
        return 1. / (1. + np.exp(self.beta * self.E))

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dmm = DMMK(
        dbeta=0.003,
        mu=-0.9,
        # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
        H=np.random.normal(size=(40,40)) + 1j * np.random.normal(size=(40,40))
        #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
    )

    dmm.propagate(30000)

    plt.title("Population")

    p_dmm = linalg.eigvalsh(dmm.rho)[::-1]

    plt.plot(dmm.E, p_dmm, '*-', label="DMM")
    # plt.plot(dmm.E, linalg.eigvalsh(dmm.get_exact_DF())[::-1], '*-', label='exact via expm')
    plt.plot(dmm.E, dmm.get_exact_pop(), '*-', label='population')

    plt.legend()

    plt.show()