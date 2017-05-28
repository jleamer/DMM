import numpy as np
from scipy import linalg, sparse
from types import MethodType, FunctionType


class DMM:
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
        self.identity = sparse.identity(self.H.shape[0], dtype=self.H.dtype)

        # First step: symmetrized Hamiltonian
        self.H += self.H.conj().T
        self.H *= 0.5

        # find eigenvalues of the Hamiltonian for comparision with the exact expression self.get_exact_pop
        self.E = linalg.eigvalsh(
            # Convert a sparse Hamiltonian to a dense matrix
            self.H.toarray() if sparse.issparse(self.H) else self.H
        )

        # Set the initial condition for the matrix
        self.rho = 0.5 * self.identity.toarray()

    def propagate_beta(self, nsteps=1, commnorm=False):
        """
        :param nsteps: number of steps in the inverse temperature to take
        :param commnorm: boolean flag to print the norm of the commutator between the density matrix and hamiltonian
        :return: self
        """
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * self.identity
        scaledH *= -0.5 * self.dbeta

        # Allocate the memory for
        tmp = np.empty_like(self.rho)

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

            # Construct the map K
            # Optimized version of
            #   K = self.identity - 0.5 * self.dbeta * self.H.dot(self.identity - self.rho)

            K = scaledH.dot(self.identity - self.rho)
            K += self.identity

            # assert not sparse.issparse(K), "K matrix must not be sparse"

            # Optimized version of
            self.rho = K.dot(self.rho).dot(K.conj().T)

            # np.dot(K, self.rho, out=tmp)
            # np.dot(tmp, K.conj().T, out=self.rho)

            self.beta += self.dbeta

        if commnorm:
            # Convert a sparce Hamiltonian to a dense matrix
            H = (self.H.toarray() if sparse.issparse(self.H) else self.H)

            print(
                "\nThe norm of the commutator of the obtained density matrix and the Hamiltonian: %.2e\n\n"
                % linalg.norm(self.rho.dot(H) - H.dot(self.rho))
            )

        return self

    def propagate_mu(self, nsteps=1):
        """
        Propagate along the chemical potential
        :param nsteps: number of steps to be taken in the chemical potential   
        :return: self
        """
        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       rho(mu + dmu) = K rho(mu) K
            #
            #   where
            #
            #       K = 1 + 0.5 * dmu * beta * (1 - rho(mu))
            #
            ###########################################################################

            # Construct K
            K = self.identity - self.rho
            K *= 0.5 * self.dmu * self.beta
            K += self.identity

            self.rho = K.dot(self.rho).dot(K)

            self.mu += self.dmu

        return self

    def get_exact_DF(self):
        """
        :return: The exact Fermi-Dirac density matrix 
        """
        # Convert a sparce Hamiltonian to a dense matrix
        H = (self.H.toarray() if sparse.issparse(self.H) else self.H.copy())
        H -= self.mu * self.identity
        H *= self.beta

        self.rho_exact = linalg.inv(self.identity + linalg.expm(H))

        return self.rho_exact

    def get_exact_pop(self):
        """
        :return: The exact Fermi-Dirac population distributions
        """
        return 1. / (1. + np.exp(self.beta * (self.E - self.mu)))

    def get_average_H(self):
        """
        :return: Tr(H \rho) / Tr(\rho)
        """
        #########################################################
        #
        #   The current implementation is not the most optimal.
        #   A better method to use is
        #
        #       Tr(AB) = A_{ij} B_{ji} = A_{ij} (B^T)_{ij}
        #
        #########################################################
        #return np.sum(self.H * self.rho.T) / self.rho.trace().sum()
        return self.H.dot(self.rho).trace().sum() / self.rho.trace().sum()

    def get_exact_average_H(self):
        """
        :return: Exact expression Tr(H \rho)  
        """
        p = self.get_exact_pop()
        return np.sum(p * self.E) / p.sum()

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dmm = DMM(
        dbeta=0.003,
        dmu=0.0005,
        mu=-0.9,
        # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
        #H=sparse.random(70, 70, density=0.1),
        H=np.random.normal(size=(40,40)) + 1j * np.random.normal(size=(40,40)),
        #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
    )

    # First propagate along the inverse temperature
    dmm.propagate_beta(3000)

    plt.subplot(121)
    plt.title("Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm.beta, dmm.mu))

    p_dmm = linalg.eigvalsh(dmm.rho)[::-1]

    plt.plot(dmm.E, p_dmm, '*-', label="DMM")
    plt.plot(dmm.E, linalg.eigvalsh(dmm.get_exact_DF())[::-1], '*-', label='exact via expm')
    plt.plot(dmm.E, dmm.get_exact_pop(), '*-', label='population')
    #plt.plot(dmm.E, p_dmm - dmm.get_exact_pop())

    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Population')

    # Second propagate along the chemical potential
    dmm.propagate_mu(3000)

    plt.subplot(122)
    plt.title("Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm.beta, dmm.mu))

    p_dmm = linalg.eigvalsh(dmm.rho)[::-1]

    plt.plot(dmm.E, p_dmm, '*-', label="DMM")
    plt.plot(dmm.E, linalg.eigvalsh(dmm.get_exact_DF())[::-1], '*-', label='exact via expm')
    plt.plot(dmm.E, dmm.get_exact_pop(), '*-', label='population')
    #plt.plot(dmm.E, p_dmm - dmm.get_exact_pop())

    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Population')

    plt.show()