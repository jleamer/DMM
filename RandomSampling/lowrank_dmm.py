import numpy as np
from scipy import linalg, sparse
from types import MethodType, FunctionType


class LowRankDMM:
    """
    Implementation of the Density Matrix Minimization (DMM) Method where the low-rank representation of the density
    matrix is used. The density matrix rho (N \times N) is represented as

        rho = R R^{\dagger},

    where R is an N \times M matrix and the Hamiltonian H is an N \times N matrix.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            H -- the Hamiltonian of the system (N \times N)
            M -- the rank of R matrix (M < N)
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
        except AttributeError:
            raise AttributeError("The Hamiltonian (H) was not specified")

        try:
            self.M
        except AttributeError:
            raise AttributeError("The rank (M) of the low-rank approximation of the density matrix R was not specified")

        try:
            self.dbeta
        except AttributeError:
            raise AttributeError("The inverse temperature increment (dbeta) was not specified")

        try:
            self.mu
        except AttributeError:
            print("Warning: Chemical potential was not specified, thus it is set to zero.")
            self.mu = 0

        try:
            self.beta
        except AttributeError:
            self.beta = 0.

        # First step: symmetrized Hamiltonian
        self.H += self.H.conj().T
        self.H *= 0.5

        # Set the initial condition for the matrix R
        # recall that the density matrix rho = R R^{\dagger}
        self.R = np.eye(self.H.shape[0], self.M, dtype=self.H.dtype)
        self.R *= np.sqrt(0.5)

        # save the identity matrix
        self.identity = sparse.identity(self.M, dtype=self.R.dtype)

        # find eigenvalues of the Hamiltonian for comparision with the exact expression self.get_exact_pop
        self.E = linalg.eigvalsh(
            # Convert a sparse Hamiltonian to a dense matrix
            self.H.toarray() if sparse.issparse(self.H) else self.H
        )

    def propagate_beta1(self, nsteps=1, commnorm=False):
        """
        The first order propagation in the inverse temperature
        :param nsteps: number of steps in the inverse temperature to take
        :param commnorm: boolean flag to print the norm of the commutator between the density matrix and hamiltonian
        :return: self
        """
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * sparse.identity(self.H.shape[0], dtype=self.H.dtype)
        scaledH *= -0.5 * self.dbeta

        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       R(beta + dbeta) = K R(beta)
            #                       = R(beta) - 0.5 * dbeta * (H - mu) R(beta) (1 - R(beta)^{\dagger} R(beta))
            #       R(beta = 0) = 1 / sqrt(2)
            #
            #   where K is from the first oder (dense) DMM propagation
            #
            #       K = 1 - 0.5 * dbeta * (H - mu) * (1 - R(beta)R(beta)^{\dagger})
            #
            ###########################################################################

            self.R += scaledH.dot(self.R).dot(self.identity - self.R.T.conj().dot(self.R))

            self.beta += self.dbeta

        return self

    def get_exact_pop(self):
        """
        :return: The exact Fermi-Dirac population distributions
        """
        return 1. / (1. + np.exp(self.beta * (self.E - self.mu)))

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    from QuantumClassicalDynamics.central_diff_qhamiltonian import CentralDiffQHamiltonian
    import matplotlib.pyplot as plt

    # Generate the hamiltonian
    qsys = CentralDiffQHamiltonian(
        X_gridDIM=300,
        X_amplitude=7.,
        K="0.5 * P ** 2",
        V="15. * (exp(-X / 2.) -2. * exp(-X / 4.))",
    )

    # initialize the class for density matrix minimization
    dmm = LowRankDMM(
        mu=1.05 * qsys.get_energy(2),
        dbeta=0.001,
        M=200,
        H=qsys.Hamiltonian,
    )
    dmm.propagate_beta1(5000)

    plt.subplot(121)
    plt.title("Population")

    # Populations for the low-rank representation can be found via SVD
    p_dmm = linalg.svd(dmm.R, compute_uv=False) ** 2
    # p_dmm = linalg.eigvalsh(dmm.R.dot(dmm.R.conj().T))[::-1]

    plt.plot(dmm.E[:p_dmm.size], p_dmm, '*-', label="DMM")
    plt.plot(dmm.E, dmm.get_exact_pop(), '*-', label='TF')
    plt.xlim([dmm.E[0], dmm.E[20]])
    plt.xlabel('Energy')
    plt.legend()

    plt.show()

