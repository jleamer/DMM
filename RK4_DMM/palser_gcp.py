import numpy as np
from numba import njit
from scipy import sparse
from scipy.io import mmwrite, mmread
from numba import njit
from scipy import linalg
import matplotlib.pyplot as plt
import sys


def palser_gcp(mu, H, nsteps):
    """
    Function to evaluate the GCP method proposed in https://journals.aps.org/prb/pdf/10.1103/PhysRevB.58.12704 by Palser
    :param mu:      the chemical potential of the system
    :param H:       the Hamiltonian of the system
    :param nsteps:  the number of steps to run
    :return:        the density matrix, rho
    """
    identity = np.identity(H.shape[0], dtype=complex)
    size = identity.shape[0]

    # Estimate H_max and H_min (max and min spectrum of H) using Greshgorin's algorithm
    min_Hi = []
    max_Hi = []
    for i in range(size):
        min_sum = 0
        max_sum = 0
        for j in range(size):
            if i != j:
                min_sum -= np.abs(H[i][j])
                max_sum += np.abs(H[i][j])
            else:
                min_sum += H[i][j]
                max_sum += H[i][j]
        min_Hi.append(min_sum)
        max_Hi.append(max_sum)
    min_H = min(min_Hi)
    max_H = max(max_Hi)

    # Choose appropriate value for lambda and calculate rho
    lamb = min([1 / (max_H - mu), 1 / (mu - min_H)])
    rho = lamb / 2 * (mu * identity - H) + 0.5 * identity
    omega = np.sum(rho * (H - mu * identity).T)

    steps = 0
    while True:
        prev_omega = omega
        rho_sq = rho.dot(rho)
        rho_cu = rho.dot(rho_sq)
        rho = 3 * rho_sq - 2 * rho_cu
        omega = np.sum(rho * (H - mu * identity).T)
        steps += 1

        if omega >= prev_omega:
            break

    print("GCP steps: ", str(steps))
    return rho


if __name__ == '__main__':
    """
    Simple calculation using tight-binding model
    """
    np.random.seed(5)
    size = 10
    alpha = np.random.random()
    beta = np.random.random()

    off_diags = np.full(size-1, beta)
    diagonal = np.full(size, alpha)
    H = sparse.diags([off_diags, diagonal, off_diags],offsets=[-1,0,1])
    mu = .05
    rho = palser_gcp(mu, H.toarray(), 1000)

    true_rho = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range(size):
            true_rho[i][j] = 0.5*np.sinc(np.pi/2 * (i-j))

    plt.subplot(121)
    plt.imshow(rho.real, origin='lower')
    plt.title("Palser GCP Rho")

    plt.subplot(122)
    plt.imshow(true_rho.real, origin='lower')
    plt.title("Exact Rho")

    plt.colorbar()
    plt.show()
