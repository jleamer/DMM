import numpy as np
from numba import njit
from scipy import sparse
from scipy.io import mmwrite, mmread
from numba import njit
from scipy import linalg
import matplotlib.pyplot as plt
import sys


def generate_H(n_cutoff, size, b, a):
    lower = [np.full(i + 1, b) for i in range(n_cutoff)]
    main_lower = [np.full(size - n_cutoff + i, b) for i in range(n_cutoff)]
    main = [np.full(size, a)]
    main_upper = [np.full(size - i - 1, b) for i in range(n_cutoff)]
    upper = [np.full(n_cutoff - i, b) for i in range(n_cutoff)]
    diags = []
    diags.extend(lower)
    diags.extend(main_lower)
    diags.extend(main)
    diags.extend(main_upper)
    diags.extend(upper)

    lower_off = [-size + i + 1 for i in range(n_cutoff)]
    mainl_off = [-n_cutoff + i for i in range(n_cutoff)]
    main = [0]
    mainu_off = [i + 1 for i in range(n_cutoff)]
    upper_off = [size - n_cutoff + i for i in range(n_cutoff)]
    offsets = []
    offsets.extend(lower_off)
    offsets.extend(mainl_off)
    offsets.extend(main)
    offsets.extend(mainu_off)
    offsets.extend(upper_off)

    H = sparse.diags(diags, offsets, format='coo', dtype=complex)
    mmwrite("hamiltonian.mtx", H)
    H = H.toarray()
    return H


def greshgorin(H):
    """
    function for estimating the maximum and minimum of H's spectrum
    :param H:
    :return:
    """
    size = H.shape[0]
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

    return max_H, min_H


def gcp(mu, H, nsteps):
    """
    Function to evaluate the GCP method proposed in https://journals.aps.org/prb/pdf/10.1103/PhysRevB.58.12704 by Palser
    :param mu:      the chemical potential of the system
    :param H:       the Hamiltonian of the system
    :param nsteps:  the number of steps to run
    :return:        the density matrix, rho
    """
    identity = np.identity(H.shape[0], dtype=complex)

    max_H, min_H = greshgorin(H)

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


def cp(num_electrons, H, nsteps):
    """
    function for performing Palser's canonical purification method
    :param num_electrons:
    :param H:
    :param nsteps:
    :return:
    """
    identity = np.identity(H.shape[0], dtype=complex)
    size = identity.shape[0]
    trace_list = []

    maxH, minH = greshgorin(H)
    mu_bar = H.trace()/size
    lamb = min(num_electrons/(maxH - mu_bar), (size - num_electrons)/(mu_bar - minH))
    rho = lamb/size * (mu_bar*identity - H) + num_electrons/size * identity
    trace_list.append(rho.trace())
    E = np.sum(rho * H)

    steps = 0
    while steps < nsteps:
        prev_E = E
        rho_sq = rho @ rho
        rho_cu = rho @ rho_sq

        c = (rho_sq - rho_cu).trace()/(rho - rho_sq).trace()

        if c >= 0.5:
            rho = ((1+c)*rho_sq - rho_cu)/c
        else:
            rho = ((1-2*c)*rho + (1+c)*rho_sq - rho_cu)/(1-c)

        E = np.sum(rho * H.T)
        steps += 1
        if E >= prev_E or c > 1 or c < 0:
            break

    print("CP steps: ", str(steps))
    return rho



if __name__ == '__main__':
    """
    Simple calculation using tight-binding model
    """
    np.random.seed(5)
    size = 100
    cutoff = 9
    alpha = np.random.random()
    beta = np.random.random()

    H = generate_H(cutoff, size, beta, alpha)
    mu = .05
    rho = gcp(mu, H, 1000)

    true_rho = np.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            true_rho[i][j] = 0.5 * np.sinc(np.pi / 2 * (i - j))

    print("Energy via GCP: ", np.trace(rho.dot(H.T)))
    print("Exact energy: ", alpha + 4 * beta / np.pi)

    plt.subplot(121)
    plt.imshow(rho.real, origin='lower')
    plt.title("Palser GCP Rho")

    plt.subplot(122)
    plt.imshow(true_rho, origin='lower')
    plt.title("Exact Rho")

    plt.colorbar()
    plt.show()
