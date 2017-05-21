##########################################################
#
#   Density Matrix Minimization Method for orbital-free Density Functional Theory
#
#   Reproduction of Fig. 1 (2 particles in a Morse potential) from
#   R. F. Ribeiro et. al. Phys. Rev. Lett. 114, 050401 (2015)
#       https://doi.org/10.1103/PhysRevLett.114.050401
#
##########################################################

from dmm import DMMK, np, linalg
#from QuantumClassicalDynamics.mub_qhamiltonian import MUBQHamiltonian
from QuantumClassicalDynamics.central_diff_qhamiltonian import CentralDiffQHamiltonian
import matplotlib.pyplot as plt

# Generate the hamiltonian
qsys = CentralDiffQHamiltonian(
    X_gridDIM=300,
    X_amplitude=6.,
    K="0.5 * P ** 2",
    V="15. * (exp(-X / 2.) -2. * exp(-X / 4.))",
)

dmm = DMMK(
    mu=1.05 * qsys.get_energy(2),
    dbeta=0.001,
    H=qsys.Hamiltonian,
)
dmm.propagate(10000)


plt.subplot(121)
plt.title("Population")
p_dmm = linalg.eigvalsh(dmm.rho)[::-1]

plt.plot(dmm.E, p_dmm, '*-', label="DMM")
plt.plot(dmm.E, dmm.get_exact_pop(), '*-', label='TF')
plt.xlim([dmm.E.min(), dmm.E[5]])
plt.xlabel('Energy')
plt.legend()

##########################################################
#
#   Calculate exact densities by diagonalizing the Hamiltonian
#
##########################################################

exact_density = np.sum(np.abs(qsys.eigenstates[:2]) ** 2, axis=0)
exact_density /= exact_density.sum() * qsys.dX

##########################################################

plt.subplot(122)

plt.title("Density [Phys. Rev. Lett. 114, 050401 (2015)]")

coord_density = np.diag(dmm.rho).real.copy()
coord_density /= coord_density.sum() * qsys.dX

plt.plot(qsys.X, coord_density, label="DMM")
plt.plot(qsys.X, exact_density, label="via diagonalization")
plt.legend()
plt.xlabel('$x$')
plt.ylabel('Density')

plt.xlim([-2., 3.])

plt.show()