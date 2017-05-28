################################################################
#
#   This file implements propagator with respect
#   to the chemical potential ($\mu$). As we can see
#   this propagation is unstable for many $\mu$ propagation
#
################################################################

from dmm import DMM

from QuantumClassicalDynamics.central_diff_qhamiltonian import CentralDiffQHamiltonian
import matplotlib.pyplot as plt

# Generate the hamiltonian
qsys = CentralDiffQHamiltonian(
    X_gridDIM=300,
    X_amplitude=6.,
    K="0.5 * P ** 2",
    V="15. * (exp(-X / 2.) -2. * exp(-X / 4.))",
)

dmm = DMM(
    mu=-10.,
    dbeta=0.001,
    dmu=0.0005,
    H=qsys.Hamiltonian,
)

dmm.propagate_beta(5000, commnorm=True)

# Function for post processing
F = lambda obj: (
    obj.mu,
    obj.get_average_H(),
    obj.get_exact_average_H()
)

mu, average_H, exact_average_H = zip(
    *[F(dmm.propagate_mu(nsteps=100)) for _ in range(110)]
)

plt.plot(mu, average_H, label='$\langle \hat{H} \\rangle$')
plt.plot(mu, exact_average_H, label="exact")
plt.xlabel('chemical potential, $\mu$')
plt.ylabel('energy')
plt.legend()

plt.show()