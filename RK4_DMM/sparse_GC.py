import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, linalg
from adaptive_GC import CAdaptive_GC_RK4, CAdaptive_GC_RK4_S
import time

# First generate a random hamiltonian of the Huckel model
# i.e. alpha on the diagonal, and gamma on the off-diagonals to represent nearest neighbors
#    for now use random alpha and gamma
np.random.seed(75)
alpha = np.random.random()
gamma = np.random.random()

# define the dimensions of the Hamiltonian and how many elements to consider before cutoff
size = 200

def huckel_hamiltonian(alpha, gamma, size):
    H = sparse.diags([gamma, alpha, gamma], [-1, 0, 1], shape=(size, size)).toarray()
    H[0][size - 1] = gamma
    H[size - 1][0] = gamma
    return H

H = huckel_hamiltonian(alpha, gamma, size)

# define a chemical potential mu
mu = 0.45
beta = 300
ferm_exact = linalg.funm(H, lambda _: np.exp(-beta * (_ - mu)) / (1 + np.exp(-beta * (_ - mu))))

numsteps = 10000
dbeta = beta / numsteps
ovlp = np.identity(H.shape[0])


gcp = CAdaptive_GC_RK4(ovlp=ovlp, H=H, mu=mu, dbeta=dbeta, epsilon=1e-1)

s_gcp = CAdaptive_GC_RK4_S(ovlp=ovlp, H=H, mu=mu, dbeta=dbeta, epsilon=1e-1)
print(s_gcp.inv_ovlp.count_nonzero())

start = time.time()
gcp.propagate(beta)
end = time.time()
print("Dense ver: " + str(end - start))

start = time.time()
s_gcp.s_propagate(beta)
end = time.time()
print("Sparse ver: " + str(end-start))

plt.figure(1)
plt.plot(linalg.eigvalsh(gcp.rho, gcp.ovlp)[::-1], '*-', label="Dense")
plt.plot(linalg.eigvalsh(s_gcp.rho.toarray(), s_gcp.ovlp.toarray())[::-1], '*--', label="Sparse")
plt.plot(linalg.eigvalsh(ferm_exact, ovlp)[::-1], '*-', label="Exact")
plt.title("Populations")
plt.legend(numpoints=1)
plt.show()