import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmwrite, mmread
from scipy import sparse
import sys

#process input parameters
for i in range(1, len(sys.argv), 2):
	argument = sys.argv[i]
	argument_value = sys.argv[i+1]
	if argument == '--hamiltonian':
		hamiltonian_file = argument_value

hamiltonian = mmread(hamiltonian_file).todense()

scipy_density = mmread("scipy_density.mtx").todense()
print("Scipy Eig Vals: ", '\n', np.linalg.eigvalsh(scipy_density))
print("Scipy Trace: ", scipy_density.trace())
print(scipy_density.dot(scipy_density).trace())
scipy_density /= scipy_density.trace()
print("Scipy Energy: ", scipy_density.dot(hamiltonian).trace(), '\n')

ntpoly_density = mmread("density.mtx").todense()
print("NTPoly Eig Vals: ", '\n', np.linalg.eigvalsh(ntpoly_density))
print("NTPoly Trace: ", ntpoly_density.trace())
print(ntpoly_density.dot(ntpoly_density).trace())
ntpoly_density /= ntpoly_density.trace()
print("NTPoly Energy: ", ntpoly_density.dot(hamiltonian).trace(), '\n')

zvode_density = mmread("zvode_density.mtx").todense()
print("Zvode Eig Vals: ", '\n', np.linalg.eigvalsh(zvode_density))
print("Zvode Trace: ", zvode_density.trace())
print(zvode_density.dot(zvode_density).trace())
zvode_density /= zvode_density.trace()
print("Zvode Energy: ", zvode_density.dot(hamiltonian).trace())

print("Norm of diff b/w scipy and ntpoly: ", np.linalg.norm(scipy_density - ntpoly_density))
print("Norm of diff b/w scipy and zvode: ", np.linalg.norm(scipy_density - zvode_density))
print("Norm of diff b/w zvode and ntpoly: ", np.linalg.norm(zvode_density - ntpoly_density))

plt.subplot(131)
plt.title("Scipy Density")
plt.imshow(scipy_density.real, origin='lower')
plt.colorbar()

plt.subplot(132)
plt.title("NTPoly Density")
plt.imshow(ntpoly_density.real, origin='lower')
plt.colorbar()

plt.subplot(133)
plt.title("Zvode Density")
plt.imshow(scipy_density.real, origin='lower')
plt.colorbar()
plt.show()

