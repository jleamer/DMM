import numpy as np
from scipy.io import mmwrite, mmread
from scipy import linalg
from scipy import sparse
import sys
from functools import partial

if __name__ == "__main__":
	
	#process input parameters
	for i in range(1, len(sys.argv), 2):
		argument = sys.argv[i]
		argument_value = sys.argv[i+1]
		if argument == '--hamiltonian':
			hamiltonian_file = argument_value
		if argument == '--chemical_potential':
			chemical_potential = float(argument_value)
		if argument == '--rows':
			rows = int(argument_value)

	#compute heaviside step function of mu*I-H
	hamiltonian = mmread(hamiltonian_file).toarray()
	scaled_H = chemical_potential * np.identity(hamiltonian[0].size) - hamiltonian	
	rho = linalg.funm(scaled_H, lambda _: np.heaviside(_.real, 0.5))
	#print(np.linalg.eigvalsh(rho))

	mmwrite("scipy_density", sparse.coo_matrix(rho))

