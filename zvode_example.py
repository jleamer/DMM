import sys
from dmm import DMM
import numpy as np
from scipy import sparse
from numpy import linalg
from scipy.integrate import ode
from types import MethodType, FunctionType
from scipy.io import mmwrite, mmread
import copy

#MPI


def rhsmatrix(beta, rho, scaledH):
	'''
	This function implements the rhs of the derivative expression 
	for use in the python ODE Solvers
	Input:
		beta - time step that the function is being called on; not actually used
		rho - the matrix that is being propagated
		scaledH - scaled Hamiltonian operator
	Output:
		f - the derivative of the matrix
	'''
	rows = int(np.sqrt(rho.size))
	rho = rho.reshape(rows,rows)
	identity = np.identity(rho.shape[0], dtype=float)
	#mu = H.dot(rho).dot(identity-rho).trace() / rho.dot(identity-rho).trace()
	#scaledH = -0.5*(H - mu* identity)
	K = scaledH.dot(identity - rho)
	f = K.dot(rho) + rho.dot(K.conj().T) - 0*rho*(K.dot(rho)).trace().sum()
	return f.reshape(-1)


if __name__ == '__main__':
	#process input parameters
	for i in range(1, len(sys.argv), 2):
		argument = sys.argv[i]
		argument_value = sys.argv[i+1]
		if argument == '--hamiltonian':
			hamiltonian_file = argument_value
		elif argument == '--chemical_potential':
			chemical_potential = float(argument_value)
		elif argument == '--density_file_out':
			density_file = argument_value

	dmm = DMM(
                dbeta=.03,
                dmu=0.0005,
                mu=chemical_potential,
                # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
                #H=sparse.random(70, 70, density=0.1),
                H= mmread(hamiltonian_file).toarray(),
                #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
                )
	
	final_beta = 1000.0
	count = 0
	scaledH = -0.5 * (dmm.H - dmm.mu * dmm.identity)
	rho = dmm.rho.copy()
	rows = int(np.sqrt(rho.size))
	solver = ode(rhsmatrix).set_integrator('zvode', method = 'bdf')
	solver.set_initial_value(rho.reshape(-1), 0.0).set_f_params(scaledH)
	
	while solver.successful() and solver.t < final_beta:
		solver.integrate(solver.t + dmm.dbeta)
		
	
	rho = solver.y.copy()
	rho = rho.reshape((rows, rows))
	mu = dmm.H.dot(rho).dot(dmm.identity-rho).trace() / rho.dot(dmm.identity-rho).trace()
	print(mu)
	
	#print(np.linalg.norm(rho - rho.dot(rho)))
	#print(rho - rho.dot(rho))
	
	try:
		density_file
	except NameError:
		density_file = "zvode_density.mtx"
	

	mmwrite(density_file, sparse.coo_matrix(rho))
