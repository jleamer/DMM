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
		
		
	print("Beta: ", final_beta)
	print("Temp: ", 1/(final_beta*1.38e-23))
	rho = solver.y.copy()
	rho = rho.reshape((rows, rows))
	#print(np.linalg.norm(rho - rho.dot(rho)))
	#print(rho - rho.dot(rho))
	
	mmwrite("zvode_density", sparse.coo_matrix(rho))