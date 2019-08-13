import numpy as np
from numba import njit
import time
from scipy.integrate import ode

@njit
def trace(array):
	'''
	:param array:	the array to be traced
	
	:return trace: 	the trace of the array
	'''
	trace = 0
	for i in range(array.shape[0]):
		trace += array[i][i]
	return trace

@njit
def eff_trace(A, B):
	'''
	:param A, B: 	the arrays whose dot product is to be traced over
	:return: 		Tr(AB)
	'''
	#############################################
	#
	#	Tr(AB) = A_ij * B_ji = A_ij * (B^T)_ij
	#	
	#	This is more optimal than taking the trace of a dot product because
	#	we can avoid a matrix multiplication
	#
	#############################################
	return np.sum(A * B.T)

@njit
def rhs(beta, rho, H, identity):
	'''
	:param beta: 		the current inverse temperature value
	:param rho:			the current density matrix value
	:param H: 			the hamiltonian
	:param identity:	the identity matrix with size equal to the hamiltonian

	:return f:			the value of the density matrix at the next beta value
	'''
	rows = int(np.sqrt(rho.size))
	rho = rho.reshape(rows,rows)
	c = rho.dot(identity-rho)
	alpha = eff_trace(H, c) / trace(c)
	scaledH = -0.5*(H - alpha* identity)
	K = scaledH.dot(identity - rho)
	f = K.dot(rho)
	f += f.conj().T
	return f.reshape(-1)

def wrapper(beta, rho, H, identity):
	return rhs(beta, rho, H, identity)

if __name__ == '__main__':
	H = np.random.rand(100,100) + 1j*np.random.rand(100,100)
	H += H.conj().T
	identity = np.identity(H.shape[0], dtype=complex)
	beta = 0.0
	dbeta = 0.003
	nsteps = 1000
	num_electrons = 5
	rho = num_electrons/identity.trace() * identity
	
	rhs(beta, rho, H, identity)
	solver = ode(wrapper).set_integrator('zvode', method = 'bdf')
	solver.set_initial_value(rho.reshape(-1), beta).set_f_params(H, identity)
	start = time.time()
	while solver.successful() and solver.t < dbeta*nsteps:
		solver.integrate(solver.t + dbeta)
	finish = time.time()

	print(np.linalg.eigvalsh(solver.y.reshape(rho.shape[0], rho.shape[0])))
	print("Time to complete: %s" % (finish-start))
