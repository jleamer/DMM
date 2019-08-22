import numpy as np
from numba import jitclass, float32, complex128, int32
import time
from scipy.integrate import ode

spec = [
	('H', complex128[:,:]),
	('dbeta', float32),
	('beta', float32),
	('mu', int32),
	('identity', complex128[:,:]),
	('rho', complex128[:,:])
]

#@jitclass(spec)
class GCP_Numba():
	def __init__(self, H, dbeta, beta, mu, identity, rho):
		'''
		The following parameters need to be specified:
			H - the hamiltonian of the system
			dbeta - the step-size in beta (inverse temperature)
		'''

		self.H = H
		self.dbeta = dbeta
		self.beta = beta
		self.mu = mu

		#Save the identity matrix
		self.identity = identity
		
		#Insure that the hamiltonian is Hermitian
		self.H += self.H.conj().T
		self.H *= 0.5

		#Define rho
		self.rho = rho

	def trace(self, A):
		'''
		:param A:		the matrix to be traced
	
		:return trace: 	the trace of A
		'''
		############################################
		#
		#	An implementation of this function is necessary
		#	because numba does not currently seem to 
		#	recognize numpy.trace
		#
		############################################
		trace = 0
		for i in range(A.shape[0]):
			trace += A[i][i]
		return trace

	def eff_trace(self, A, B):
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

	def rhs(self, beta, rho, H, identity):
		'''
		:param beta: 		the current inverse temperature value
		:param rho:			the current density matrix value
		:param H: 			the hamiltonian
		:param identity:	the identity matrix with size equal to the hamiltonian

		:return:			the value of the density matrix at the next beta value
		'''
		rows = int(np.sqrt(rho.size))
		rho = rho.reshape(rows,rows)
		scaledH = -0.5*(H - self.mu * identity)
		K = scaledH.dot(identity - rho)
		f = K.dot(rho) + rho.dot(K.conj().T)
		return f.reshape(-1)

	def zvode(self, nsteps):
		'''
		This function implements scipy's complex valued ordinary differential equation (ZVODE) using the rhs function above
			:param nsteps:	the number of steps to propagate beta
			:returns: 		the density matrix after propagating through beta
		'''
		solver = ode(self.rhs).set_integrator('zvode', method = 'bdf')
		solver.set_initial_value(self.rho.reshape(-1), self.beta).set_f_params(self.H, self.identity)
		start = time.time()
		while solver.successful() and solver.t < self.dbeta*nsteps:
			solver.integrate(solver.t + self.dbeta)
		end = time.time()
		self.rho = solver.y.reshape(self.rho.shape[0], self.rho.shape[0])
		return [solver.y.reshape(self.rho.shape[0], self.rho.shape[0]), end-start]

	def purify(self):
		'''
		:returns: self with rho made idempotent
		'''
		while True:
			rho_sq = self.rho.dot(self.rho)
			rho_cu = self.rho.dot(rho_sq)

			if np.allclose(rho_sq, self.rho, rtol = 1e-10, atol=1e-10):
				break
			
			self.rho = 3*rho_sq - 2*rho_cu

		return self

if __name__ == '__main__':
	H = np.random.rand(100,100) + 1j*np.random.rand(100,100)
	beta = 0.0
	dbeta = 0.003
	nsteps = 1000
	num_electrons = 5
	identity = np.identity(H.shape[0], dtype=H.dtype)
	rho = num_electrons/identity.trace() * identity
	test = GCP_Numba(H, dbeta, beta, num_electrons, identity, rho)	
	
	test.rhs(beta, rho, H, identity)
	results = test.zvode(nsteps)
	print(np.linalg.eigvalsh(results[0]))
	print("Time to complete: %s" % results[1])
