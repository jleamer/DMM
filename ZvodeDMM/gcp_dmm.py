import numpy as np
from numba import jit
from types import MethodType, FunctionType
from scipy import linalg, sparse
from scipy.integrate import ode
from dmm import DMM

class GCP_DMM(DMM):
	def __init__(self, **kwargs):
		#Call general DMM constructor
		DMM.__init__(self, **kwargs)

		#Assign remaining variables
		self.mu = mu
	
		try:
			self.mu
		except AttributeError:
			raise AttributeError("Chemical potential needs to be specified")

		#Create the initial density matrix, which is really just the identity matrix in this case
		self.rho = 0.5 * self.identity

	def rhs(self, beta, rho, H, identity, mu):
		'''
		This function implements the cp version of the rhs of the derivative expression 
		for use in the python ODE Solvers
			:param beta:		time step that the function is being called on; not actually used
			:param rho:			the matrix that is being propagated
			:param H:			Hamiltonian operator
			:param identity:	the identity matrix
			:param mu:			the chemical potential
		
			:return f: 			the derivative of the matrix
		'''
		rows = int(np.sqrt(rho.size))
		rho = rho.reshape(rows,rows)
		scaledH = -0.5*(H - mu * identity)
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
		solver.set_initial_value(self.rho.reshape(-1), self.beta).set_f_params(self.H, self.identity, self.mu)
	
		while solver.successful() and solver.t < self.dbeta*nsteps:
			solver.integrate(solver.t + self.dbeta)

		return solver.y.reshape(self.rho.shape[0], self.rho.shape[0])


if __name__ == '__main__':
	H = np.random.rand(2,2)
	dbeta = 0.003
	mu = -0.09
	num_steps = 100

	dmm = GCP_DMM(H=H, dbeta=dbeta, mu=mu)
	print(dmm.zvode(num_steps))
