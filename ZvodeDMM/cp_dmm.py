import numpy as np
from numba import jit
from types import MethodType, FunctionType
from scipy import linalg, sparse
from scipy.integrate import ode
from dmm import DMM
import time

class CP_DMM(DMM):
	def __init__(self, **kwargs):
		#Call the DMM constructor
		DMM.__init__(self, **kwargs)

		try:
			self.num_electrons
		except AttributeError:
			raise AttributeError("Number of electrons need to be specified")

		#Define the initial density matrix
		self.rho = self.num_electrons/self.identity.trace() * self.identity

	def rhs(self, beta, rho, H, identity):
		'''
		This function implements the cp version of the rhs of the derivative expression 
		for use in the python ODE Solvers
			:param beta:		time step that the function is being called on; not actually used
			:param rho:			the matrix that is being propagated
			:param H:			Hamiltonian operator
			:param identity:	the identity matrix
		
			:return f: 			the derivative of the matrix
		'''
		rows = int(np.sqrt(rho.size))
		rho = rho.reshape(rows,rows)
		c = rho.dot(identity-rho)
		alpha = np.sum(H*c.T) / c.trace()
		scaledH = -0.5*(H - alpha* identity)
		K = scaledH.dot(identity - rho)
		f = K.dot(rho)
		f += f.conj().T
		return f.reshape(-1)

	def zvode(self, nsteps):
		'''
		This function implements scipy's complex valued ordinary differential equation (ZVODE) using the rhs function above
			:param nsteps:	the number of steps to propagate beta
			:returns: 		the density matrix after propagating through beta
		'''
		solver = ode(self.rhs).set_integrator('zvode', method = 'bdf')
		solver.set_initial_value(self.rho.reshape(-1), self.beta).set_f_params(self.H, self.identity)
	
		while solver.successful() and solver.t < self.dbeta*nsteps:
			solver.integrate(solver.t + self.dbeta)

		return solver.y.reshape(self.rho.shape[0], self.rho.shape[0])

if __name__ == '__main__':
	H = np.random.rand(100,100) + 1j*np.random.rand(100,100)
	dbeta = 0.003
	num_electrons = 5
	num_steps = 1000
	
	dmm = CP_DMM(H=H, dbeta=dbeta, num_electrons=num_electrons)
	start = time.time()
	print(np.linalg.eigvalsh(dmm.zvode(num_steps)))
	end = time.time()
	print("Time to complete: %s" % (end-start))
