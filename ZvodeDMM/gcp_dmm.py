import numpy as np
from numba import jit
from types import MethodType, FunctionType
from scipy import linalg, sparse
from scipy.integrate import ode
import matplotlib.pyplot as plt
from dmm import DMM

class GCP_DMM(DMM):
	def __init__(self, **kwargs):
		#Call general DMM constructor
		DMM.__init__(self, **kwargs)
	
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
		steps = 0
		while solver.successful() and solver.t < self.dbeta*nsteps:
			solver.integrate(solver.t + self.dbeta)
			steps += 1
		print("GCP Zvode steps: ", str(steps))
		self.rho = solver.y.reshape(self.rho.shape[0], self.rho.shape[0])
		self.beta = solver.t
		return self

	def rk4(self, nsteps):
		'''
		This function implements an adaptive 4th order Runge-Kutta method using this class's rhs function
		:param nsteps:	the number of steps to propagate beta
		:returns: 		self
		'''

		#######################################################
		#
		#	The method is:
		#
		#		rho(beta + dbeta) = rho(beta) + (1/6)*dbeta*(k1 + 2k2 + 2k3 + k4)
		#
		#	with
		#		 f(rho) = K.rho + rho.K^{\dagger}
		#		 K = -0.5*(H-mu)(I-rho)
		#		 k1 = f(rho)
        #        k2 = f(rho + 0.5*dbeta*k1)
        #        k3 = f(rho + 0.5*dbeta*k2)
        #        k4 = f(rho + dbeta*k3)
		#
		#######################################################
		
		for i in range(nsteps):
			#First make a copy of rho
			rhocopy = self.rho.copy()
			rows = rhocopy.shape[0]
		
			#k1
			k1 = self.rhs(self.beta, rhocopy, self.H, self.identity, mu)
			k1 = k1.reshape(rows,rows)

			#k2
			rhotemp = rhocopy + 0.5 * self.dbeta * k1
			k2 = self.rhs(self.beta, rhotemp, self.H, self.identity, mu)
			k2 = k2.reshape(rows, rows)
	
			#k3
			rhotemp = rhocopy + 0.5 * self.dbeta * k2
			k3 = self.rhs(self.beta, rhotemp, self.H, self.identity, mu)
			k3 = k3.reshape(rows, rows)

			#k4
			rhotemp = rhocopy + self.dbeta*k3
			k4 = self.rhs(self.beta, rhotemp, self.H, self.identity, mu)
			k4 = k4.reshape(rows, rows)

			self.beta += self.dbeta
			self.rho += (1/6)*self.dbeta*(k1+2*k2+2*k3+k4)

		return self

if __name__ == '__main__':
	H = np.random.rand(100,100) + 1j*np.random.rand(100,100)
	dbeta = 0.003
	mu = -0.09
	num_steps = 1000

	dmm = GCP_DMM(H=H, dbeta=dbeta, mu=mu)
	dmm.zvode(num_steps)
	#dmm.purify()
	zvode_eig = np.linalg.eigvalsh(dmm.rho)

	dmm2 = GCP_DMM(H=H, dbeta=dbeta, mu=mu)
	dmm2.rk4(num_steps)
	#dmm2.purify()
	rk4_eig = np.linalg.eigvalsh(dmm.rho)

	plt.subplot(111)
	plt.ylabel("Population")
	plt.xlabel("Energy")
	plt.plot(dmm.E, zvode_eig[::-1], label='Zvode')
	plt.plot(dmm.E, rk4_eig[::-1], 'o', label='RK4')
	plt.legend(numpoints=1)
	plt.show()
