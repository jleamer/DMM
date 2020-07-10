import numpy as np
from numba import jit
from types import MethodType, FunctionType
from scipy import linalg, sparse
from scipy.integrate import ode
import matplotlib.pyplot as plt
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

		try:
			self.mf
		except AttributeError:
			raise AttributeError("MF not specified from DFT")

		#Define the initial density matrix
		#self.rho = self.num_electrons/self.identity.trace() * self.identity
		self.rho = self.num_electrons/self.ovlp.trace() * self.ovlp
		#self.rho = 0.5 * self.ovlp
		self.mu_list = [self.get_mu()]
		self.no_mu_list = [self.no_get_mu()]


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

	# TODO: Make sure that this expression is correct
	def non_orth_rhs(self, beta, P, H, identity):
		'''
		This function implements a non-orthogonal version of the cp rhs derivative above
		:param beta: 		inverse temperature step; not actually used
		:param P: 		the matrix that is being propagated
		:param H: 			Hamiltonian operator
		:param identity: 	the identity matrix
		:return f:			the derivative of the matrix
		'''

		rows = int(np.sqrt(P.size))
		P = P.reshape(rows, rows)
		c = P.dot(identity-self.inv_ovlp.dot(P))
		d = H.dot(c)
		alpha = np.sum(self.inv_ovlp*d.T)/c.trace()
		scaledH = -0.5*(self.inv_ovlp.dot(H) - alpha*identity)
		K = (identity - self.inv_ovlp.dot(P)).dot(scaledH)
		f = P.dot(K) + K.conj().T.dot(P)
		return f.reshape(-1)

	def zvode(self, nsteps):
		'''
		This function implements scipy's complex valued ordinary differential equation (ZVODE) using the rhs function above
			:param nsteps:	the number of steps to propagate beta
			:returns: 		self
		'''
		solver = ode(self.rhs).set_integrator('zvode', method = 'bdf')
		solver.set_initial_value(self.rho.reshape(-1), self.beta).set_f_params(self.H, self.identity)
		steps = 0
		while solver.successful() and solver.t < self.dbeta*nsteps:
			solver.integrate(solver.t + self.dbeta)
			self.mu_list.append(self.get_mu())
			steps += 1
		#print("CP_Zvode steps: ", str(steps))
		self.rho = solver.y.reshape(self.rho.shape[0], self.rho.shape[0])
		self.beta = solver.t
		return self

	def no_zvode(self, nsteps):
		'''
		This function implements scipy's complex valued ordinary differential equation (ZVODE) using the rhs function above
			:param nsteps:	the number of steps to propagate beta
			:returns: 		the density matrix after propagating through beta
		'''
		solver = ode(self.non_orth_rhs).set_integrator('zvode', method = 'bdf')
		solver.set_initial_value(self.rho.reshape(-1), self.beta).set_f_params(self.H, self.identity)
		steps = 0
		while solver.successful() and solver.t < self.dbeta*nsteps:
			solver.integrate(solver.t + self.dbeta)
			self.mu_list.append(self.get_mu())
			self.rho = solver.y.reshape(self.rho.shape[0], self.rho.shape[0])
			self.no_mu_list.append(self.no_get_mu())
			steps += 1
		#print("CP Zvode steps: ", str(steps))
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
			k1 = self.rhs(self.beta, rhocopy, self.H, self.identity)
			k1 = k1.reshape(rows, rows)

			#k2
			rhotemp = rhocopy + 0.5 * self.dbeta * k1
			k2 = self.rhs(self.beta, rhotemp, self.H, self.identity)
			k2 = k2.reshape(rows, rows)
	
			#k3
			rhotemp = rhocopy + 0.5 * self.dbeta * k2
			k3 = self.rhs(self.beta, rhotemp, self.H, self.identity)
			k3 = k3.reshape(rows, rows)

			#k4
			rhotemp = rhocopy + self.dbeta*k3
			k4 = self.rhs(self.beta, rhotemp, self.H, self.identity)
			k4 = k4.reshape(rows, rows)

			self.beta += self.dbeta
			self.rho += (1/6)*self.dbeta*(k1+2*k2+2*k3+k4)
			self.mu_list.append(self.get_mu())

		return self

	def non_orth_rk4(self, nsteps):
		'''
		This function implements a 4th order Runge Kutta method for the non-orthogonal rhs function
		:param nsteps: the number of steps to run
		:return: self
		'''

		for i in range(nsteps):
			#First make a copy of rho
			rhocopy = self.rho.copy()
			rows = rhocopy.shape[0]

			#k1
			k1 = self.non_orth_rhs(self.beta, rhocopy, self.H, self.identity)
			k1 = k1.reshape(rows, rows)

			#k2
			rhotemp = rhocopy + 0.5 * self.dbeta * k1
			k2 = self.non_orth_rhs(self.beta, rhotemp, self.H, self.identity)
			k2 = k2.reshape(rows, rows)

			#k3
			rhotemp = rhocopy + 0.5 * self.dbeta * k2
			k3 = self.non_orth_rhs(self.beta, rhotemp, self.H, self.identity)
			k3 = k3.reshape(rows, rows)

			#k4
			rhotemp = rhocopy + self.dbeta*k3
			k4 = self.non_orth_rhs(self.beta, rhotemp, self.H, self.identity)
			k4 = k4.reshape(rows, rows)

			self.beta += self.dbeta
			self.rho += (1/6)*self.dbeta*(k1+2*k2+2*k3+k4)
			self.mu_list.append(self.get_mu())
			self.no_mu_list.append(self.no_get_mu())

		return self

	def get_mu(self):
		'''
		This function calculates the chemical potential of the system
		:returns: the chemical potential, mu
		'''
		temp = self.rho.dot(self.identity - self.rho)
		return np.sum(self.H*temp.T)/temp.trace()

	def no_get_mu(self):
		'''
		This function calculates the chemical potential of the system in a non-orthogonal basis
		:return: the chemical potential, mu
		'''
		temp = self.rho.dot(self.identity - self.inv_ovlp.dot(self.rho))
		temp2 = self.H.dot(temp)
		return np.sum(self.inv_ovlp*temp2.T)/temp.trace()

if __name__ == '__main__':
	H = np.random.rand(100,100) + 1j*np.random.rand(100,100)
	dbeta = 0.003
	num_electrons = 5
	num_steps = 1000

	dmm = CP_DMM(H=H, dbeta=dbeta, num_electrons=num_electrons)
	dmm.zvode(num_steps)
	dmm.purify()
	zvode_eig = np.linalg.eigvalsh(dmm.rho)
	
	dmm2 = CP_DMM(H=H, dbeta=dbeta, num_electrons=num_electrons)
	dmm2.rk4(num_steps)
	dmm2.purify()
	rk4_eig = np.linalg.eigvalsh(dmm2.rho)

	plt.subplot(111)
	plt.ylabel("Population")
	plt.xlabel("Energy")
	plt.plot(dmm.E, zvode_eig[::-1], label='Zvode')
	plt.plot(dmm.E, rk4_eig[::-1], 'o', label='RK4')
	plt.legend(numpoints=1)
	plt.show()


