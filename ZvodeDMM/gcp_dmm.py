import numpy as np
from numba import jit
from types import MethodType, FunctionType
from scipy import linalg, sparse
from scipy.io import mmread, mmwrite
from scipy.integrate import ode
from scipy.linalg import eigh, eig
import matplotlib.pyplot as plt
from ZvodeDMM.dmm import DMM
from pyscf import gto, dft


class GCP_DMM(DMM):
	def __init__(self, **kwargs):
		# Call general DMM constructor
		DMM.__init__(self, **kwargs)
	
		try:
			self.mu
		except AttributeError:
			raise AttributeError("Chemical potential needs to be specified")

		try:
			self.mf
		except AttributeError:
			raise AttributeError("MF not specified from DFT")

		# Create the initial density matrix
		# self.rho = 0.5 * self.identity
		self.rho = 0.5 * self.ovlp
		self.num_electrons = [self.rho.trace()]
		self.y = -1/4*(self.H + self.mf.get_veff(self.mf.mol, self.rho) - self.mu*self.ovlp)
		self.hexc = [self.mf.get_veff(self.mf.mol, self.rho)]

		"""
		# Create the initial y for iteration
		self.A = 0.062184
		self.b = 3.72744
		self.c = 12.9352
		self.Q = np.sqrt(4*self.c-self.b**2)
		self.x0 = -0.10498
		self.rs = (3/(4*np.pi*self.rho.trace()))**(1/3)
		self.x = np.sqrt(self.rs)

		# Note that Hexc is vxld+vcld using LDA approximation
		self.y0 = -1/4*(self.H + self.vxld()+self.vcld() - self.mu*self.ovlp)

	def X(self, x):
		return x**2 + self.b*x + self.c

	def vxld(self):
		return -(3*np.pi**2*self.rho.trace())**(1/3)/np.pi

	def vcld(self):
		tan_arg = self.Q/(2*self.x+self.b)
		x_diff = self.x-self.x0
		return self.A/2*(np.log(self.x**2/self.X(self.x)) + 2*self.b/self.Q*np.arctan(tan_arg) -\
						self.b*self.x0/self.X(self.x0)*(np.log(x_diff**2/self.X(self.x)) + 2*(self.b+2*self.x0)/self.Q*np.arctan(tan_arg))) -\
							self.A/6 * (self.c*x_diff-self.b*self.x*self.x0)/(x_diff*self.X(self.x))
		"""

	def rhs(self, beta, rho, H, identity, mu):
		"""
		This function implements the gcp version of the rhs of the derivative expression
		for use in the python ODE Solvers
			:param beta:		time step that the function is being called on; not actually used
			:param rho:			the matrix that is being propagated
			:param H:			Hamiltonian operator
			:param identity:	the identity matrix
			:param mu:			the chemical potential

			:return f: 			the derivative of the matrix
		"""
		rows = int(np.sqrt(rho.size))
		rho = rho.reshape(rows,rows)
		scaledH = -0.5*(H - mu * identity)
		K = scaledH.dot(identity - rho)
		f = K.dot(rho) + rho.dot(K.conj().T)
		return f.reshape(-1)

	def non_orth_rhs(self, beta, P, H, identity, mu):
		"""
		This function implements the gcp version of the rhs of the derivative expression
		for use in the python ODE solvers for non-orthonormal bases
		:param beta:
		:param rho:
		:param H:
		:param identity:
		:param mu:
		:return:
		"""
		rows = int(np.sqrt(P.size))
		P = P.reshape(rows, rows)
		scaledH = -0.5*(self.inv_overlap.dot(H) - mu*identity)
		K = (identity - self.inv_overlap.dot(P)).dot(scaledH)
		f = P.dot(K) + K.conj().T.dot(P)
		return f.reshape(-1)

	def sc_rhs(self, beta, P, H, identity, mu, Hexc):
		"""
		This function implements the gcp version of the rhs of the derivative expression
		for use in the python ODE solvers
		:param beta:
		:param P:
		:param H:
		:param identity:
		:param mu:
		:return:
		"""
		rows = int(np.sqrt(P.size))
		P = P.reshape(rows, rows)
		scaledH = -0.5*(self.inv_overlap.dot(H) + self.inv_overlap.dot(self.vxld()+self.vcld()))
		return

	def F(self, eigval):
		return 1/(1+np.exp(eigval))

	def deriv_Hexc(self):
		return (self.mf.get_veff(self.mf.mol, self.rho + 0.001) - self.mf.get_veff(self.mf.mol, self.rho - 0.001))/(0.001**2)
		#return self.mf.get_veff(self.mf.mol, self.rho)
		#return 3*self.rho @ self.rho

	def dkeq(self, eigvals, eigvecs, y, beta, P):
		total = 0
		rows = int(np.sqrt(y.size))
		proj = self.get_projectors(eigvals, eigvecs)
		for i in range(rows):
			for j in range(rows):
				if i == j:
					total += -1/(1+np.exp(eigvals[j]))**2 * np.exp(eigvals[j]) * proj[j] @ self.inv_overlap @ self.deriv_Hexc() @ y @ proj[j]

				else:
					total += (self.F(eigvals[i])-self.F(eigvals[j]))/(eigvals[i]-eigvals[j]) * proj[i] @ self.inv_overlap @ self.deriv_Hexc() @ y @ proj[j]

		return total

	def get_projectors(self, eigvals, eigvecs):
		"""
		Function for computing the projectors onto the space of the eigenvalues
		:param eigvals: array of eigenvalues - needed to determine degeneracy
		:param eigvecs: array of eigenvectors
		:return proj: 	array of projectors
		"""
		degen = False
		pairs = []
		for i in range(eigvals.size):
			for j in range(eigvals.size):
				if i != j:
					if eigvals[i] == eigvals[j]:
						degen = True
						pairs.append((i, j))

		proj = np.zeros((eigvals.size, self.rho.shape[0], self.rho.shape[0]), dtype=complex)
		if degen:
			print("Degenerate spectrum")
			for i in range(eigvecs.shape[0]):
				proj[i] = np.outer(eigvecs[i], eigvecs[i].conj().T)
		else:
			for i in range(eigvecs.shape[0]):
				proj[i] = np.outer(eigvecs[i], eigvecs[i].conj().T)
		return proj

	def calc_ynext(self, beta, P, H, identity, mu, y):
		rows = int(np.sqrt(P.size))
		P = P.reshape(rows, rows)
		Hexc = self.mf.get_veff(self.mf.mol, P)
		scriptH = beta*(self.inv_overlap.dot(H) + self.inv_overlap.dot(Hexc) - mu*identity)
		eigvals, eigvecs = eig(scriptH, self.ovlp)
		print("Eigvals: ", eigvals)
		#print("Eigvecs: ", eigvecs)

		scaledH = -0.5*(self.inv_overlap.dot(H) + self.inv_overlap.dot(Hexc) - mu*identity)
		K = (identity-self.inv_overlap.dot(P)).dot(scaledH)
		f = P.dot(K) + K.conj().T.dot(P)

		DK = 0.5*self.dkeq(eigvals, eigvecs, y, beta, P)

		ynext = f + DK + DK.conj().T
		return ynext

	def test_rhs(self, dbeta, P, y):
		"""
		Function for testing self-consistent algorithm using an Hexc that we know analytically
		:param beta:
		:param P:
		:param H:
		:param identity:
		:param mu:
		:return:
		"""

		return P + dbeta*y

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

	def no_zvode(self, nsteps):
		'''
		This function implements scipy's complex valued ordinary differential equation (ZVODE) using the rhs function above
			:param nsteps:	the number of steps to propagate beta
			:returns: 		the density matrix after propagating through beta
		'''
		solver = ode(self.non_orth_rhs).set_integrator('zvode', method = 'bdf')
		solver.set_initial_value(self.rho.reshape(-1), self.beta).set_f_params(self.H, self.identity, self.mu)
		steps = 0
		while solver.successful() and solver.t < self.dbeta*nsteps:
			solver.integrate(solver.t + self.dbeta)
			self.num_electrons.append(solver.y.reshape(self.rho.shape[0], self.rho.shape[0]).trace())
			#self.hexc.append(self.mf.get_veff(self.mf.mol, solver.y.reshape(self.rho.shape[0], self.rho.shape[0])))
			steps += 1
		print("GCP Zvode steps: ", str(steps))
		self.rho = solver.y.reshape(self.rho.shape[0], self.rho.shape[0])
		self.beta = solver.t
		return self

	def rk4(self, nsteps):
		'''
		This function implements a 4th order Runge-Kutta method using this class's rhs function
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
			k1 = self.rhs(self.beta, rhocopy, self.H, self.identity, self.mu)
			k1 = k1.reshape(rows,rows)

			#k2
			rhotemp = rhocopy + 0.5 * self.dbeta * k1
			k2 = self.rhs(self.beta, rhotemp, self.H, self.identity, self.mu)
			k2 = k2.reshape(rows, rows)
	
			#k3
			rhotemp = rhocopy + 0.5 * self.dbeta * k2
			k3 = self.rhs(self.beta, rhotemp, self.H, self.identity, self.mu)
			k3 = k3.reshape(rows, rows)

			#k4
			rhotemp = rhocopy + self.dbeta*k3
			k4 = self.rhs(self.beta, rhotemp, self.H, self.identity, self.mu)
			k4 = k4.reshape(rows, rows)

			self.beta += self.dbeta
			self.rho += (1/6)*self.dbeta*(k1+2*k2+2*k3+k4)
			self.num_electrons.append(self.rho.trace())
			#self.hexc.append(self.mf.get_veff(self.mf.mol, self.rho))

		return self

	def non_orth_rk4(self, nsteps):
		'''
		This function implements a 4th order Runge-Kutta method using the non-orthogonal rhs
		:param nsteps: the number of steps to run
		:return: self
		'''
		for i in range(nsteps):
			# First make a copy of rho
			rhocopy = self.rho.copy()
			rows = rhocopy.shape[0]

			# k1
			k1 = self.non_orth_rhs(self.beta, rhocopy, self.H, self.identity, self.mu)
			k1 = k1.reshape(rows,rows)

			# k2
			rhotemp = rhocopy + 0.5 * self.dbeta * k1
			k2 = self.non_orth_rhs(self.beta, rhotemp, self.H, self.identity, self.mu)
			k2 = k2.reshape(rows, rows)

			# k3
			rhotemp = rhocopy + 0.5 * self.dbeta * k2
			k3 = self.non_orth_rhs(self.beta, rhotemp, self.H, self.identity, self.mu)
			k3 = k3.reshape(rows, rows)

			# k4
			rhotemp = rhocopy + self.dbeta*k3
			k4 = self.non_orth_rhs(self.beta, rhotemp, self.H, self.identity, self.mu)
			k4 = k4.reshape(rows, rows)

			self.beta += self.dbeta
			self.rho += (1/6)*self.dbeta*(k1+2*k2+2*k3+k4)
			self.num_electrons.append(self.rho.trace())

		return self

if __name__ == '__main__':
	H = np.random.rand(11,11) + 1j*np.random.rand(11,11)
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
	rk4_eig = np.linalg.eigvalsh(dmm2.rho)

	ovlp = mmread("dft_overlap.mtx").toarray()
	dmm3 = GCP_DMM(H=H, dbeta=dbeta, mu=mu, ovlp=ovlp)
	dmm3.no_zvode(num_steps)
	no_zvode_eig = np.linalg.eigvalsh(dmm3.rho)

	plt.subplot(111)
	plt.ylabel("Population")
	plt.xlabel("Energy")
	plt.plot(dmm.E, zvode_eig[::-1], label='Zvode')
	plt.plot(dmm.E, rk4_eig[::-1], 'o', label='RK4')
	plt.plot(dmm.E, no_zvode_eig[::-1], '*', label='NO Zvode')
	plt.legend(numpoints=1)
	plt.show()
