import numpy as np
from numba import jit
from types import MethodType, FunctionType
from scipy import linalg, sparse
from scipy.integrate import ode

class DMM:
	def __init__(self, **kwargs):
		'''
		The following parameters need to be specified:
			H - the hamiltonian of the system
			dbeta - the step-size in beta (inverse temperature)
		'''

		# save all attributes
		for name, value in kwargs.items():
			# if the value supplied is a function, then dynamically assign it as a method;
			# otherwise bind it a property
			if isinstance(value, FunctionType):
				setattr(self, name, MethodType(value, self, self.__class__))
			else:
				setattr(self, name, value)

		#Check that everything was assigned
		try:
			self.H
		except AttributeError:
			raise AttributeError("The Hamiltonian (H) was not specified")

		try:
			self.dbeta
		except AttributeError:
			raise AttributeError("The step size dbeta was not specified")
		
		try:
			self.beta
		except AttributeError:
			self.beta = 0.0


		#Save the identity matrix
		self.identity = np.identity(self.H.shape[0], dtype=self.H.dtype)
		
		#Insure that the hamiltonian is Hermitian
		self.H += self.H.conj().T
		self.H *= 0.5

		#Save energy eigvenvalues for later analysis
		self.E = linalg.eigvalsh(self.H)
	
	def get_exact_pop(self, **kwargs):
		'''
		:returns: the exact Fermi-Dirac population distribution
		'''
		if 'mu' in kwargs:
			mu = kwargs['mu']
		else:
			mu = self.mu
		print(self.beta)
		return 1 / (1 + np.exp(self.beta*(self.E - mu)))

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
