import numpy as np
from scipy import sparse
from numba import njit
import matplotlib.pyplot as plt
from dmm import DMM
from cp_dmm import CP_DMM
from gcp_dmm import GCP_DMM

def palser_cp(rho, H, nsteps):
	'''
	Function to implement palser's CP algorithm
	'''
	for i in range(nsteps):
		rho_sq = rho.dot(rho)
		rho_cu = rho.dot(rho_sq)
		cn = (rho_sq - rho_cu).trace()/(rho-rho_sq).trace()
		
		if cn >= 0.5:
			rho = ((1+cn)*rho_sq - rho_cu)/cn
		else:
			rho = ((1-2*cn)*rho + (1+cn)*rho_sq - rho_cu)/(1-cn)
	return rho

if __name__ == '__main__':
	#Set up constants
	pi = np.pi
	
	#Set up the initial parameters for the classes
	beta = 0.0
	dbeta = 0.003
	nsteps = 1000
	num_electrons = 5
	mu = 0.0

	#Set up the hamiltonian
	#############################################################
	#
	#	The general form of the Hamiltonian is:
	#
	#		a	B	0	...	0	B
	#		B	a	B	...	0	0
	#		0	B	a	...	0	0
	#		.					.
	#		.					.	
	#		0				a	B
	#		B	0	0	...	B	a
	#
	#	This represents a 1D chain with on-site parameters alpha
	#	and nearest-neighbor parameters beta.  Truncation to n
	#	nearest neighbors determine how many betas there are away
	#	from the main diagonal.
	#
	#	The energy of the orbital is determined from: E = a + (4/pi)*beta
	#
	#	Because energy levels are relative, alpha can be set to 0
	#	without changing final results.  Thus, beta/E = pi/4
	#
	#	The exact expression for the density matrix of such a systems is:
	#		rho_ij = 0.5 * sinc[(i-j)*pi/2]
	#
	##############################################################

	size = 100
	n_cutoff = 10
	beta = pi/4
	lower = [np.full(i+1, beta) for i in range(n_cutoff)]
	main_lower = [np.full(size-n_cutoff+i, beta) for i in range(n_cutoff)]
	main_upper = [np.full(size-i-1, beta) for i in range(n_cutoff)]
	upper = [np.full(n_cutoff-i, beta) for i in range(n_cutoff)]
	diags = []	
	diags.extend(lower)
	diags.extend(main_lower)
	diags.extend(main_upper)
	diags.extend(upper)

	lower_off = [-size+i+1 for i in range(n_cutoff)]
	mainl_off = [-n_cutoff+i for i in range(n_cutoff)]
	mainu_off = [i+1 for i in range(n_cutoff)]
	upper_off = [size-n_cutoff+i for i in range(n_cutoff)]
	offsets = []
	offsets.extend(lower_off)
	offsets.extend(mainl_off)
	offsets.extend(mainu_off)
	offsets.extend(upper_off)
	
	H = sparse.diags(diags, offsets, format='coo', dtype=complex).toarray()

	#Create classes and run zvode
	cp_dmm = CP_DMM(H=H, dbeta=dbeta, num_electrons=num_electrons)
	cp_rho = cp_dmm.zvode(nsteps)
	
	gcp_dmm = GCP_DMM(H=H, dbeta=dbeta, mu=mu)
	gcp_rho = gcp_dmm.zvode(nsteps)	

	#Create exact solution density matrix
	# Note: numpy's sinc function multiplies the argument by pi
	exact_rho = np.array([[0.5*np.sinc((i-j)/2) for i in range(size)] for j in range(size)])

	#Get palser's solution
	min_Hi = []
	max_Hi = []
	for i in range(size):
		min_sum = 0
		max_sum = 0
		for j in range(size):
			if i != j:
				min_sum -= H[i][j]
				max_sum += H[i][j]
		min_Hi.append(min_sum)
		max_Hi.append(max_sum)
	min_H = min(min_Hi)
	max_H = max(max_Hi)
	
	mu_bar = H.trace()/size
	lamb = min([num_electrons/(max_H-mu_bar), (size-num_electrons)/(mu_bar-min_H)])
	rho_0 = lamb/size*(mu_bar*cp_dmm.identity-H) + num_electrons/size * cp_dmm.identity
	
	palser_rho = palser_cp(rho_0, H, nsteps)
	
	#Plot exact solution
	plt.subplot(111)
	plt.ylabel('P_ij')
	plt.xlabel('|i-j|')
	plt.plot(exact_rho[0][:10], label='Exact')
	plt.plot(cp_rho[0][:10].real, label='CP')
	plt.plot(gcp_rho[0][:10].real, label='GCP')
	plt.plot(palser_rho[0][:10].real, 'o', label='Palser')
	plt.plot(np.zeros(10), 'k')
	plt.legend(numpoints=1)
	plt.show()
	
