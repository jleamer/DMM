import numpy as np
from numba import njit
from scipy import sparse
from numba import njit
from scipy import linalg
import matplotlib.pyplot as plt
from dmm import DMM
from cp_dmm import CP_DMM
from gcp_dmm import GCP_DMM
from cp_numba_class_test import CP_Numba
from gcp_numba_class_test import GCP_Numba

def generate_H(n_cutoff, size, beta, alpha):
	'''
	Function to generate the Hamiltonian
	'''
	lower = [np.full(i+1, beta) for i in range(n_cutoff)]
	main_lower = [np.full(size-n_cutoff+i, beta) for i in range(n_cutoff)]
	main = [np.full(size, alpha)]
	main_upper = [np.full(size-i-1, beta) for i in range(n_cutoff)]
	upper = [np.full(n_cutoff-i, beta) for i in range(n_cutoff)]
	diags = []	
	diags.extend(lower)
	diags.extend(main_lower)
	diags.extend(main)
	diags.extend(main_upper)
	diags.extend(upper)

	lower_off = [-size+i+1 for i in range(n_cutoff)]
	mainl_off = [-n_cutoff+i for i in range(n_cutoff)]
	main = [size]
	mainu_off = [i+1 for i in range(n_cutoff)]
	upper_off = [size-n_cutoff+i for i in range(n_cutoff)]
	offsets = []
	offsets.extend(lower_off)
	offsets.extend(mainl_off)
	offsets.extend(main)
	offsets.extend(mainu_off)
	offsets.extend(upper_off)
	
	H = sparse.diags(diags, offsets, format='coo', dtype=complex).toarray()
	return H

def palser_cp(num_electrons, H, nsteps):
	'''
	Function to implement palser's CP algorithm
	'''
	identity = np.identity(H.shape[0], dtype=complex)
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
	rho = lamb/size*(mu_bar*identity-H) + num_electrons/size * identity
	E = np.sum(rho * H.T)

	steps = 0
	while True:
		prev_E = E
		rho_sq = rho.dot(rho)
		rho_cu = rho.dot(rho_sq)
		cn = (rho_sq - rho_cu).trace()/(rho-rho_sq).trace()
		
		if cn >= 0.5:
			rho = ((1+cn)*rho_sq - rho_cu)/cn
		else:
			rho = ((1-2*cn)*rho + (1+cn)*rho_sq - rho_cu)/(1-cn)

		E = np.sum(rho * H.T)
		steps += 1
		if E >= prev_E or cn > 1 or cn < 0:
			break

	print("CP steps: ", str(steps))
	return rho

def palser_gcp(mu, H, nsteps):
	'''
	Function to implement palser's GCP algorithm
	'''
	identity = np.identity(H.shape[0], dtype=complex)
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

	lamb = min([1/(max_H-mu), 1/(mu-min_H)])
	rho = lamb/2 * (mu*identity - H) + 0.5*identity
	omega = np.sum(rho * (H-mu*identity).T)
	
	steps = 0
	while True:
		prev_omega = omega
		rho_sq = rho.dot(rho)
		rho_cu = rho.dot(rho_sq)
		rho = 3*rho_sq - 2*rho_cu
		omega = np.sum(rho * (H-mu*identity).T)
		steps += 1

		if omega >= prev_omega:
			break
	
	print("GCP steps: ", str(steps))
	return rho

if __name__ == '__main__':
	#Set up constants
	pi = np.pi
	
	#Set up the initial parameters for the classes
	beta = 0.0
	dbeta = 0.003
	nsteps = 1000
	num_electrons = 5
	mu = -0.09

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
	n_cutoff = 9
	alpha = 1
	beta = pi/4
	H = generate_H(n_cutoff, size, beta, alpha)

	#Create classes and run zvode
	cp_dmm = CP_DMM(H=H, dbeta=dbeta, num_electrons=num_electrons)
	cp_dmm.zvode(nsteps)
	cp_dmm.purify()
	print(np.linalg.norm(cp_dmm.rho.dot(cp_dmm.rho) - cp_dmm.rho, np.inf))
	
	gcp_dmm = GCP_DMM(H=H, dbeta=dbeta, mu=mu)
	gcp_dmm.zvode(nsteps)
	gcp_dmm.purify()

	#Create exact solution density matrix
	# Note: numpy's sinc function multiplies the argument by pi
	exact_rho = np.array([[0.5*np.sinc((i-j)/2) for i in range(size)] for j in range(size)])

	#Exact density matrix via heaviside step function
	scaled_H = mu*cp_dmm.identity.copy() - H
	hs_rho = linalg.funm(scaled_H, lambda _: np.heaviside(_.real, 0.5))

	c = cp_dmm.rho.dot(cp_dmm.identity-cp_dmm.rho)
	alpha = np.sum(H * c.T)/ c.trace()
	scaled_H = alpha*cp_dmm.identity.copy() - H
	cphs_rho = linalg.funm(scaled_H, lambda _: np.heaviside(_.real, 0.5))

	#Get palser's solutions	
	palsercp_rho = palser_cp(num_electrons, H, nsteps)
	print(np.linalg.norm(palsercp_rho.dot(palsercp_rho) - palsercp_rho, np.inf))
	palsergcp_rho = palser_gcp(mu, H, nsteps)

	#Run numba class(es)
	cp_test = CP_Numba(H, 0.0, dbeta, num_electrons, cp_dmm.identity.copy(), cp_dmm.rho.copy())
	cp_test.rhs(0.0, cp_test.rho, H, cp_test.identity)
	cp_test.zvode(nsteps)
	cp_test.purify()
	
	gcp_test = GCP_Numba(H, 0.0, dbeta, mu, gcp_dmm.identity.copy(), gcp_dmm.rho.copy())
	gcp_test.rhs(0.0, gcp_test.rho, H, gcp_test.identity)
	gcp_test.zvode(nsteps)
	gcp_test.purify()

	#Plot exact solution
	plt.figure(1)
	plt.subplot(111)
	plt.ylabel('P_ij')
	plt.xlabel('|i-j|')
	plt.title("CP Comparisons")
	#plt.plot(exact_rho[0][:11], label='Exact')
	plt.plot(cp_dmm.rho[0][:11].real, label='CP')
	plt.plot(palsercp_rho[0][:11].real, 'o', label='Palser CP')
	plt.plot(cp_test.rho[0][:11].real, '^', label='Numba CP')
	plt.plot(cphs_rho[0][:11].real, '*-', label='CP HS Exact')
	#plt.plot(np.zeros(11), 'k')
	plt.legend(numpoints=1)

	plt.figure(2)
	plt.subplot(111)
	plt.ylabel('P_ij')
	plt.xlabel('|i-j|')
	plt.title('GCP Comparisons')
	plt.plot(gcp_dmm.rho[0][:11].real, label='GCP')
	plt.plot(palsergcp_rho[0][:11].real, 'o', label='Palser GCP')
	plt.plot(gcp_test.rho[0][:11].real, '^', label='Numba GCP')
	plt.plot(hs_rho[0][:11].real, '*-', label='GCP HS Exact')
	plt.legend(numpoints=1)
	
	plt.figure(3)
	plt.subplot(131)
	plt.title('Numba CP')
	plt.ylabel('P_j')
	plt.xlabel('P_i')
	plt.imshow(cp_test.rho.real)

	plt.subplot(132)
	plt.title('Palser CP')
	plt.xlabel('P_i')
	plt.imshow(palsercp_rho.real, label='Palser CP')

	plt.subplot(133)
	plt.title('HS Exact')
	plt.xlabel('P_i')
	plt.imshow(cphs_rho.real)
	plt.colorbar()

	plt.figure(4)
	plt.subplot(131)
	plt.title('Numba GCP')
	plt.ylabel('P_j')
	plt.xlabel('P_i')
	plt.imshow(gcp_test.rho.real)
	
	plt.subplot(132)
	plt.title('Palser GCP')
	plt.xlabel('P_i')
	plt.imshow(palsergcp_rho.real)
	
	plt.subplot(133)
	plt.title('HS Exact')
	plt.xlabel('P_i')
	plt.imshow(hs_rho.real)
	plt.colorbar()
	
	plt
	plt.show()
	
