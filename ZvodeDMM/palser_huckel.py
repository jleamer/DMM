import numpy as np
from numba import njit
from scipy import sparse
from scipy.io import mmwrite, mmread
from numba import njit
from scipy import linalg
import matplotlib.pyplot as plt
from dmm import DMM
from cp_dmm import CP_DMM
from gcp_dmm import GCP_DMM
from cp_numba_class_test import CP_Numba
from gcp_numba_class_test import GCP_Numba
import sys

sys.path.insert(1, '/home/jacob/Documents/DMM')

# NT Poly
from NTPoly.Build.python import NTPolySwig as NT

# MPI Module
from mpi4py import MPI
comm = MPI.COMM_WORLD

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
	main = [0]
	mainu_off = [i+1 for i in range(n_cutoff)]
	upper_off = [size-n_cutoff+i for i in range(n_cutoff)]
	offsets = []
	offsets.extend(lower_off)
	offsets.extend(mainl_off)
	offsets.extend(main)
	offsets.extend(mainu_off)
	offsets.extend(upper_off)
	
	H = sparse.diags(diags, offsets, format='coo', dtype=complex)
	mmwrite("hamiltonian.mtx", H)
	H = H.toarray()
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
	dbeta = 0.03
	nsteps = 1000
	num_electrons = 16
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
	#	We use ethylene as an example here
	#		alpha = -11.4 eV
	#		beta = 3 eV (~65 kcal/mol)
	##############################################################

	size = 100
	n_cutoff = 1
	alpha = -11.4
	beta = 3
	H = generate_H(n_cutoff, size, beta, alpha)
	identity = np.identity(size, dtype=complex)
	
	#Run NTPoly's TRS4 method on our Hamiltonian
	# Set up parameters
	convergence_threshold = 1e-10
	threshold = 1e-10
	process_rows = 1
	process_columns = 1
	process_slices = 1
	hamiltonian_file = "hamiltonian.mtx"
	density_file = "density.mtx"

	#Construct process grid
	NT.ConstructGlobalProcessGrid(process_rows, process_columns, process_slices)
	
	#Setup solver parameters
	solver_parameters = NT.SolverParameters()
	solver_parameters.SetConvergeDiff(convergence_threshold)
	solver_parameters.SetThreshold(threshold)
	solver_parameters.SetVerbosity(False)

	overlap = sparse.identity(size, format='coo', dtype='complex')
	mmwrite("overlap", overlap)
	ntpoly_hamiltonian = NT.Matrix_ps("hamiltonian.mtx")
	ntpoly_overlap = NT.Matrix_ps("overlap.mtx")
	Density = NT.Matrix_ps(ntpoly_hamiltonian.GetActualDimension())
		
	#Compute the density matrix
	energy_value, chemical_potential = \
			NT.DensityMatrixSolvers.PM(ntpoly_hamiltonian, ntpoly_overlap, num_electrons, Density, solver_parameters)
	print("NTPoly mu: ", chemical_potential)

	#Output density matrix
	Density.WriteToMatrixMarket(density_file)
	ntpoly_density = mmread(density_file).toarray()
	ntcp_eigs = np.linalg.eigvalsh(ntpoly_density)
	NT.DestructGlobalProcessGrid()	


	#Create classes and run zvode
	cp_dmm = CP_DMM(H=H, dbeta=dbeta, num_electrons=num_electrons)
	cp_dmm.zvode(nsteps)
	cp_dmm.purify()
	print("CP Idempotency: ", str(np.linalg.norm(cp_dmm.rho.dot(cp_dmm.rho) - cp_dmm.rho, np.inf)))
	cp_eigs = np.linalg.eigvalsh(cp_dmm.rho)
	print("CP Energy: ", np.sum(cp_dmm.rho * H.T))
	
	gcp_dmm = GCP_DMM(H=H, dbeta=dbeta, mu=mu)
	gcp_dmm.zvode(nsteps)
	gcp_dmm.purify()
	gcp_eigs = np.linalg.eigvalsh(gcp_dmm.rho)
	
	
	#Create exact solution density matrix
	# Note: numpy's sinc function multiplies the argument by pi
	exact_rho = np.array([[0.5*np.sinc((i-j)/2) for i in range(size)] for j in range(size)])

	#Get palser's solutions	
	palsercp_rho = palser_cp(num_electrons, H, nsteps)
	print("Palser Idempotency: ", np.linalg.norm(palsercp_rho.dot(palsercp_rho) - palsercp_rho, np.inf))
	pcp_eigs = np.linalg.eigvalsh(palsercp_rho)
	print("Palser Energy: ", np.sum(palsercp_rho * H.T))

	palsergcp_rho = palser_gcp(mu, H, nsteps)
	pgcp_eigs = np.linalg.eigvalsh(palsergcp_rho)

	#Exact density matrix via heaviside step function
	#Compare mu for palser cp and our cp methods	
	#temp = palsercp_rho.dot(identity - palsercp_rho)
	#p_alpha = np.sum(H*temp.T)/temp.trace()
	p_alpha = np.trace(H)/size
	print("Palser mu: ", str(p_alpha))
	scaled_H = p_alpha*identity-H
	p_alpha_ex = linalg.funm(scaled_H, lambda _: np.heaviside(_.real, 0.5))

	alpha = cp_dmm.get_mu()
	scaled_H = alpha*identity-H
	cphs_rho = linalg.funm(scaled_H, lambda _: np.heaviside(_.real, 0.5))
	print("Our mu: ", str(alpha))

	scaled_H = mu*identity - H
	hs_rho = linalg.funm(scaled_H, lambda _: np.heaviside(_.real, 0.5))

	#Plot exact solution
	plt.figure(1)
	plt.subplot(111)
	plt.ylabel('P_ij')
	plt.xlabel('|i-j|')
	plt.title("CP Comparisons")
	#plt.plot(exact_rho[0][:11], label='Exact')
	plt.plot(cp_dmm.rho[0][:11].real, 's', label='CP')
	plt.plot(palsercp_rho[0][:11].real, 'o', label='Palser CP')
	plt.plot(cphs_rho[0][:11].real, '*-', label='CP HS Exact')
	plt.plot(p_alpha_ex[0][:11].real, 'o-', label="Palser HS Exact")
	plt.plot(ntpoly_density[0][:11].real, 'k', label="NTPoly")
	#plt.plot(np.zeros(11), 'k')
	plt.legend(numpoints=1)

	plt.figure(2)
	plt.subplot(111)
	plt.ylabel('P_ij')
	plt.xlabel('|i-j|')
	plt.title('GCP Comparisons')
	plt.plot(gcp_dmm.rho[0][:11].real, label='GCP')
	plt.plot(palsergcp_rho[0][:11].real, 'o', label='Palser GCP')
	plt.plot(hs_rho[0][:11].real, '*-', label='GCP HS Exact')
	plt.legend(numpoints=1)
	
	plt.figure(3)
	plt.title("CP Eigs")
	plt.xlabel("Energy")
	plt.ylabel("Population")
	plt.plot(cp_dmm.E, cp_eigs[::-1], '*-', label='CP')
	plt.plot(cp_dmm.E, pcp_eigs[::-1], '^', label='Palser CP')
	plt.plot(cp_dmm.E, ntcp_eigs[::-1], 'o-', label='NTPoly CP')
	plt.plot(cp_dmm.E, cp_dmm.get_exact_pop(mu=alpha.real), 'k', label='Exact')
	plt.legend(numpoints=1)
	
	plt.figure(4)
	plt.title("GCP Eigs")
	plt.xlabel("Energy")
	plt.ylabel("Population")
	plt.plot(gcp_dmm.E, gcp_eigs[::-1], '*-', label='GCP')
	plt.plot(gcp_dmm.E, pgcp_eigs[::-1], '^', label='Palser GCP')
	plt.plot(gcp_dmm.E, gcp_dmm.get_exact_pop(), 'k', label='Exact')
	plt.legend(numpoints=1)
	plt.show()
	
