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
import sys

sys.path.insert(1, '/home/jacob/Documents/DMM')

# NT Poly
from NTPoly.Build.python import NTPolySwig as NT

# MPI Module
from mpi4py import MPI
comm = MPI.COMM_WORLD

if __name__ == '__main__':
	#Set up Hamiltonian as given by simple_dft.py and saved in fock.mtx
	H = mmread('fock.mtx').toarray()
	size = H.shape[0]

	#Set up the overlap matrix as given by simple_dft.py and saved in dft_overlap.mtx
	ovlp = mmread('dft_overlap.mtx').toarray()

	#Read in the dft density matrix
	dft_density = mmread('dft_density.mtx').toarray()
	
	#Read in the dft chemical potential
	f = open('dft_mu.txt', 'r')
	dft_mu = float(f.read())

	#Set up parameters for NTPoly
	convergence_threshold = 1e-10
	threshold = 1e-10
	process_rows = 1
	process_columns = 1
	process_slices = 1
	hamiltonian_file = "fock.mtx"
	density_file = "ntpoly_density.mtx"
	num_electrons = int(dft_density.trace())
	print(num_electrons)

	#Construct process grid
	NT.ConstructGlobalProcessGrid(process_rows, process_columns, process_slices)
	
	#Setup solver parameters
	solver_parameters = NT.SolverParameters()
	solver_parameters.SetConvergeDiff(convergence_threshold)
	solver_parameters.SetThreshold(threshold)
	solver_parameters.SetVerbosity(False)

	ntpoly_hamiltonian = NT.Matrix_ps("fock.mtx")
	ntpoly_overlap = NT.Matrix_ps("dft_overlap.mtx")
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


	#Implement classes and run zvode
	dbeta = 0.003
	nsteps = 1000

	cp_dmm = CP_DMM(H=H, dbeta=dbeta, num_electrons=num_electrons)
	cp_dmm.zvode(nsteps)
	cp_dmm.purify()

	gcp_dmm = GCP_DMM(H=H, dbeta=dbeta, mu=dft_mu)
	gcp_dmm.zvode(nsteps)
	gcp_dmm.purify()
	
	#Plot density matrices
	plt.figure(1)
	
	plt.subplot(121)
	plt.title('DFT Density')
	plt.ylabel('j')
	plt.xlabel('i')
	plt.imshow(dft_density)
	
	plt.subplot(122)
	plt.xlabel('i')
	plt.title('NTPoly Density')
	plt.imshow(ntpoly_density)
	
	plt.gcf().subplots_adjust(right=0.8)
	cbax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
	plt.colorbar(cax=cbax)
	plt.show()
