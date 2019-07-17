# import standard modules
import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse
from scipy.sparse import rand
import subprocess
import sys
import os.path

SAVE_PATH = "/home/jacob/Documents/DMM/Sampling_Results/"

# NT Poly
from NTPoly.Build.python import NTPolySwig as NT

# MPI Module
from mpi4py import MPI
comm = MPI.COMM_WORLD

def saveResults(filename, matrix):
	"""
	Function for saving density matrices to a file using np.savez
	"""
	complete_path = os.path.join(SAVE_PATH, filename)
	print(type(matrix[0]))
	np.savez(complete_path, *matrix)

def getNorm(filename):
	"""
	Function for taking the norms of the density matrices saved by np.savez
	"""
	complete_path = os.path.join(SAVE_PATH, filename)
	matrices = np.load(complete_path)
	print(matrices.files)
	norms = [np.linalg.norm(matrices['arr_' + str(i)]) for i in range(len(matrices.files))]
	return norms
		

if __name__ == "__main__":

	#set up mpi
	rank = comm.Get_rank()
	total_processors = comm.Get_size()

	#process input parameters
	for i in range(1, len(sys.argv), 2):
		argument = sys.argv[i]
		argument_value = sys.argv[i + 1]
		if argument == '--convergence_threshold':
			convergence_threshold = float(argument_value)
		elif argument == '--density_file':
			density_file_out = argument_value
		elif argument == '--threshold':
			threshold = float(argument_value)
		elif argument == '--process_rows':
			process_rows = int(argument_value)
		elif argument == '--process_columns':
			process_columns = int(argument_value)
		elif argument == '--process_slices':
			process_slices = int(argument_value)
		elif argument == '--rows':
			rows = int(argument_value)
		elif argument == '--density':
			#i.e. density of matrix
			density = float(argument_value)
		elif argument == '--number_of_electrons':
			number_of_electrons = int(argument_value)
	
	#Save density matrices in coo format
	num_runs = 2
	ntpoly_densities = []
	scipy_densities = []
	zvode_densities = []	

	i = 0
	while i < num_runs:
		#Construct process grid
		NT.ConstructGlobalProcessGrid(process_rows, process_columns, process_slices)
	
		#Setup solver parameters
		solver_parameters = NT.SolverParameters()
		solver_parameters.SetConvergeDiff(convergence_threshold)
		solver_parameters.SetThreshold(threshold)
		solver_parameters.SetVerbosity(False)
	
		#Run the ScipyMatrixGenerator script to generate a random hamiltonian of size rows x rows
		#Also construct the overlap matrix
		subprocess.run(["python3", "ScipyMatrixGenerator.py", '--rows', str(rows), '--density', str(density)])
		#hamiltonian = mmread("hamiltonian.mtx").toarray()
		overlap = sparse.identity(rows, format='coo', dtype='complex')
		mmwrite("overlap", overlap)
		ntpoly_hamiltonian = NT.Matrix_ps("hamiltonian.mtx")
		ntpoly_overlap = NT.Matrix_ps("overlap.mtx")
		Density = NT.Matrix_ps(ntpoly_hamiltonian.GetActualDimension())
		
		#Compute the density matrix
		energy_value, chemical_potential = \
			NT.DensityMatrixSolvers.TRS2(ntpoly_hamiltonian, ntpoly_overlap, number_of_electrons, Density, solver_parameters)

		#Output density matrix
		Density.WriteToMatrixMarket(density_file_out)
		ntpoly_hamiltonian.WriteToMatrixMarket("test.mtx")
		ntpoly_density = mmread(density_file_out)
		ntpoly_densities.append(ntpoly_density.toarray())
		NT.DestructGlobalProcessGrid()


		#Compute rho from H(mu*I - hamiltonian) where H is the heaviside function using scipy
		subprocess.run(["python3", "MatrixFunction.py", '--hamiltonian', 'hamiltonian.mtx', 
				'--chemical_potential', str(chemical_potential), '--rows', str(rows)])
		scipy_density = mmread("scipy_density.mtx")
		scipy_densities.append(scipy_density.toarray())

		#Run zvode and obtain its density matrix
		subprocess.run(["python3", "zvode_example.py", '--hamiltonian', 'hamiltonian.mtx',
				'--chemical_potential', str(chemical_potential), '--rows', str(rows)])
		zvode_density = mmread("zvode_density.mtx")
		zvode_densities.append(zvode_density.toarray())

		"""
		#Compare the results
		subprocess.run(["python3", "plot.py", "--hamiltonian", 'hamiltonian.mtx'])
		"""
		
		i += 1
		

	#Save scipy results
	scipy_filename = "scipy_density_" + str(rows) + "_" + str(number_of_electrons) + ".npz"
	saveResults(scipy_filename, scipy_densities)
	
	#Save zvode results
	zvode_filename = "zvode_density_" + str(rows) + "_" + str(number_of_electrons) + ".npz"
	saveResults(zvode_filename, zvode_densities)
	
	#Save NTPoly results
	ntpoly_filename = "ntpoly_density_" + str(rows) + "_" + str(number_of_electrons) + ".npz"
	saveResults(ntpoly_filename, ntpoly_densities)

	
	#Pull data from .npz files and analyze them
	scipy_norms = getNorm(scipy_filename)
	print(scipy_norms)

	zvode_norms = getNorm(zvode_filename)
	print(zvode_norms)

	ntpoly_norms = getNorm(scipy_filename)
	print(ntpoly_norms)
