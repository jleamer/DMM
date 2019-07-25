# import standard modules
import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse
from scipy.sparse import rand
import subprocess
import sys
import os
import zipfile
import matplotlib.pyplot as plt

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
	if not os.path.exists(complete_path):
		archive = zipfile.ZipFile(complete_path, mode="w")
		for i in range(len(matrix)):
			"""
			Save a matrix to a .npy file and then write it to the zipfile
			Then delete the file that was made outside the zipfile
			"""
			filename = "arr_" + str(i) + ".npy"
			numpy_file = np.save(filename, matrix[i])
			archive.write(filename)
			os.remove(filename)
		archive.close()
	
	else:
		archive = zipfile.ZipFile(complete_path, mode="a")
		numfiles = len(archive.namelist())
		for i in range(len(matrix)):
			"""
			Save the matrices to .npy file and write each one to the zipfile
			The name of the files takes into account any files already there
			Then delete the file that was made outside the zipfile
			"""
			filename = "arr_" + str(i + numfiles) + ".npy"
			numpy_file = np.save(filename, matrix[i])
			archive.write(filename)
			os.remove(filename)
		archive.close()


def getNorm(filename):
	"""
	Function for taking the norms of the rho^2-rho where rho was saved by np.savez
	"""
	complete_path = os.path.join(SAVE_PATH, filename)
	matrices = np.load(complete_path)
	print(len(matrices.files))
	diff = [matrices['arr_' + str(i)].dot(matrices['arr_' + str(i)]) - matrices['arr_' + str(i)] for i in range(len(matrices.files))]
	norms = [np.linalg.norm(diff[i]) for i in range(len(matrices.files))]
	return norms
		
def getEnergy(filename, hamiltonian_file):
	"""
	Function for obtaining the energies of the density matrices saved by np.saves
	"""
	hamiltonian = mmread(hamiltonian_file).toarray()
	complete_path = os.path.join(SAVE_PATH, filename)
	matrices = np.load(complete_path)
	energies = [matrices['arr_' + str(i)].dot(hamiltonian).trace() for i in range(len(matrices.files))]
	return energies

def getAverage(filename):
	"""
	Function for obtaining the average of the density matrices in the file
	"""
	complete_path = os.path.join(SAVE_PATH, filename)
	matrices = np.load(complete_path)
	
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
	
	#Set up lists for storing the density matrix computed during each run
	num_runs = 10
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
		print(i)


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
	#Specifically, we want to plot norm of rho^2-rho vs. energy
	scipy_norms = getNorm(scipy_filename)
	print(scipy_norms)
	scipy_energies = getEnergy(scipy_filename, 'hamiltonian.mtx')
	print(scipy_energies)

	zvode_norms = getNorm(zvode_filename)
	print(zvode_norms)
	zvode_energies = getEnergy(zvode_filename, 'hamiltonian.mtx')
	print(zvode_energies)
	
	ntpoly_norms = getNorm(ntpoly_filename)
	print(ntpoly_norms)
	ntpoly_energies = getEnergy(ntpoly_filename, 'hamiltonian.mtx')
	print(ntpoly_energies)
	

	#Plot the norms vs energy
	plt.subplot(131)
	plt.title("Scipy")
	plt.scatter(scipy_energies, scipy_norms)
	plt.xlabel("Energy")
	plt.ylabel("Norm")

	plt.subplot(132)
	plt.title("Zvode")
	plt.scatter(zvode_energies, zvode_norms)
	plt.xlabel("Energy")
	#plt.ylabel("Norm")

	plt.subplot(133)
	plt.title("NTPoly")
	plt.scatter(ntpoly_energies, ntpoly_norms)
	plt.xlabel("Energy")
	#plt.ylabel("Norm")
	

	
	plt.show()
	
