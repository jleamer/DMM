#import standard modules
import numpy as np
from scipy import sparse
from scipy.sparse import rand
from scipy.io import mmwrite, mmread
import matplotlib.pyplot as plt
import sys
import subprocess
sys.settrace

# NT Poly
from NTPoly.Build.python import NTPolySwig as NT

# MPI Module
from mpi4py import MPI
comm = MPI.COMM_WORLD

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
		elif argument == '--metal_density_file':
			metal_density_file = argument_value
		elif argument == '--insul_density_file':
			insul_density_file = argument_value
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

	#create density file names for each case
	metal_ntpoly = "ntpoly_" + metal_density_file + ".mtx"
	metal_scipy = "scipy_" + metal_density_file + ".mtx"
	metal_zvode = "zvode_" + metal_density_file + ".mtx"
	insul_ntpoly = "ntpoly_" + insul_density_file + ".mtx"
	insul_scipy = "scipy_" + insul_density_file + ".mtx"
	insul_zvode = "zvode_" + insul_density_file + ".mtx"

	#generate our huckel theory hamiltonian for a metal
	#this is a 1D ring where alpha - epsilon is set to 0 and beta is set to 1
	diags = [1 for i in range(rows)]
	
	metal_hamiltonian = sparse.diags([[1], diags, diags, [1]], [-99, -1, 1, 99], format="coo", dtype='complex')
	test = metal_hamiltonian.toarray()
	print(test)
	#hamiltonian = rand(rows, rows, density=density) + 1j*rand(rows, rows, density=density)
	#hamiltonian += hamiltonian.conj().T
	mmwrite("metal_hamiltonian.mtx", metal_hamiltonian)

	#Run NTPoly on our Hamiltonian
	#Construct process grid
	NT.ConstructGlobalProcessGrid(process_rows, process_columns, process_slices)
	
	#Setup solver parameters
	solver_parameters = NT.SolverParameters()
	solver_parameters.SetConvergeDiff(convergence_threshold)
	solver_parameters.SetThreshold(threshold)
	solver_parameters.SetVerbosity(False)
	
	overlap = sparse.identity(rows, format='coo', dtype='complex')
	mmwrite("overlap", overlap)
	ntpoly_hamiltonian = NT.Matrix_ps("hamiltonian.mtx")
	ntpoly_overlap = NT.Matrix_ps("overlap.mtx")
	Density = NT.Matrix_ps(ntpoly_hamiltonian.GetActualDimension())
		
	#Compute the density matrix
	energy_value, chemical_potential = \
		NT.DensityMatrixSolvers.TRS2(ntpoly_hamiltonian, ntpoly_overlap, number_of_electrons, Density, solver_parameters)
	print(chemical_potential)

	#Output density matrix
	Density.WriteToMatrixMarket(metal_ntpoly)
	NT.DestructGlobalProcessGrid()

	#Compute rho from H(mu*I - hamiltonian) where H is the heaviside function using scipy
	subprocess.run(["python3", "MatrixFunction.py", '--hamiltonian', 'hamiltonian.mtx', 
			'--chemical_potential', str(chemical_potential), '--rows', str(rows), '--density_file_out', metal_scipy])
	
	#Run zvode and obtain its density matrix
	subprocess.run(["python3", "zvode_example.py", '--hamiltonian', 'hamiltonian.mtx',
			'--chemical_potential', str(chemical_potential), '--rows', str(rows), '--density_file_out', metal_zvode])

	

	#generate huckel theory hamiltonian for an insulator
	diags = []
	for i in range(rows):
		if i % 2 == 0:
			diags.append(1)
		else:
			diags.append(0.5)
	insul_hamiltonian = sparse.diags([[0.5], diags, diags, [0.5]], [-100, -1, 1, 100], format='coo', dtype='complex')

	test = insul_hamiltonian.toarray()
	print(test)
	#hamiltonian = rand(rows, rows, density=density) + 1j*rand(rows, rows, density=density)
	#hamiltonian += hamiltonian.conj().T
	mmwrite("insul_hamiltonian.mtx", insul_hamiltonian)

	#Run NTPoly on our Hamiltonian
	#Construct process grid
	NT.ConstructGlobalProcessGrid(process_rows, process_columns, process_slices)
	
	#Setup solver parameters
	solver_parameters = NT.SolverParameters()
	solver_parameters.SetConvergeDiff(convergence_threshold)
	solver_parameters.SetThreshold(threshold)
	solver_parameters.SetVerbosity(False)
	
	overlap = sparse.identity(rows, format='coo', dtype='complex')
	mmwrite("overlap", overlap)
	ntpoly_hamiltonian = NT.Matrix_ps("hamiltonian.mtx")
	ntpoly_overlap = NT.Matrix_ps("overlap.mtx")
	Density = NT.Matrix_ps(ntpoly_hamiltonian.GetActualDimension())
		
	#Compute the density matrix
	energy_value, chemical_potential = \
		NT.DensityMatrixSolvers.TRS2(ntpoly_hamiltonian, ntpoly_overlap, number_of_electrons, Density, solver_parameters)
	print(chemical_potential)

	#Output density matrix
	Density.WriteToMatrixMarket(insul_ntpoly)
	NT.DestructGlobalProcessGrid()

	#Compute rho from H(mu*I - hamiltonian) where H is the heaviside function using scipy
	subprocess.run(["python3", "MatrixFunction.py", '--hamiltonian', 'hamiltonian.mtx', 
			'--chemical_potential', str(chemical_potential), '--rows', str(rows), '--density_file_out', insul_scipy])
	
	#Run zvode and obtain its density matrix
	subprocess.run(["python3", "zvode_example.py", '--hamiltonian', 'hamiltonian.mtx',
			'--chemical_potential', str(chemical_potential), '--rows', str(rows), '--density_file_out', insul_zvode])
	
