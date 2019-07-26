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

def saveResults(filename, matrix):
	"""
	Function for saving density matrices to a file using np.savez
	Params:
		filename - string containing the name of the archive
		matrix - list of density matrices that are to be saved
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
	Params:
		filename - string containing the name of the archive
	Returns:
		norms - a list of the norms rho^2-rho for each matrix in the archive
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
	Params:
		filename - string containing the name of the archive
		hamiltonian_file - the name of the hamiltonian file to be used in energy calculations
	Returns:
		energy - a list of the energies (i.e. Tr(rho*H)) calculated for each density matrix in the archive
	"""
	hamiltonian = mmread(hamiltonian_file).toarray()
	complete_path = os.path.join(SAVE_PATH, filename)
	matrices = np.load(complete_path)
	energies = [matrices['arr_' + str(i)].dot(hamiltonian).trace() for i in range(len(matrices.files))]
	return energies


#process input parameters
for i in range(1, len(sys.argv), 2):
	argument = sys.argv[i]
	argument_value = sys.argv[i + 1]
	print(argument + ", " + argument_value)
	if argument == '--scipy_archive':
		scipy_archive = argument_value
	elif argument == '--zvode_archive':
		zvode_archive = argument_value
	elif argument == '--ntpoly_archive':
		ntpoly_archive = argument_value

#Pull data from .npz files and analyze them
#Specifically, we want to plot norm of rho^2-rho vs. energy
scipy_norms = getNorm(scipy_archive)
scipy_energies = getEnergy(scipy_archive, 'hamiltonian.mtx')

zvode_norms = getNorm(zvode_archive)
zvode_energies = getEnergy(zvode_archive, 'hamiltonian.mtx')
	
ntpoly_norms = getNorm(ntpoly_archive)
ntpoly_energies = getEnergy(ntpoly_archive, 'hamiltonian.mtx')	

#Plot the norms vs energy
plt.subplot(131)
plt.title("Scipy")
plt.scatter(scipy_energies, scipy_norms)
plt.ylim([-0.1, 0.5])
plt.xlabel("Energy")
plt.ylabel("Norm")

plt.subplot(132)
plt.title("Zvode")
plt.scatter(zvode_energies, zvode_norms)
plt.ylim([-0.1, 0.5])
plt.xlabel("Energy")
#plt.ylabel("Norm")

plt.subplot(133)
plt.title("NTPoly")
plt.scatter(ntpoly_energies, ntpoly_norms)
plt.ylim([-0.1, 0.5])
plt.xlabel("Energy")
#plt.ylabel("Norm")

plt.show()
