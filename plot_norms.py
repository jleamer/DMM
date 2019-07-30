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

def getNorm(filename, save_path):
	"""
	Function for taking the norms of the rho^2-rho where rho was saved by np.savez
	Params:
		filename - string containing the name of the archive
	Returns:
		norms - a list of the norms rho^2-rho for each matrix in the archive
	"""
	complete_path = os.path.join(save_path, filename)
	matrices = np.load(complete_path)
	print(len(matrices.files))
	diff = [matrices['arr_' + str(i)].dot(matrices['arr_' + str(i)]) - matrices['arr_' + str(i)] for i in range(len(matrices.files))]
	norms = [np.linalg.norm(diff[i]) for i in range(len(matrices.files))]
	return norms
		
def getEnergy(filename, hamiltonians, save_path):
	"""
	Function for obtaining the energies of the density matrices saved by np.saves
	Params:
		filename - string containing the name of the archive
		hamiltonians - a list of the hamiltonians to be used in energy calculations
	Returns:
		energy - a list of the energies (i.e. Tr(rho*H)) calculated for each density matrix in the archive
	"""
	
	complete_path = os.path.join(save_path, filename)
	matrices = np.load(complete_path)
	energies = [matrices['arr_' + str(i)].dot(hamiltonians['arr_' + str(i)]).trace() for i in range(len(matrices.files))]
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
	elif argument == '--hamiltonian_archive':
		hamiltonian_archive = argument_value

#Pull data from .npz files and analyze them
#Specifically, we want to plot norm of rho^2-rho vs. energy
#Start with loading hamiltonians
save_path = os.path.join("/home/jacob/Documents/DMM/Sampling_Results/Hamiltonian", hamiltonian_archive)
hamiltonians = np.load(save_path)

save_path = "/home/jacob/Documents/DMM/Sampling_Results/Scipy"
scipy_norms = getNorm(scipy_archive, save_path)
scipy_energies = getEnergy(scipy_archive, hamiltonians, save_path)

save_path = "/home/jacob/Documents/DMM/Sampling_Results/Zvode"
zvode_norms = getNorm(zvode_archive, save_path)
zvode_energies = getEnergy(zvode_archive, hamiltonians, save_path)
	
save_path = "/home/jacob/Documents/DMM/Sampling_Results/NTPoly"
ntpoly_norms = getNorm(ntpoly_archive, save_path)
ntpoly_energies = getEnergy(ntpoly_archive, hamiltonians, save_path)	

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
