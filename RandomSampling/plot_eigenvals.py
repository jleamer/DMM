import matplotlib.pyplot as plt
import numpy as np


def getEigenvalues(file_path):
	densities = np.load(file_path)
	eigenvals = np.zeros(densities['arr_0'][0].size, dtype='complex')
	print(np.linalg.eigvalsh(densities['arr_0']))
	for i in range(len(densities.files)):
		eigenvals += np.linalg.eigvalsh(densities['arr_' + str(i)])
	eigenvals /= len(densities.files)
	return eigenvals

scipy_file_path = "/home/jacob/Documents/DMM/Sampling_Results/Scipy/scipy_density_10_5.npz"
zvode_file_path = "/home/jacob/Documents/DMM/Sampling_Results/Zvode/zvode_density_10_5.npz"
ntpoly_file_path = "/home/jacob/Documents/DMM/Sampling_Results/NTPoly/ntpoly_density_10_5.npz"

scipy_eigvals = getEigenvalues(scipy_file_path)
zvode_eigvals = getEigenvalues(zvode_file_path)
ntpoly_eigvals = getEigenvalues(ntpoly_file_path)

plt.subplot(131)
plt.title("Avg. Scipy Eigenvalues")
plt.ylabel("Population")
plt.xlabel("Energy")
