#import standard modules
import numpy as np
from scipy import sparse
from scipy.io import mmwrite, mmread
import matplotlib.pyplot as plt
import sys
import subprocess
sys.settrace

#Plot rho[i][j] vs j for each
#Scipy
scipy_density = mmread("scipy_density.mtx").toarray()
rows = scipy_density[0].size - 1
rhos = [np.linalg.norm(scipy_density[rows-j][j]) for j in range(rows)]
plt.subplot(131)
plt.title("Scipy Decay")
plt.xlabel("j")
plt.ylabel("P[i][j]")
plt.plot(rhos)

#Zvode
zvode_density = mmread("zvode_density.mtx").toarray()
rhos = [np.linalg.norm(zvode_density[rows-j][j]) for j in range(rows)]
plt.subplot(132)
plt.title("Zvode Decay")
plt.xlabel("j")
plt.plot(rhos)

#NTPoly
ntpoly_density = mmread("density.mtx").toarray()
rhos = [np.linalg.norm(ntpoly_density[rows-j][j]) for j in range(rows)]
plt.subplot(133)
plt.title("NTPoly Decay")
plt.xlabel("j")
plt.plot(rhos)
plt.show()
