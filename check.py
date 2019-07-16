import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse

scipy_density = mmread("scipy_density")
scipy_density = scipy_density.toarray()
ntpoly_density = mmread("density")
ntpoly_density = ntpoly_density.toarray()
zvode_density = mmread("zvode_matrix")
zvode_density = zvode_density.toarray()


print("Norms:")
print("	NTPoly: ", np.linalg.norm(ntpoly_density))
print("	Scipy: ", np.linalg.norm(scipy_density))
print("	Zvode: ", np.linalg.norm(zvode_density))

absolute_error = np.abs(np.linalg.norm(scipy_density) - np.linalg.norm(ntpoly_density))
print("Scipy and NTPoly Comparison:")
print("	Absolute error b/w norms: ", absolute_error)
print("	Relative error b/w norms: ", absolute_error/np.linalg.norm(scipy_density))

zvode_nt_abs = np.abs(np.linalg.norm(zvode_density) - np.linalg.norm(ntpoly_density))
print("Zvode and NTPoly Comparison:")
print("	Absolute error b/w norms: ", zvode_nt_abs)
print("	Relative error b/w norms: ", zvode_nt_abs/np.linalg.norm(ntpoly_density))

scipy_zvode_abs = np.abs(np.linalg.norm(zvode_density) - np.linalg.norm(scipy_density))
print("Zvode and Scipy Comparison:")
print("	Absolute error b/w norms: ", scipy_zvode_abs)
print("	Relative error b/w norms: ", scipy_zvode_abs/np.linalg.norm(scipy_density))
