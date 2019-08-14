import numpy as np
from scipy import sparse
from numba import njit
import matplotlib.pyplot as plt
from dmm import DMM
from cp_dmm import CP_DMM
from gcp_dmm import GCP_DMM


if __name__ == '__main__':
	#Set up intial parameters
	size=100
	diags = [1, np.ones(size), np.ones(size), 1]
	offsets = [-size+1, -1, 1, size-1]
	beta = 0.0
	dbeta = 0.003
	nsteps = 1000
	num_electrons = 5
	mu = 0.0
	
	#############################################################
	#
	#	The general form the Hamiltonian is:
	#
	#		0	1	0	...	0	c
	#		1	0	c	...	0	0
	#		0	c	0	...	0	0
	#		.					.
	#		.					.	
	#		0					1
	#		c	0	0	...	1	0
	#
	#	For metallic systems, c = 1
	#	For insulator systems, c = 0.5
	#
	##############################################################

	#Create the classes for the metallic system
	metal_hamiltonian = sparse.diags(diags, offsets, format='coo', dtype=complex)
	metal_hamiltonian = metal_hamiltonian.toarray()
	cp_metal = CP_DMM(H=metal_hamiltonian, dbeta=dbeta, num_electrons=num_electrons)
	gcp_metal = GCP_DMM(H=metal_hamiltonian, dbeta=dbeta, mu=mu)
	
	#Use the zvode methods to obtain the final density matrix and then get the anti-diagonal
	cp_metal_rho = cp_metal.zvode(nsteps)
	cp_metal_anti_diag = [cp_metal_rho[size-1-j][j].real for j in range(size)]
	
	gcp_metal_rho = gcp_metal.zvode(nsteps)
	gcp_metal_anti_diag = [gcp_metal_rho[size-1-j][j].real for j in range(size)]

	#Create an array containing the exact solution for metallic systems, which is sinc(|i-j|)
	arg = np.zeros(size)
	for i in range(size):
		arg[i] = size-2*i
	metal_sol = np.sinc(np.abs(arg/np.pi))

	#Create the classes for the insulator system
	diags[0] = 0.5
	diags[3] = 0.5
	for i in range(size):
		if i%2 != 0:
			diags[1][i] = 0.5
			diags[2][i] = 0.5
	insul_hamiltonian = sparse.diags(diags, offsets, format='coo', dtype=complex).toarray()
	cp_insul = CP_DMM(H=insul_hamiltonian, dbeta=dbeta, num_electrons=num_electrons)
	gcp_insul = GCP_DMM(H=insul_hamiltonian, dbeta=dbeta, mu=mu)
	
	#Use the zvode methods to obtain the final density matrices and then get their anti-diagonals
	cp_insul_rho = cp_insul.zvode(nsteps)
	cp_insul_anti_diag = [cp_insul_rho[size-1-j][j].real for j in range(size)]
	
	gcp_insul_rho = gcp_insul.zvode(nsteps)
	gcp_insul_anti_diag = [gcp_insul_rho[size-i-j][j].real for j in range(size)]

	#Create an array containing the exact solution for insulator systems, which is exp(-a|i-j|)
	insul_sol = np.exp(-np.abs(arg))
	
	#Plot the decay along the metallic density matrix anti-diagonals and compare with the exact solution
	plt.figure(1)
	plt.subplot(121)
	plt.title("CP Metal Decay")
	plt.ylabel("P_ij")
	plt.xlabel("j")
	plt.plot(cp_metal_anti_diag, label='CP')
	plt.plot(metal_sol, label='Exact Sol.')
	plt.legend(numpoints=1)

	plt.subplot(122)
	plt.title("GCP Metal Decay")
	plt.xlabel("j")
	plt.plot(gcp_metal_anti_diag, label='GCP')
	plt.plot(metal_sol, label='Exact Sol.')
	plt.legend(numpoints=1)
	
	#Plot the decay along the insulator density matrix anti-diagonals and compare with the exact solution
	plt.figure(2)
	plt.subplot(121)
	plt.title("CP Insulator Decay")
	plt.ylabel("P_ij")
	plt.xlabel("j")
	plt.plot(cp_insul_anti_diag, label='CP')
	plt.plot(insul_sol, label='Exact Sol.')
	plt.legend(numpoints=1)

	plt.subplot(122)
	plt.title("GCP Insulator Decay")
	plt.xlabel("j")
	plt.plot(gcp_insul_anti_diag, label='GCP')
	plt.plot(insul_sol, label='Exact Sol.')
	plt.legend(numpoints=1)

	plt.show()
	
