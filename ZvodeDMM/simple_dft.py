from pyscf import gto, dft
import numpy
from scipy.io import mmwrite, mmread
from scipy import sparse
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from ZvodeDMM.dmm import DMM
from ZvodeDMM.gcp_dmm import GCP_DMM
from ZvodeDMM.cp_dmm import CP_DMM

# import things for NTPoly
from NTPoly.Build.python import NTPolySwig as NT
from mpi4py import MPI
comm = MPI.COMM_WORLD


def setup_animation():
    im.set_array(numpy.zeros((11, 11)))
    return im

def self_consistent_Aiken(h1e, rho, inv_ovlp, mu, beta, cc, nsteps, ovlp):

    def single_step(rho_):
        h = h1e + cc*rho_ @ rho_
        arg = inv_ovlp@h
        identity = numpy.identity(rho.shape[0])
        return ovlp @ linalg.funm(arg, lambda _: 1/(1+numpy.exp(beta*(_ - mu))))
        #return ovlp @ linalg.inv(identity + linalg.expm(beta*(arg-mu*identity)))


    norm_diff = []
    rho_list = numpy.zeros((nsteps, 11, 11))

    rho_0 = rho.copy()
    for i in range(nsteps):
        print(i)
        rho_list[i] = rho_0.copy()
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0)
        rho_2 = single_step(rho_1)

        #aitken_rho = (rho_0 @ rho_2 - rho_1 @ rho_1) @ linalg.pinv(rho_0 + rho_2 - 2*rho_1)
        aitken_rho = rho_2 - (rho_2 - rho_1) @ (rho_2 - rho_1)  @ linalg.pinv((rho_2 - rho_1) - (rho_1 - rho_0))
        rho_0 = aitken_rho

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

    return norm_diff, rho_list
'''
A simple example to run DFT calculation.
'''

# PBE Hydrogen flouride in a 6-31G basis set.
mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '631g',
    symmetry = True,
)

mf = dft.RKS(mol)
#mf.xc = 'blyp' # shorthand for b88,lyp
mf.xc = 'pbe' # shorthand for pbe,pbe
#mf.xc = 'lda,vwn_rpa'
#mf.xc = 'pbe0'
#mf.xc = 'b3lyp'

# this where self-content diagonalization happens
mf.kernel()

# Orbital energies, Mulliken population etc.
mf.analyze()

# Get the converged density matrix (it generates the density matrix)
dm = mf.make_rdm1()
mmwrite('dft_density.mtx', sparse.coo_matrix(dm))


# Get the nuclear-nuclear repulsion energy
e_nuc = mf.energy_nuc()
# Get the 'core' hamiltonian, corresponding to kinetic energy and e-nuclear repulsion terms
h1e = mf.get_hcore()
e1 = numpy.einsum('ij,ji', h1e, dm) # Compute the kinetic + e-nuclear repulsion energy
# Get the kohn-sham potential, including the Hartree coulomb repulsion and exchange-correlation potential, integrated on a grid
vhf = mf.get_veff(mf.mol, dm)
#print(type(vhf))
#print(vhf)
# Total energy
tot_e = e1 + vhf.ecoul + vhf.exc + e_nuc    # Total energy is sum of terms
print('Total dft energy: {}'.format(tot_e))

# chemical potential
index = int(mol.nelectron/2)
mu = (mf.mo_energy[index] + mf.mo_energy[index - 1]) / 2.
print('Chemical Potential: ', str(mu))
f = open('dft_mu.txt', 'w+')
f.write(str(mu))
f.close()

# get the overlap matrix and print to file
ovlp = mf.get_ovlp()
mmwrite('dft_overlap.mtx', sparse.coo_matrix(ovlp))

# Full fock matrix is sum of h1e and vhf
fock = h1e + vhf

# Get whole fock matrix directly corresponding to this density, without computing individual components
fock_direct = mf.get_fock(dm=dm)

# Check that ways to get the fock matrix are the same
assert(numpy.allclose(fock_direct,fock))

#Write fock matrix to file
mmwrite('fock', sparse.coo_matrix(fock))

print("DFT trace: ", dm.trace())

# Now calculate density matrix using NTPoly
# first write h1e to hamiltonian file for later use
mmwrite('core_hamiltonian.mtx', sparse.coo_matrix(h1e))

# set up parameters for NTPoly solvers
convergence_threshold = 1e-10
threshold = 1e-10
process_rows = 1
process_columns = 1
process_slices = 1
hamiltonian_file = 'core_hamiltonian.mtx'
density_file = 'NT_density.mtx'

# construct process grid
NT.ConstructGlobalProcessGrid(process_rows, process_columns, process_slices)

# Set up solver parameters
solver_parameters = NT.SolverParameters()
solver_parameters.SetConvergeDiff(convergence_threshold)
solver_parameters.SetThreshold(threshold)
solver_parameters.SetVerbosity(False)

NT_hamiltonian = NT.Matrix_ps(hamiltonian_file)
NT_ovlp = NT.Matrix_ps('dft_overlap.mtx')
density = NT.Matrix_ps(NT_hamiltonian.GetActualDimension())

# compute the density matrix
energy_value, chemical_potential = NT.DensityMatrixSolvers.TRS2(NT_hamiltonian, NT_ovlp, 10, density, solver_parameters)
print("NTPoly chemical potential: ", chemical_potential)

# output density matrix
density.WriteToMatrixMarket(density_file)
NT_density = mmread(density_file).toarray()
ntcp_eigs = linalg.eigvalsh(NT_density)
NT.DestructGlobalProcessGrid()
print("NTPoly num electrons: ", NT_density.trace())


#Now we want to test that our DMM method arrives at the same density matrix
# first get the inverse overlap matrix
inv_ovlp = linalg.inv(ovlp)

# propagate using GCP DMM zvode
gcp = GCP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, mu=mu, mf=mf)
gcp.no_zvode(1000)
print("GCP Zvode trace: ", gcp.rho.trace())
#gcp.purify()

#propagate using GCP DMM rk4
gcp2 = GCP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, mu=mu, mf=mf)
gcp2.non_orth_rk4(1000)
print("GCP RK4 trace: ", gcp2.rho.trace())
#gcp2.purify()

fig1 = plt.figure(1)
plt.subplot(111)
plt.plot(gcp.num_electrons, 'r-', label='Zvode')
plt.plot(gcp2.num_electrons, 'b--', label='RK4')
plt.title("Tr(P) for GCP")
plt.xlabel('iteration #')
plt.ylabel('Tr(P)')
plt.legend(numpoints=1)

# get exact DM
P = ovlp@linalg.pinv(gcp.identity + linalg.expm(gcp.beta*(inv_ovlp@gcp.H-gcp.mu*gcp.identity)))
print("Exact trace: ", P.trace())

print("FD GCP norm: ", linalg.norm(P))
print("GCP Zvode norm: ", linalg.norm(P-gcp.rho))
print("GCP RK4 norm: ", linalg.norm(P-gcp2.rho))

# Plot each
gcp_title = [["Zvode", "RK4"], ["Exact Expression", "DFT"]]
gcp_data = [[gcp.rho.real, gcp2.rho.real], [P.real, dm.real]]
vmax = numpy.amax(dm)
vmin = numpy.amin(dm)

fig2, axes = plt.subplots(2, 2)
for col in range(2):
    for row in range(2):
        ax = axes[row, col]
        im = ax.imshow(gcp_data[row][col], origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(gcp_title[row][col])
        ax.set_xlabel("i")
        ax.set_ylabel("j")
fig2.colorbar(im, ax=axes)


# Repeat for cp case
cp = CP_DMM(H=h1e, dbeta=0.0003, ovlp=ovlp, num_electrons=dm.trace())
cp.no_zvode(1000)
#cp.purify()
print("CP Zvode trace: ",cp.rho.trace())
print("CP Zvode chemical potential: ", cp.no_get_mu())

cp2 = CP_DMM(H=h1e, dbeta=0.0003, ovlp=ovlp, num_electrons=dm.trace())
cp2.non_orth_rk4(1000)
#cp2.purify()
print("CP RK4 trace: ", cp2.rho.trace())
print("CP RK4 chemical potential: ", cp2.no_get_mu())

exact2 = ovlp@linalg.pinv(cp.identity + linalg.expm(cp.beta*(inv_ovlp@cp.H-cp.no_get_mu()*cp.identity)))
print("CP TF trace: ", exact2.trace())
print("FD CP norm: ", linalg.norm(exact2))
print("CP RK4 norm: ", linalg.norm(cp2.rho - exact2))
print("CP Zvode norm: ", linalg.norm(cp.rho - exact2))

fig3 = plt.figure(3)
plt.subplot(121)
plt.plot(cp.mu_list, 'r-', label='Zvode')
plt.plot(cp2.mu_list, 'b--', label='RK4')
plt.title("O u")
plt.xlabel('iteration #')
plt.ylabel('\mu')
plt.legend(numpoints=1)
plt.subplot(122)
plt.plot(cp.no_mu_list, 'r-', label='Zvode')
plt.plot(cp2.no_mu_list, 'b--', label='RK4')
plt.title("Non-O u")
plt.xlabel('iteration #')
plt.ylabel('\mu')
plt.legend(numpoints=1)

cp_titles = [['Zvode', 'RK4', 'Exact Expression'], ['DFT', 'NTPoly', 'Identity']]
cp_data = [[cp.rho.real/cp.rho.trace().real, cp2.rho.real/cp2.rho.trace().real, exact2.real/exact2.trace().real], [dm.real/dm.trace().real, NT_density.real/NT_density.trace().real, cp.identity.real/cp.identity.trace().real]]
vmin = numpy.min(cp_data)
vmax = numpy.max(cp_data)
fig4, axes = plt.subplots(2, 3)
for col in range(3):
    for row in range(2):
        ax = axes[row, col]
        im = ax.imshow(cp_data[row][col], origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(cp_titles[row][col])
fig4.colorbar(im, ax=axes)

#print("Hexc: ", gcp.hexc[0])
#print("Hexc end: ", gcp.hexc[999])

# Self consistency checking
gcp3 = GCP_DMM(H=h1e, dbeta=0.003, ovlp=gcp.identity, mu=mu, mf=mf)
gcp3.no_zvode(900)

rho = gcp3.rho.copy()
nsteps = 100
norm_diff, rho_list = self_consistent_Aiken(h1e, rho, gcp3.inv_overlap, gcp3.mu, gcp3.beta, .1, nsteps, gcp3.ovlp)

fig5 = plt.figure(5)
plt.subplot(111)
plt.plot(norm_diff)
plt.title("Aitken's Convergence")
plt.ylabel("Norm difference")
plt.xlabel("Iteration number")


fig6, axes = plt.subplots(1, 1)
im = axes.imshow(rho_list[0].real, origin='lower', vmin=numpy.min(rho_list), vmax=numpy.max(rho_list))
fig6.colorbar(im, ax=axes)

def animate(i):
    im.set_array(rho_list[i].real)
    axes.set_title(str(i))
    return im

anim = ani.FuncAnimation(fig6, animate, init_func=setup_animation, frames=range(nsteps), interval=500, blit=False)

'''
for i in range(4):
    print(i+901)
    temp = gcp3.test_rhs(gcp3.dbeta, gcp3.rho, gcp3.y)
    gcp3.y = gcp3.calc_ynext(gcp3.beta, gcp3.rho, gcp3.H, gcp3.identity, gcp3.mu, gcp3.y)
    gcp3.rho = temp
    gcp3.beta += gcp3.dbeta
plt.subplot(122)
plt.imshow(gcp3.rho.real, origin='lower')
plt.title("SC check")
'''

plt.show()
