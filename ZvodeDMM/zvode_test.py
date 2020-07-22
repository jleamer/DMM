from pyscf import gto, dft
import numpy as np
from scipy.io import mmwrite, mmread
from scipy.optimize import fixed_point
from scipy import sparse
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from ZvodeDMM.dmm import DMM
from ZvodeDMM.gcp_dmm import GCP_DMM
from ZvodeDMM.cp_dmm import CP_DMM
import numpy.ma as ma

def setup_animation():
    im.set_array(np.zeros((11, 11)))
    return im

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

# Compute the kinetic + e-nuclear repulsion energy
e1 = np.einsum('ij,ji', h1e, dm)

# Get the kohn-sham potential, including the Hartree coulomb repulsion and exchange-correlation potential, integrated on a grid
vhf = mf.get_veff(mf.mol, dm)

# Total energy
tot_e = e1 + vhf.ecoul + vhf.exc + e_nuc    # Total energy is sum of terms
print('Total dft energy: {}'.format(tot_e))

# chemical potential
index = int(mol.nelectron/2)
print(mol.nelectron)
mu = (mf.mo_energy[index] + mf.mo_energy[index - 1]) / 2.
print('Chemical Potential: ', str(mu))
f = open('dft_mu.txt', 'w+')
f.write(str(mu))
f.close()

print("mo_energy: ", mf.mo_energy)

# get the overlap matrix and print to file
ovlp = mf.get_ovlp()
mmwrite('dft_overlap.mtx', sparse.coo_matrix(ovlp))
fig1 = plt.figure(1)
ax11 = fig1.add_subplot(111)
im = ax11.imshow(ovlp.real, origin='lower')
ax11.set_title("real ovlp")
fig1.colorbar(im, ax=ax11)

# Full fock matrix is sum of h1e and vhf
fock = h1e + vhf

# Get whole fock matrix directly corresponding to this density, without computing individual components
fock_direct = mf.get_fock(dm=dm)

# Check that ways to get the fock matrix are the same
assert(np.allclose(fock_direct,fock))

# Write fock matrix to file
mmwrite('fock', sparse.coo_matrix(fock))

print("DFT trace: ", dm.trace())
dm_norm = dm
print("||DFT^2 - DFT||: ", linalg.norm(dm_norm @ linalg.inv(ovlp) @ dm_norm - dm_norm))
print("non_overlap: ", linalg.norm(dm_norm @ dm_norm - dm_norm))

nsteps = 100
gcp_rho_list = []
gcp = GCP_DMM(H=h1e, dbeta=0.01, ovlp=ovlp, mu=-10, mf=mf, num_electrons=10)
#gcp = GCP_DMM(H=-np.diag(np.arange(11, dtype=float)), dbeta=0.01, ovlp=np.identity(11), mu=-5.1, mf=mf)
gcp_rho_list.append(gcp.rho.copy())
for i in range(nsteps):
    gcp.non_orth_rk4(1)
    gcp_rho_list.append(gcp.rho.copy())

'''
fig2, axes = plt.subplots(1,1)
im = axes.imshow(gcp_rho_list[0].real, origin='lower', vmin=np.min(gcp_rho_list).real, vmax=np.max(gcp_rho_list).real)
fig2.colorbar(im, ax=axes)

print(np.linalg.norm(ovlp - gcp.rho))
def gcp_animate(i):
    im.set_array(gcp_rho_list[i].real)
    axes.set_title(str(i))
    return im

anim = ani.FuncAnimation(fig2, gcp_animate, init_func=setup_animation, frames=range(int(nsteps)), interval=75, blit=False)

cp_rho_list = []
cp = CP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, num_electrons=dm.trace(), mf=mf)
cp_rho_list.append(cp.rho.copy())
for i in range(nsteps):
    cp.no_zvode(1)
    cp_rho_list.append(cp.rho.copy())
'''


fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
im = ax3.imshow(gcp_rho_list[-1].real, origin='lower')
fig3.colorbar(im, ax=ax3)



print("Trace: ", gcp.rho.trace())
print("rho eigs: ", linalg.eigvalsh(gcp.rho, gcp.ovlp))
gcp.non_ortho_purify(50)
print("after eigs: ", linalg.eigvalsh(gcp.rho, gcp.ovlp))
print("After purify: ", gcp.rho.trace())
temp = gcp.rho.copy()
gcp_norm = temp
print("Our rho norm: ", linalg.norm(gcp_norm @ linalg.inv(ovlp) @ gcp_norm - gcp_norm))
h1e_eigs = linalg.eigvalsh(h1e, ovlp)
print(h1e_eigs)

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
im = ax4.imshow(gcp.rho.real, origin='lower')
fig4.colorbar(im, ax=ax4)

plt.show()
