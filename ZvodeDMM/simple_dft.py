from pyscf import gto, dft
import numpy
from scipy.io import mmwrite, mmread
from scipy import sparse
from scipy import linalg
import matplotlib.pyplot as plt
from ZvodeDMM.dmm import DMM
from ZvodeDMM.gcp_dmm import GCP_DMM
from ZvodeDMM.cp_dmm import CP_DMM


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

#Now we want to test that our DMM method arrives at the same density matrix
# first get the inverse overlap matrix
inv_ovlp = linalg.inv(ovlp)

# propagate using GCP DMM
gcp = GCP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, mu=mu)
gcp.no_zvode(1000)

# get exact DM
P = ovlp@linalg.inv(gcp.identity + linalg.expm(gcp.beta*(inv_ovlp@gcp.H-gcp.mu*gcp.identity)))

# Plot each
fig1 = plt.figure(1)
plt.subplot(131)
plt.imshow(gcp.rho.real, origin='lower')
plt.colorbar()
plt.ylabel('i')
plt.xlabel('j')
plt.title("DMM (real)")

plt.subplot(132)
plt.imshow(P.real, origin='lower')
plt.colorbar()
plt.xlabel('j')
plt.title("Exact (real)")

plt.subplot(133)
plt.imshow(dm.real, origin='lower')
plt.colorbar()
plt.ylabel('j')
plt.title("DFT (real)")


# Repeat for cp case
cp = CP_DMM(H=h1e, dbeta = 0.003, ovlp=ovlp, num_electrons=10)
cp.no_zvode(1000)

fig2 = plt.figure(2)
plt.subplot(131)
plt.imshow(cp.rho.real, origin='lower')
plt.colorbar()
plt.ylabel('i')
plt.xlabel('j')
plt.title("DMM (real)")

plt.subplot(132)
plt.imshow(P.real, origin='lower')
plt.colorbar()
plt.xlabel('j')
plt.title("Exact (real)")

plt.subplot(133)
plt.imshow(dm.real, origin='lower')
plt.colorbar()
plt.ylabel('j')
plt.title("DFT (real)")
plt.show()
