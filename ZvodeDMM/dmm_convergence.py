from pyscf import gto, dft
import numpy as np
from scipy.io import mmwrite, mmread
from scipy import sparse
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from ZvodeDMM.dmm import DMM
from ZvodeDMM.gcp_dmm import GCP_DMM
from ZvodeDMM.cp_dmm import CP_DMM
import numpy.ma as ma

def self_consistent_Aiken_zvode(h1e, gcp, nsteps):

    def single_step(rho_):
        h = h1e + gcp.mf.get_veff(gcp.mf.mol, rho_)
        gcp_ = GCP_DMM(H=h, ovlp=gcp.ovlp, mu=gcp.mu, dbeta=0.003, mf=gcp.mf)
        gcp_.no_zvode(1000)
        return gcp_.rho
        #gcp.H = h
        #gcp.beta = 0
        #gcp.rho = gcp.ovlp.copy()
        #gcp.no_zvode(1000)
        #return gcp.rho
        #h = h1e + gcp.mf.get_veff(gcp.mf.mol, rho_)
        #mu = gcp.mu
        #arg = gcp.inv_ovlp@h
        #return ovlp @ linalg.inv(gcp.identity + linalg.expm(gcp.beta*(arg-mu*gcp.identity)))
        #gcp_ = GCP_DMM(H=h, ovlp=gcp.ovlp, mu=gcp.mu, dbeta=0.003, mf=gcp.mf)
        #gcp_.no_zvode(1000)
        #return gcp_.rho


        #identity = numpy.identity(rho.shape[0])


        #return ovlp @ linalg.funm(arg, lambda _: numpy.exp(-beta*(_ - mu))/(1+numpy.exp(-beta*(_ - mu))))
        #return ovlp @ linalg.funm(arg, lambda _: 1/(1+numpy.exp(beta*(_ - mu))))
        #return ovlp @ linalg.funm(arg, lambda _: _ >= mu)
        #return ovlp @ linalg.inv(identity + linalg.expm(beta*(arg-mu*identity)))


    norm_diff = []
    rho_list = []

    rho_0 = gcp.rho.copy()
    rho_list.append(rho_0.copy())
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0)
        rho_2 = single_step(rho_1)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        rho_list.append(rho_0.copy())

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

    return norm_diff, rho_list

# Seems that the RK4 self-consistency convergence is less effective than Zvode
def self_consistent_Aiken_rk4(h1e, gcp, nsteps):

    def single_step(rho_):
        h = h1e + gcp.mf.get_veff(gcp.mf.mol, rho_)
        gcp_ = GCP_DMM(H=h, ovlp=gcp.ovlp, mu=gcp.mu, dbeta=0.003, mf=gcp.mf)
        gcp_.non_orth_rk4(1000)
        return gcp_.rho

    norm_diff = []
    rho_list = []

    rho_0 = gcp.rho.copy()
    rho_list.append(rho_0.copy())
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0)
        rho_2 = single_step(rho_1)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        rho_list.append(rho_0.copy())

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

    return norm_diff, rho_list

# Seems that the direct inversion method for self-consistent converge is also less effective than Zvode
def self_consistent_Aiken_inv(h1e, gcp, nsteps):

    def single_step(rho_):
        h = h1e + gcp.mf.get_veff(gcp.mf.mol, rho_)
        mu = gcp.mu
        arg = gcp.inv_ovlp@h
        return ovlp @ linalg.inv(gcp.identity + linalg.expm(gcp.beta*(arg-mu*gcp.identity)))
        #return ovlp @ linalg.funm(arg, lambda _: numpy.exp(-beta*(_ - mu))/(1+numpy.exp(-beta*(_ - mu))))
        #return ovlp @ linalg.funm(arg, lambda _: 1/(1+np.exp(gcp.beta*(_ - mu))))
        #return ovlp @ linalg.inv(identity + linalg.expm(beta*(arg-mu*identity)))


    norm_diff = []
    rho_list = []

    rho_0 = gcp.rho.copy()
    rho_list.append(rho_0.copy())
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0)
        rho_2 = single_step(rho_1)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        rho_list.append(rho_0.copy())

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

    return norm_diff, rho_list

# Seems that this one is the winner.  it gets to a similar level as using Zvode but faster (like 20 steps)
def self_consistent_Aiken_funm(h1e, gcp, nsteps):

    def single_step(rho_):
        h = h1e + gcp.mf.get_veff(gcp.mf.mol, rho_)
        mu = gcp.mu
        arg = gcp.inv_ovlp@h
        #return ovlp @ linalg.inv(gcp.identity + linalg.expm(gcp.beta*(arg-mu*gcp.identity)))
        #return ovlp @ linalg.funm(arg, lambda _: numpy.exp(-beta*(_ - mu))/(1+numpy.exp(-beta*(_ - mu))))
        return ovlp @ linalg.funm(arg, lambda _: 1/(1+np.exp(gcp.beta*(_ - mu))))
        #return ovlp @ linalg.inv(identity + linalg.expm(beta*(arg-mu*identity)))


    norm_diff = []
    rho_list = []

    rho_0 = gcp.rho.copy()
    rho_list.append(rho_0.copy())
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0)
        rho_2 = single_step(rho_1)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        rho_list.append(rho_0.copy())

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

    return norm_diff, rho_list

def cp_self_consistent_Aiken_funm(h1e, cp, nsteps):

    def single_step(rho_):
        h = h1e + cp.mf.get_veff(cp.mf.mol, rho_)
        mu = cp.no_get_mu()
        arg = cp.inv_ovlp@h
        #return ovlp @ linalg.inv(gcp.identity + linalg.expm(gcp.beta*(arg-mu*gcp.identity)))
        #return ovlp @ linalg.funm(arg, lambda _: numpy.exp(-beta*(_ - mu))/(1+numpy.exp(-beta*(_ - mu))))
        return ovlp @ linalg.funm(arg, lambda _: 1/(1+np.exp(cp.beta*(_ - mu))))
        #return ovlp @ linalg.inv(identity + linalg.expm(beta*(arg-mu*identity)))


    norm_diff = []
    rho_list = []

    rho_0 = gcp.rho.copy()
    rho_list.append(rho_0.copy())
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0)
        rho_2 = single_step(rho_1)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        rho_list.append(rho_0.copy())

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

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

# Compute the kinetic + e-nuclear repulsion energy
e1 = np.einsum('ij,ji', h1e, dm)

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
assert(np.allclose(fock_direct,fock))

# Write fock matrix to file
mmwrite('fock', sparse.coo_matrix(fock))

print("DFT trace: ", dm.trace())
print("||DFT^2 - DFT||: ", linalg.norm(dm@dm - dm))


# propagate using GCP DMM zvode
gcp = GCP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, mu=mu, mf=mf)
gcp.no_zvode(1000)
print("GCP Zvode trace: ", gcp.rho.trace())
#gcp.purify
nsteps = 100
#gcp_zvode_norm_diff, gcp_zvode_rho_list = self_consistent_Aiken_zvode(h1e, gcp, nsteps)
gcp_funm_norm_diff, gcp_funm_rho_list = self_consistent_Aiken_funm(h1e, gcp, nsteps)

fig1 = plt.figure(1)
ax11 = fig1.add_subplot(131)
ax11.semilogy(gcp_funm_norm_diff, '*-')
ax11.set_xlabel("Iteration #")
ax11.set_ylabel("||P_current - P_prev||")
ax11.set_title("Conv. of P")

ax12 = fig1.add_subplot(132)
im = ax12.imshow(gcp_funm_rho_list[-1].real/gcp_funm_rho_list[-1].trace().real, origin='lower')
ax12.set_xlabel("i")
ax12.set_ylabel("j")
ax12.set_title("Converged P")
fig1.colorbar(im, ax=ax12)

ax13 = fig1.add_subplot(133)
im = ax13.imshow(dm.real/dm.trace().real, origin='lower')
ax13.set_xlabel("j")
ax13.set_title("DFT")
fig1.colorbar(im, ax=ax13)

# propagate using CP DMM zvode
cp = CP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, num_electrons=dm.trace(), mf=mf)
cp.no_zvode(1000)
#cp.purify()
print("CP Zvode trace: ",cp.rho.trace())
print("CP Zvode chemical potential: ", cp.no_get_mu())
cp_funm_norm_diff, cp_funm_rho_list = cp_self_consistent_Aiken_funm(h1e, cp, nsteps)

fig2 = plt.figure(2)
ax21 = fig2.add_subplot(131)
ax21.semilogy(cp_funm_norm_diff, '*-')
ax21.set_xlabel("Iteration #")
ax21.set_ylabel("||P_current - P_prev||")
ax21.set_title("Conv. of P")

ax22 = fig2.add_subplot(132)
im = ax22.imshow(cp_funm_rho_list[-1].real/cp_funm_rho_list[-1].trace().real, origin='lower')
ax22.set_xlabel("i")
ax22.set_ylabel("j")
ax22.set_title("Converged P")
fig2.colorbar(im, ax=ax22)

ax23 = fig2.add_subplot(133)
im = ax23.imshow(dm.real/dm.trace().real, origin='lower')
ax23.set_xlabel("j")
ax23.set_title("DFT")
fig2.colorbar(im, ax=ax23)

plt.show()
