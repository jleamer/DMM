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


def gcp_self_consistent_Aiken_zvode(h1e, gcp, nsteps, single_step, zvode_steps):
    norm_diff = []
    rho_list = []

    rho_0 = gcp.rho.copy()
    rho_list.append(rho_0.copy())
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0, gcp, h1e, zvode_steps)
        rho_2 = single_step(rho_1, gcp, h1e, zvode_steps)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        rho_list.append(rho_0.copy())

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

    return norm_diff, rho_list


def cp_self_consistent_Aiken_zvode(h1e, cp, nsteps, single_step, zvode_steps):
    norm_diff = []
    rho_list = []

    rho_0 = gcp.rho.copy()
    rho_list.append(rho_0.copy())
    for i in range(nsteps):
        prev_aitken_rho = rho_0.copy()
        rho_1 = single_step(rho_0, cp, h1e, zvode_steps)
        rho_2 = single_step(rho_1, cp, h1e, zvode_steps)

        aitken_rho = rho_2 - (rho_2 - rho_1)**2 / ma.array(rho_2 - 2*rho_1 + rho_0)
        aitken_rho = ma.filled(aitken_rho, fill_value=rho_2)

        rho_0 = aitken_rho
        rho_list.append(rho_0.copy())

        norm_diff.append(linalg.norm(aitken_rho - prev_aitken_rho))

        if np.allclose(aitken_rho, prev_aitken_rho):
            print("Iterations converged!")
            break

    return norm_diff, rho_list


def gcp_single_step_zvode(rho_, gcp, h1e, zvode_steps):
    h = h1e + gcp.mf.get_veff(gcp.mf.mol, rho_)
    gcp_ = GCP_DMM(H=h, ovlp=gcp.ovlp, mu=gcp.mu, dbeta=0.003, mf=gcp.mf)
    gcp_.no_zvode(zvode_steps)
    #gcp_.non_ortho_purify(50)
    return gcp_.rho.copy()


def cp_single_step_zvode(rho_, cp, h1e, zvode_steps):
    h = h1e + cp.mf.get_veff(cp.mf.mol, rho_)
    cp_ = CP_DMM(H=h, ovlp=cp.ovlp, dbeta=0.003, mf=cp.mf, num_electrons=cp.num_electrons)
    cp_.no_zvode(zvode_steps)
    #cp_.non_ortho_purify(50)
    return cp_.rho.copy()

#attempt using scipy.optimize.fixed_point
def exact_function(rho_, h1e, gcp):
    h = h1e + gcp.mf.get_veff(gcp.mf.mol, rho_)
    arg = gcp.inv_ovlp@h
    return ovlp @ linalg.funm(arg, lambda _: 1/(1+np.exp(gcp.beta*(_ - mu))))

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
fig4 = plt.figure(4)
ax41 = fig4.add_subplot(121)
im = ax41.imshow(ovlp.real, origin='lower')
ax41.set_title("real ovlp")
fig4.colorbar(im, ax=ax41)

ax42 = fig4.add_subplot(122)
im = ax42.imshow(ovlp.imag, origin='lower')
ax42.set_title("complex ovlp")
fig4.colorbar(im, ax=ax42)

# Full fock matrix is sum of h1e and vhf
fock = h1e + vhf

# Get whole fock matrix directly corresponding to this density, without computing individual components
fock_direct = mf.get_fock(dm=dm)

# Check that ways to get the fock matrix are the same
assert(np.allclose(fock_direct,fock))

# Write fock matrix to file
mmwrite('fock', sparse.coo_matrix(fock))

print("DFT trace: ", dm.trace())
dm_norm = dm/dm.trace()
print("||DFT^2 - DFT||: ", linalg.norm(dm_norm @ linalg.inv(ovlp) @ dm_norm - dm_norm))


# propagate using GCP DMM zvode
zvode_steps = 100000000
gcp = GCP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, mu=-10, mf=mf)
gcp.no_zvode(zvode_steps)
print("GCP Zvode trace: ", gcp.rho.trace())

#gcp.purify()
nsteps = 50
temp = gcp.rho.copy()
gcp_norm_diff, gcp_rho_list = gcp_self_consistent_Aiken_zvode(h1e, gcp, nsteps, gcp_single_step_zvode, zvode_steps)
gcp.non_ortho_purify(50)

fig1 = plt.figure(1)
ax11 = fig1.add_subplot(221)
ax11.semilogy(gcp_norm_diff, '*-')
ax11.set_xlabel("Iteration #")
ax11.set_ylabel("||P_current - P_prev||")
ax11.set_title("Conv. of P")

ax12 = fig1.add_subplot(222)
im = ax12.imshow(gcp_rho_list[-1].real/gcp_rho_list[-1].trace().real, origin='lower')
ax12.set_xlabel("j")
ax12.set_ylabel("i")
ax12.set_title("Converged P")
fig1.colorbar(im, ax=ax12)

ax13 = fig1.add_subplot(223)
im = ax13.imshow(dm.real/dm.trace().real, origin='lower')
ax13.set_xlabel("j")
ax13.set_title("DFT")
fig1.colorbar(im, ax=ax13)

ax14 = fig1.add_subplot(224)
im = ax14.imshow(temp.real/temp.trace().real, origin='lower')
ax14.set_xlabel("j")
ax14.set_title("GCP Zvode Before SC")
fig1.colorbar(im, ax=ax14)

#print("Final GCP mu: ", gcp_rho_list[-1].mu)
print("Final GCP trace: ", gcp_rho_list[-1].trace())

# propagate using CP DMM zvode
cp = CP_DMM(H=h1e, dbeta=0.003, ovlp=ovlp, num_electrons=10, mf=mf)
cp.no_zvode(zvode_steps)
print("CP Zvode trace: ",cp.rho.trace())
print("CP Zvode chemical potential: ", cp.no_get_mu())
cp_norm_diff, cp_rho_list = cp_self_consistent_Aiken_zvode(h1e, cp, nsteps, cp_single_step_zvode, zvode_steps)
cp.non_ortho_purify(50)

fig2 = plt.figure(2)
ax21 = fig2.add_subplot(221)
ax21.semilogy(cp_norm_diff, '*-')
ax21.set_xlabel("Iteration #")
ax21.set_ylabel("||P_current - P_prev||")
ax21.set_title("Conv. of P")

ax22 = fig2.add_subplot(222)
im = ax22.imshow(cp_rho_list[-1].real/cp_rho_list[-1].trace().real, origin='lower')
ax22.set_xlabel("j")
ax22.set_ylabel("i")
ax22.set_title("Converged P")
fig2.colorbar(im, ax=ax22)

ax23 = fig2.add_subplot(223)
im = ax23.imshow(dm.real/dm.trace().real, origin='lower')
ax23.set_xlabel("j")
ax23.set_title("DFT")
fig2.colorbar(im, ax=ax23)

ax24 = fig2.add_subplot(224)
im = ax24.imshow(cp.rho.real/cp.rho.trace().real, origin='lower')
ax24.set_xlabel("j")
ax24.set_title("CP Zvode before SC")
fig2.colorbar(im, ax=ax24)

#print("Final CP mu: ", cp_rho_list[-1].no_get_mu())
print("Final CP trace: ", cp_rho_list[-1].trace())

'''
fixed_rho = fixed_point(exact_function, dm.copy(), args=(h1e, gcp))
fig3 = plt.figure(3)
ax31 = fig3.add_subplot(111)
im = ax31.imshow(fixed_rho, origin='lower')
ax31.set_xlabel("j")
ax31.set_ylabel("i")
ax31.set_title("Fixed point test")
fig3.colorbar(im, ax=ax31)
'''

plt.show()
