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
ovlp = mmread("dft_overlap.mtx").toarray()

h1e_eigs = linalg.eigvalsh(h1e, ovlp)
print(h1e_eigs)
