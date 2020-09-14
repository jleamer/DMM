import numpy as np
from numba import njit
from scipy import sparse
from scipy.io import mmwrite, mmread
from numba import njit
from scipy import linalg
import matplotlib.pyplot as plt
import sys

# Add NTPoly library to path and import
sys.path.insert(0,"/home/jacob/PycharmProjects/DMM/NTPoly/Build/python")

import NTPolySwig as nt

# MPI4PY module
from mpi4py import MPI
comm = MPI.COMM_WORLD


def NTPoly_cp(H, ovlp, num_electrons, convergence_threshold=1e-10, threshold=1e-10, process_rows=1, process_columns=1, process_slices=1):
    sparseH = sparse.coo_matrix(H)
    mmwrite("sparseH.mtx", sparseH)
    sparse_ovlp = sparse.coo_matrix(ovlp)
    mmwrite("sparse_ovlp.mtx", sparse_ovlp)

    # Construct process grid
    nt.ConstructGlobalProcessGrid(process_rows, process_columns, process_slices)

    # Set up solver parameters
    solver_parameters = nt.SolverParameters()
    solver_parameters.SetConvergeDiff(convergence_threshold)
    solver_parameters.SetThreshold(threshold)
    solver_parameters.SetVerbosity(False)

    nt_H = nt.Matrix_ps("sparseH.mtx")
    nt_ovlp = nt.Matrix_ps("sparse_ovlp.mtx")
    inv_sqrt_ovlp = nt.Matrix_ps(nt_ovlp.GetActualDimension())
    nt.SquareRootSolvers.InverseSquareRoot(nt_ovlp, inv_sqrt_ovlp, solver_parameters)
    Density = nt.Matrix_ps(nt_H.GetActualDimension())

    # Compute the density matrix
    energy_value, mu = \
        nt.DensityMatrixSolvers.PM(nt_H, inv_sqrt_ovlp, num_electrons, Density, solver_parameters)
    Density.WriteToMatrixMarket("ntpoly_density.mtx")
    nt_rho = mmread("ntpoly_density.mtx").toarray()

    # Destruct Grid
    nt.DestructGlobalProcessGrid()

    return nt_rho




