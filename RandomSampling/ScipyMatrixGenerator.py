import numpy as np
from scipy.sparse import dok_matrix
from scipy.io import mmwrite
from scipy.sparse import rand
import sys

#process input parameters
if len(sys.argv) == 1:
	raise ValueError("No arguments provided!")

for i in range(1, len(sys.argv), 2):
	argument = sys.argv[i]
	argument_value = sys.argv[i + 1]
	if argument == '--rows':
		rows = int(argument_value)
	elif argument == '--density':
		density = float(argument_value)

hamiltonian = rand(rows, rows, density=density) + 1j*rand(rows, rows, density=density)
hamiltonian += hamiltonian.conj().T

mmwrite("hamiltonian", hamiltonian)
