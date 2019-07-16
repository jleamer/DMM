import numpy as np
from scipy.sparse import dok_matrix
from scipy.io import mmwrite
from scipy.sparse import rand
import sys

#process input parameters
for i in range(1, len(sys.argv), 2):
	argument = sys.argv[i]
	argument_value = sys.argv[i + 1]
	if argument == '--rows':
		rows = int(argument_value)
	elif argument == '--density':
		density = float(argument_value)

hamiltonian = rand(rows, rows, density=0.3) + 1j*rand(rows, rows, density=0.3)
hamiltonian += hamiltonian.conj().T

file_name = "hamiltonian"
mmwrite(file_name, hamiltonian)
