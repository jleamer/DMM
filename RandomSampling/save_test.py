import numpy as np

x = np.arange(10)
y = np.sin(x)
args = [x, y]

filename = "temp.npz"
np.savez(filename, *args)
npzfile = np.load(filename)
print(npzfile.files)
print(npzfile['arr_0'])
