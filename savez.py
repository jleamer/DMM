import numpy as np
import tempfile
import os
import sys
import zipfile

filename = "test.zip"
zipped = zipfile.ZipFile(filename, "a")

filenames = ["test_" + str(i) + ".npy" for i in range(5)]
for i in range(5):
	f = tempfile.TemporaryFile(suffix=".npy", mode="w+")
	f.write("This is file " + str(i))
	zipped.write(filenames[i])
	f.close()
zipped.close()
