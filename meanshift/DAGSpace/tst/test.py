import DAGSpace
import numpy as np

space = DAGSpace.Space()

arr = np.load("1.npy")
space.init(arr)

band = 20000
idx = list(range(30))
result1 = space.meanshiftAdaptive(band, 1)
print("\n[band = {}] meanshiftAdaptive cluster numbers = {}".format(band, len(result1)))
print(result1)

result2 = space.meanshiftInSubsetAdaptive(idx, band, 1)
print("\n[band = {}] meanshiftInSubsetAdaptive cluster numbers = {}".format(band, len(result2)))
print(result2)

