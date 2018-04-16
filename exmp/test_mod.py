import hello
import numpy as np

hello.hello_python()
K = np.array([[1,2,3],[4,5,6],[7,8,9]])
K = np.array([1,2,3])

resources = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
resources = np.array(resources)
#resources arg, flattened
#will have 11*3 = 33 elements
#include array, flattened
c = np.zeros((343,343)).reshape((343*343,))
print(hello.hello_numpy(resources.reshape((33,)), np.array(range(343)), c))
print(resources)
print(list(c))
