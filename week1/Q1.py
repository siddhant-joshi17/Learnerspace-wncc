import numpy as np
import pandas as pd
import random


# Generate array
array_2d = np.random.randint(1, 51, size=(5, 4))
print("\nArray:")
print(array_2d)

# Anti-diagonal elements
anti_diag = [array_2d[i, -1-i] for i in range(min(array_2d.shape))]
print("\nAnti-diagonal:", anti_diag)

# Max value in each row
row_max = np.max(array_2d, axis=1)
print("\nMax in each row:", row_max)

# Elements <= mean
mean_val = np.mean(array_2d)
below_mean = array_2d[array_2d <= mean_val]
print("\nElements <= mean ({}):".format(mean_val))
print(below_mean)

# Boundary traversal

def numpy_boundary_traversal(matrix):
    top = list(matrix[0])
    right = [matrix[i][-1] for i in range(1, matrix.shape[0]-1)]
    bottom = list(matrix[-1][::-1])
    left = [matrix[i][0] for i in range(matrix.shape[0]-2, 0, -1)]
    return top + right + bottom + left

print("\nBoundary Traversal:", numpy_boundary_traversal(array_2d))

