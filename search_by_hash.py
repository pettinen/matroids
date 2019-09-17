from __future__ import print_function
import random as py_random
import time

from sage.all import *
#from sage.matrix.constructor import matrix

from binary_matroid import BinaryMatroid2
from configuration import Configuration


ROWS = 3
COLS = 6
PRINT_PROGRESS = True

results = {}
n = 0
start = time.time()

## Exhaustive:
for matrix_ in MatrixSpace(GF(2), ROWS, COLS):

## Random:
#while True:
#    matrix_ = matrix(GF(2),
#        [[py_random.randint(0, 1) for _ in range(COLS)] for _ in range(ROWS)])

    matroid = BinaryMatroid2(matrix=matrix_)
    n += 1
    if PRINT_PROGRESS:
        print('\r{}'.format(n), end='', file=sys.stderr)

    #if not matroid.is_simple():
    #    continue
    config = matroid.cf_lattice_config()
    if config in results:
        if not matroid.is_isomorphic(results[config][0]):
            print("\nFound:")
            print(matrix_)
            print('--------')
            print(results[config][1])
            config.show(index=True)
            break
    else:
        results[config] = (matroid, matrix_)

print('\n{} different configurations'.format(len(results)), file=sys.stderr)
end = time.time()
print("Completed in {} seconds".format(round(end - start, 2)), file=sys.stderr)
