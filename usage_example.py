from __future__ import print_function
from pprint import pprint

from sage.all import *
from sage.matrix.constructor import matrix

from binary_matroid import BinaryMatroid2

## Usage: In Sage console: cd to this directory and load('usage_example.py')

matrix_11x4 = matrix(GF(2), [
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
])
matroid = BinaryMatroid2(matrix_11x4)

print("Cyclic flats:")
pprint(matroid.cyclic_flats())

# Configuration of the lattice of cyclic flats
cf_config = matroid.cf_lattice_config()

# Show lattice of cyclic flats
matroid.show_cf_lattice(title="{}\n{}".format(matroid, matrix_11x4))

# Show configuration
cf_config.show(label=True)
