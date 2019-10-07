from __future__ import print_function
from pprint import pprint

from sage.all import *
from binary_matroid import BinaryMatroid2

## Usage: In Sage console: cd to this directory and load('usage_example.py')

matrix6_3_3 = matrix(GF(2), [
    [1, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 1],
])
matrix7_3_4 = matrix(GF(2), [
    [1, 0, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1],
])
matrix8_4_4 = matrix(GF(2), [
    [1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
])
matrix11_4_5 = matrix(GF(2), [
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
])
matroid6_3_3 = BinaryMatroid2(matrix6_3_3)
matroid7_3_4 = BinaryMatroid2(matrix7_3_4)
matroid8_4_4 = BinaryMatroid2(matrix8_4_4)
matroid11_4_5 = BinaryMatroid2(matrix11_4_5)

print("Cyclic flats:")
pprint(matroid11_4_5.cyclic_flats())

# Configuration of the lattice of cyclic flats
cf_config = matroid11_4_5.cf_lattice_config()

# Show lattice of cyclic flats
matroid11_4_5.show_cf_lattice(title="{}\n{}".format(matroid11_4_5, matrix11_4_5))

# Show configuration
cf_config.show(label=True)
