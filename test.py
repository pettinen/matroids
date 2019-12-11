from __future__ import print_function
from pprint import pprint
import operator
import random as py_random
import sys
import time

from sage.all import *

from binary_matroid import BinaryMatroid2
from configuration import Configuration, edge_type
from configuration import cyclic_flats_height3, cyclic_flats_height4


matrix6_3_3 = matrix(GF(2), [
    [1, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 1],
])
matrix6_4_2 = matrix(GF(2), [
    [1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
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
matroid6_4_2 = BinaryMatroid2(matrix6_4_2)
matroid7_3_4 = BinaryMatroid2(matrix7_3_4)
matroid8_4_4 = BinaryMatroid2(matrix8_4_4)
matroid11_4_5 = BinaryMatroid2(matrix11_4_5)


#print("Isomorphic:", matroid1.is_isomorphic(matroid2))
#print("Same config:",
#   matroid1.cf_lattice_config() == matroid2.cf_lattice_config())

# Return atoms (in Z) of a poset, configuration or matroid
def atoms(data):
    if isinstance(data, sage.matroids.matroid.Matroid):
        poset = Poset((data.cyclic_flats(), operator.le))
    elif isinstance(data, Configuration):
        poset = data.poset()
    else:
        poset = data
    return poset.upper_covers(poset.bottom())

def coatoms(data):
    if isinstance(data, sage.matroids.matroid.Matroid):
        poset = Poset((data.cyclic_flats(), operator.le))
    elif isinstance(data, Configuration):
        poset = data.poset()
    else:
        poset = data
    return poset.lower_covers(poset.top())

# Pretty figure printing
def _show_base(show_func, matroid, **kwargs):
    show_func(title="{}\n{}".format(matroid, matroid.representation()), **kwargs)
def show_cf(matroid, **kwargs):
    _show_base(matroid.show_cf_lattice, matroid, **kwargs)
def show_config(matroid, **kwargs):
    _show_base(matroid.cf_lattice_config().show, matroid, **kwargs)


#h4_1 = matroid11_4_5.restrict({0,2,4,5,6,7,8,9,10})
#h4_2 = matroid11_4_5.restrict({0,1,3,4,5,6,7,8,9,10})
matroid7_4_2 = (matroid11_4_5
    .restrict({0,1,3,4,5,6,7,8,9,10})
    .restrict({0,2,3,4,5,6,7}))
print(matroid7_4_2.representation())
# lattice = LatticePoset(Poset(({
    # frozenset({0,1,2,3,4,5,6}),
    # frozenset({0,1,2,3,4}),
    # frozenset({0,1,2,5}),
    # frozenset({0,3,4,5}),
    # frozenset({0,2,3}),
    # frozenset({1,2,4}),
    # frozenset({}),
# }, operator.le)))
# lattice.show(figsize=15)

#show_cf(matroid11_4_5)
#show_config(h4_2)
#h4 = BinaryMatroid2.from_matroid(h4,
#    groundset=[3,5,2,6,7,4,1,8,9]).restrict({3,4,6,7,8}).dual()
#show_config(h4, index=True)

#config = h4.cf_lattice_config()
#config.show(index=True)
#print(restricted)
#restricted.show(index=True)
#print(cyclic_flats_height4(len(h4), h4.cf_lattice_config()))


#for cf in coatoms(matroid11_4_5):
#    h3 = matroid11_4_5.restrict(cf)
#    if len(h3) - h3.full_rank() <= 2:
#        show_config(h3)

#cf = matroid11_4_5.cyclic_flats()
#poset = Poset((cf, operator.le))
#atoms = poset.upper_covers(poset.bottom())
#elem_counts = {}
# for atom in atoms:
    # for x in atom:
        # try:
            # elem_counts[x] += 1
        # except KeyError:
            # elem_counts[x] = 1
# print(elem_counts)

# # Count intersections of length n
# intersect_counts = {}
# for x in atoms:
    # for y in atoms:
        # if x != y:
            # sz = len(x & y)
            # try:
                # intersect_counts[sz] += 1
            # except KeyError:
                # intersect_counts[sz] = 1
# print(intersect_counts)
