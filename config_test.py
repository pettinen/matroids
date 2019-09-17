from __future__ import print_function
import random as py_random
import sys
import time

from sage.all import *
from sage.matrix.constructor import matrix

from configuration import Configuration, Element
from binary_matroid import BinaryMatroid2


start = time.time()

SHOW = True

def make_relations(elements, relations):
    elements = {elem.index: elem for elem in elements}
    return [(elements[x], elements[y]) for x, y in relations]

elements1 = [
    Element(size=0, rank=0, index=0),
    Element(3, 2, 1),
    Element(3, 2, 2),
    Element(5, 4, 3),
    Element(5, 4, 4),
    Element(5, 4, 5),
    Element(8, 5, 6),
]
relations1 = make_relations(elements1,
    [(0, 1), (0, 2), (0, 4), (0, 5), (1, 3), (2, 3), (3, 6), (4, 6), (5, 6)])
config1 = Configuration(elements1, relations1)

elements2 = [
    Element(size=0, rank=0, index=6),
    Element(3, 2, 5),
    Element(3, 2, 4),
    Element(5, 4, 3),
    Element(5, 4, 2),
    Element(5, 4, 1),
    Element(8, 5, 0),
]
relations2 = make_relations(elements2,
    [(6, 5), (6, 4), (6, 2), (6, 1), (5, 3), (4, 3), (3, 0), (2, 0), (1, 0)])
config2 = Configuration(elements2, relations2)

matroid = BinaryMatroid2(matrix=matrix(GF(2), [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]))
config3 = matroid.cf_lattice_config()

if SHOW:
    config1.show(index=True, title='config1')
    config2.show(index=True, 
        title="config2 ("
              + ("not " if config1 != config2 else "")
              + "equal to config1)")
    config3.show(label=True, title=config3)

assert config1 == config2
assert config1 != config3
assert not config1 != config2
assert not config1 == config3

end = time.time()
print("Completed in {} seconds".format(end - start), file=sys.stderr)
