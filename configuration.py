from collections import Counter, namedtuple
from six.moves import range
import itertools
import string

from sage.all import *


Element = namedtuple('Element', ['size', 'rank', 'index'])

class Configuration:
    # `elements` is a list of type Element(size, rank, index),
    # where the indices are consecutive integers starting from zero.
    # `covers` is a list of pairs (x, y), where x and y are
    # integers corresponding to element indices.
    def __init__(self, elements, covers, sanity_checks=False):
        if sanity_checks:
            for elem in elements:
                assert isinstance(elem, Element)
                assert elem.size >= 0 and elem.rank >= 0
                assert any(x == elem or y == elem for x, y in covers)
            for index in range(len(elements)):
                assert any(elem.index == index for elem in elements)
            for x, y in covers:
                assert x in elements and y in elements
        self.elements = elements
        self.covers = covers
        self.poset = Poset((self.elements, self.covers), cover_relations=True)

    def show(self, label=True, index=False, **kwargs):
        lattice = LatticePoset(self.poset)
        if index:
            labels_dict = {
                elem: '{0}\n{1.size}, {1.rank}'.format(
                    string.ascii_uppercase[elem.index], elem)
                for elem in self.elements
            }
        else:
            labels_dict = {
                elem: '{0.size}, {0.rank}'.format(elem)
                for elem in self.elements
            }
        heights = {}
        for elem in self.elements:
            try:
                heights[elem.rank].append(elem)
            except KeyError:
                heights[elem.rank] = [elem]
        if label:
            element_labels = labels_dict
            vertex_size = 1000
        else:
            element_labels = {_label: '' for _label in labels_dict}
            vertex_size = 20

        lattice.show(
            element_labels=element_labels,
            heights=heights,
            vertex_shape='o',
            vertex_size=vertex_size,
            figsize=[10, 10],
            vertex_color='white',
            **kwargs
        )

    # Removes loops.
    def simplify(self):
        bottom_elem = self.poset.bottom()
        def modify(elem):
            return Element(elem.size - bottom_elem.size, elem.rank, elem.index)
        elements = [modify(elem) for elem in self.elements]
        covers = [(modify(x), modify(y)) for x, y in self.covers]
        return self.__class__(elements, covers)

    def __eq__(self, other):
        if len(self.covers) != len(other.covers):
            return False

        counts_self = Counter(
            map(lambda elem: (elem.size, elem.rank), self.elements))
        counts_other = Counter(
            map(lambda elem: (elem.size, elem.rank), other.elements))
        if counts_self != counts_other:
            return False

        def indices_dict(config):
            indices = {}
            for elem in config.elements:
                try:
                    indices[(elem.size, elem.rank)].append(elem.index)
                except KeyError:
                    indices[(elem.size, elem.rank)] = [elem.index]
            return indices

        indices_dict_self = indices_dict(self)
        indices_dict_other = indices_dict(other)
        multiples = {}
        swaps = {}
        for (size, rank), indices_other in indices_dict_other.items():
            try:
                indices_self = indices_dict_self[(size, rank)]
            except KeyError:
                return False
            if len(indices_other) == 1:
                swaps[indices_self[0]] = indices_other[0]
            else:
                multiples[tuple(indices_self)] = list(
                    itertools.permutations(indices_other))

        permutations = []
        permutations_amount = 1
        for list_ in multiples.values():
            permutations_amount *= len(list_)
        for i in range(permutations_amount):
            permutation = copy(swaps)
            for indices_self, perms in multiples.items():
                perm = perms[i % len(perms)]
                for i, index_self in enumerate(indices_self):
                    permutation[index_self] = perm[i]
            permutations.append(permutation)

        def find_element(index, elements):
            for element in elements:
                if element.index == index:
                    return element
            raise IndexError("no such index: {}".format(index))

        other_poset = other.poset
        for permutation in permutations:
            elements = [
                Element(elem.size, elem.rank, permutation[elem.index])
                for elem in self.elements
            ]
            covers = [
                (find_element(permutation[x.index], elements),
                 find_element(permutation[y.index], elements))
                for x, y in self.covers
            ]
            poset = Poset((elements, covers), cover_relations=True)
            if poset == other_poset:
                return True
        return False

    def __ne__(self, other):
        return not self == other

    # It seems to me that the list of cover relationss (without indices)
    # determines the configuration. Not sure, but at worst a bad hasher
    # should just generate false positives.
    def __hash__(self):
        covers = sorted(((x.size, x.rank), (y.size, y.rank))
                           for x, y in self.covers)
        return hash(tuple(covers))

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return "Lattice configuration containing {} elements".format(len(self))


# Returns: type of edge; (k, n) corresponding to a minor U(k, n) in the matroid
def edge_type(cover_relation):
    x, y = cover_relation
    if y.size > x.size:
        x, y = y, x
    rank_diff = x.rank - y.rank
    nullary_diff = (x.size - x.rank) - (y.size - y.rank)
    
    if rank_diff > 1 and nullary_diff == 1:
        return 'rank_edge', (rank_diff, rank_diff + 1)
    elif nullary_diff > 1 and rank_diff == 1:
        return 'nullary_edge', (nullary_diff, nullary_diff + 1)
    elif rank_diff == 1 and nullary_diff == 1:
        return 'elementary_edge', (1, 2)
    else:
        raise ValueError("Impossible edge")

def reconstruct(groundset_size, config):
    loops_amount = config.poset.bottom().size
    groundset_size -= loops_amount
    config = config.simplify()
    top_elem = config.poset.top()
    isthmuses_amount = groundset_size - top_elem.size
    groundset = set(range(groundset_size - isthmuses_amount))
    print("{}x{} matrix", top_elem.rank, top_elem.size)
