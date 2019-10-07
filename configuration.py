from collections import Counter, namedtuple
from six.moves import range
import itertools
import string

from sage.all import *


Element = namedtuple('Element', ['size', 'rank', 'index'])

class Configuration(object):
    # Should improve speed and decrease memory usage
    __slots__ = 'elements', 'covers'

    # `elements` is a list of type Element(size, rank, index),
    # where the indices are consecutive integers starting from zero.
    # `covers` is a list of pairs (x, y), where x and y are
    # integers corresponding to element indices.
    def __init__(self, elements, covers, sanity_checks=False):
        if sanity_checks:
            for elem in elements:
                assert isinstance(elem, Element)
                assert elem.size >= 0 and elem.rank >= 0
                if covers:
                    assert any(x == elem or y == elem for x, y in covers)
            for index in range(len(elements)):
                assert any(elem.index == index for elem in elements)
            for x, y in covers:
                assert x in elements and y in elements
            if not covers:
                assert len(elements) == 1
        self.elements = elements
        self.covers = sorted(covers)

    def poset(self):
        return Poset((self.elements, self.covers), cover_relations=True)

    def show(self, label=True, index=False, **kwargs):
        lattice = LatticePoset(self.poset())
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
        else:
            element_labels = {_label: '' for _label in labels_dict}

        if 'figsize' not in kwargs:
            kwargs['figsize'] = 15
        if 'vertex_color' not in kwargs:
            kwargs['vertex_color'] = 'white'
        if 'vertex_shape' not in kwargs:
            kwargs['vertex_shape'] = 'o'
        if 'vertex_size' not in kwargs:
            kwargs['vertex_size'] = 1000 if label else 20
        lattice.show(
            element_labels=element_labels,
            heights=heights,
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

    # equals_fast (and __hash__) assumes that the list of cover relations
    # (size, rank pairs) determines the configuration (except in the
    # one-element case there are no relations, of course).
    # Haven't proved this, but seems to work. Worst case, false positives
    # might be generated, but no positive result should be missed.
    def equals_fast(self, other):
        if not self.covers: # one-element case
            return self.elements == other.elements
        covers_self = [((x.size, x.rank), (y.size, y.rank))
                       for x, y in self.covers]
        covers_other = [((x.size, x.rank), (y.size, y.rank))
                        for x, y in other.covers]
        return covers_self == covers_other

    def equals_robust(self, other):
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

        other_poset = other.poset()
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

    __eq__ = equals_fast

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        if not self.covers: # one-element case
            return hash(tuple(self.elements))
        return hash(tuple(((x.size, x.rank), (y.size, y.rank))
                          for x, y in self.covers))

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
    print("Looking for a ({}, {})-matroid", top_elem.size, top_elem.rank)


def reconstruct_height3(config, sanity_checks=True):
    # Assumes simple matroid with no isthmuses, with all atoms the same size.
    # Assumes | z_i & z_j | = 1 for distinct atoms z_i, z_j.
    # Works for e.g. the unique (6,3,3)-matroid and the simplex (7,3,4)-matroid.
    poset = config.poset()
    if sanity_checks:
        assert poset.height() == 3
        assert poset.has_top() and poset.has_bottom()

    groundset = set(range(poset.top().size))
    atoms = poset.upper_covers(poset.bottom())

    # Assign {0, 1, ..., size-1} to be the first atom
    cyclic_flats = {atoms[0]: set(range(atoms[0].size))}
    # Keep track of the amount of times each element is used
    uses = {i: 0 for i in groundset}
    for i in cyclic_flats[atoms[0]]:
        uses[i] += 1

    for atom in atoms[1:]:
        current = set()
        for found_cf in cyclic_flats.values():
            candidates = sorted(found_cf - current,
                                key=lambda i: uses[i])
            #candidates = found_cf - current
            for candidate in candidates:
                test_set = copy(current)
                test_set.add(candidate)
                if all(len(test_set & x) <= 1 for x in cyclic_flats.values()):
                    current.add(candidate)
                    uses[candidate] += 1
                    break

        # Add unused elements to fill the current set
        for i in uses:
            if len(current) == atom.size:
                break
            if uses[i] == 0:
                current.add(i)
                uses[i] += 1
        cyclic_flats[atom] = current

    if sanity_checks:
        for atom, x in cyclic_flats.items():
            assert len(x) == atom.size
            for y in cyclic_flats.values():
                assert x == y or len(x & y) == 1

    return cyclic_flats


def reconstruct_height3_general(config, sanity_checks=True):
    # WORK IN PROGRESS
    # Assume simple (n, k, d)-matroid with no isthmuses and k >= 3.
    # Assume atoms have rank k-1. Then |Z1 & Z2| <= k/2 for atoms Z1, Z2.
    poset = config.poset()
    if sanity_checks:
        assert poset.height() == 3
        assert poset.has_top() and poset.has_bottom()
        assert poset.top().rank >= 3

    n = poset.top().size
    k = poset.top().rank
    groundset = set(range(n))
    atoms = poset.upper_covers(poset.bottom())
    intersect_max = k / 2

    cyclic_flats = {}
    # Keep track of the amount of times each element is used
    uses = {i: 0 for i in groundset}

    for atom in atoms:
        current = set()
        for found_cf in cyclic_flats.values():
            candidates = sorted(found_cf - current,
                                key=lambda i: uses[i])
            for candidate in candidates:
                test_set = copy(current)
                test_set.add(candidate)
                if all(len(test_set & x) <= intersect_max
                       for x in cyclic_flats.values()):
                    current.add(candidate)
                    uses[candidate] += 1

        # Add unused elements to fill the current set
        for i in uses:
            if len(current) == atom.size:
                break
            if uses[i] == 0:
                current.add(i)
                uses[i] += 1
        cyclic_flats[atom] = current
    return cyclic_flats
