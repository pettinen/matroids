from __future__ import print_function
from collections import Counter, namedtuple
from copy import deepcopy
from pprint import pprint
from six.moves import range
import itertools
import string

from sage.all import *

from binary_matroid import BinaryMatroid2, Uniform


Element = namedtuple('Element', ['size', 'rank', 'index'])

class Configuration(object):
    # Should improve speed and decrease memory usage
    __slots__ = 'elements', 'covers', '_poset', '_lattice'

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
        self._poset = self._lattice = None

    def poset(self):
        if self._poset is None:
            self._poset = Poset((self.elements, self.covers),
                cover_relations=True)
        return self._poset

    def lattice(self):
        if self._lattice is None:
            self._lattice = LatticePoset(self.poset())
        return self._lattice

    def atoms(self):
        return self.lattice().atoms()

    def coatoms(self):
        return self.lattice().coatoms()

    def height(self):
        return self.poset().height()

    def top(self):
        return self.poset().top()

    def bottom(self):
        return self.poset().bottom()

    def upper_covers(self, elem):
        return self.poset().upper_covers(elem)

    def lower_covers(self, elem):
        return self.poset().lower_covers(elem)

    def restrict(self, top_elem):
        new_elements = filter(
            lambda elem: self.poset().is_lequal(elem, top_elem),
            self.elements
        )
        new_covers = filter(
            lambda rel: rel[0] in new_elements and rel[1] in new_elements,
            self.covers
        )
        return Configuration(new_elements, new_covers)

    def show(self, label=True, index=False, **kwargs):
        lattice = LatticePoset(self.poset())
        if index:
            if len(self.elements) > 26:
                indices = range(1, len(self.elements) + 1)
            else:
                indices = string.ascii_uppercase
            labels_dict = {
                elem: '{0}\n{1.size}, {1.rank}'.format(
                    indices[elem.index], elem)
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
        bottom_elem = self.bottom()
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

def cyclic_flats_height3(config, groundset=None):
    # Assume config is of a simple matroid with no isthmuses

    # Replace default groundset with the given one
    def change_groundset(cyclic_flats):
        if groundset is None:
            return cyclic_flats
        permutation = dict(enumerate(groundset))
        new_cfs = set()
        for cf in cyclic_flats:
            new_cfs.add(frozenset(permutation[elem] for elem in cf))
        return new_cfs

    matroid6_3_3 = BinaryMatroid2(matrix(GF(2), [
        [1, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1],
    ]))
    if config == matroid6_3_3.cf_lattice_config():
        return change_groundset(matroid6_3_3.cyclic_flats())

    matroid7_3_4 = BinaryMatroid2(matrix(GF(2), [
        [1, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1],
    ]))
    if config == matroid7_3_4.cf_lattice_config():
        return change_groundset(matroid7_3_4.cyclic_flats())
    if config == matroid7_3_4.dual().cf_lattice_config():
        return change_groundset(matroid7_3_4.dual().cyclic_flats())

    matroid8_4_4 = BinaryMatroid2(matrix(GF(2), [
        [1, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
    ]))
    if config == matroid8_4_4.cf_lattice_config():
        return change_groundset(matroid8_4_4.cyclic_flats())

    # the matroid is none of the above, so it is one with nullity 2.
    # assume those are also known
    matroid5_3_2 = BinaryMatroid2(matrix(GF(2), [
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 1],
    ]))
    if config == matroid5_3_2.cf_lattice_config():
        return change_groundset(matroid5_3_2.cyclic_flats())

    # uniform matroids
    if len(config) == 2:
        poset = config.poset()
        assert poset.bottom().size == 0 and poset.bottom().rank == 0
        matroid = Uniform(poset.top().rank, poset.top().size)
        return change_groundset(matroid.cyclic_flats())


# Attempt to reconstruct the cyclic flats for a height-4 config.
# Assume matroids with height-3 lattices are known.
def cyclic_flats_height4(groundset_size, config):
    # Start by removing possible loops and isthmuses.
    loops_amount = config.bottom().size
    groundset_size -= loops_amount
    config = config.simplify()
    top_elem = config.top()
    isthmuses_amount = groundset_size - top_elem.size
    groundset = set(range(groundset_size - isthmuses_amount))
    print("Looking for a ({}, {})-matroid".format(top_elem.size, top_elem.rank))
    assert top_elem.size == len(groundset)
    assert config.bottom().size == 0 and config.bottom().rank == 0

    cyclic_flats = {elem: set() for elem in config.elements}
    cyclic_flats[config.top()].update(groundset)
    excluded = {elem: set() for elem in config.elements}

    def is_filled(vertex, cyclic_flats):
        return len(cyclic_flats[vertex]) == vertex.size

    # Check if cyclic flats for all vertices have been filled
    def done(cyclic_flats):
        return all(is_filled(vertex, cyclic_flats)
                   for vertex in config.elements)

    # Find the intersection of discovered cyclic flats of given vertices
    def intersect(cyclic_flats, vertices):
        try:
            result = cyclic_flats[vertices[0]]
        except IndexError:
            return set()
        for vertex in vertices[1:]:
            result &= cyclic_flats[vertex]
        return result

    # Pretty-print vertex index
    def idx(vertex):
        if len(config.elements) > 26:
            indices = range(1, len(config.elements) + 1)
        else:
            indices = string.ascii_uppercase
        return str(indices[vertex.index])

    def show_progress(cyclic_flats, show_excluded=True, index=True):
        lattice = LatticePoset(config.poset())
        label_str = '({0.size}, {0.rank})\n{1}'
        if show_excluded:
            label_str += '\nex. {2}'
        if index:
            label_str = '{3}\n' + label_str

        def fmt_elem(x):
            if x < 10:
                return str(x)
            return ',' + str(x)

        labels = {
            elem: label_str.format(
                elem,
                ''.join(fmt_elem(x) for x in sorted(cyclic_flats[elem])) or '{ }',
                ''.join(fmt_elem(x) for x in sorted(excluded[elem])) or '{ }',
                idx(elem) if index else None)
            for elem in config.elements
        }
        heights = {}
        for elem in config.elements:
            try:
                heights[elem.rank].append(elem)
            except KeyError:
                heights[elem.rank] = [elem]

        lattice.show(
            element_labels=labels,
            heights=heights,
            figsize=18,
            vertex_color='white',
            vertex_shape='o',
            vertex_size=8000,
        )

    # Returns a 3-tuple of:
    # - match: whether the unfilled_cfs can be filled to obtain candidate_cfs
    # - not_everywhere: a set of elements not filled into every unfilled CF
    # - somewhere: a set of elements filled into some unfilled CF
    # The last two are meaningful only if match is True.
    def sublattice_check(unfilled_cfs, candidate_cfs, groundset):
        # Dumb method: try every choice of filling cfs
        filled = {}
        unfilled_choices = {}
        for vertex, cf in unfilled_cfs.items():
            missing = vertex.size - len(cf)
            if missing <= 0:
                filled[vertex] = cf
                # Remove already filled CFs from the candidate set.
                # If a filled CF is not included there, this cannot be a match.
                if cf in candidate_cfs:
                    candidate_cfs.remove(cf)
                else:
                    return False, set(), set()
            elif missing > 0:
                unfilled_choices[vertex] = []
                for combination in itertools.combinations(groundset - cf, missing):
                    choice = cf.copy()
                    choice.update(combination)
                    unfilled_choices[vertex].append(choice)
            else:
                print(vertex.size, cf)
                raise ValueError("something went horribly wrong (overfilled cyclic flat)")

        # Convert to sorted lists, because sorting sets doesn't work as expected
        candidate_cfs = sorted(sorted(cf) for cf in candidate_cfs)

        for combination in itertools.product(*unfilled_choices.values()):
            if sorted(sorted(cf) for cf in combination) == candidate_cfs:
                not_everywhere = set()
                somewhere = set()
                for cf in combination:
                    not_everywhere |= groundset - cf
                    somewhere |= cf
                return True, not_everywhere, somewhere
        return False, set(), set()

    atoms = config.atoms()
    coatoms = config.coatoms()

    # Start by filling a coatom of maximal size.
    # Use knowledge of the matroid restricted to that coatom
    # to fill its lower covers.
    # TODO: Perhaps make sure that a height-3 sublattice is chosen here?
    largest_coatom = max(coatoms, key=lambda coatom: coatom.size)
    cyclic_flats[largest_coatom] = set(range(largest_coatom.size))
    first_cyclic_flats = cyclic_flats_height3(config.restrict(largest_coatom))

    for atom in config.lower_covers(largest_coatom):
        cf = filter(lambda x: len(x) == atom.size, first_cyclic_flats)[0]
        first_cyclic_flats.remove(cf)
        cyclic_flats[atom].update(cf)
        for coatom in config.upper_covers(atom):
            cyclic_flats[coatom].update(cf)

    # Keep track of elements chosen without loss of generality
    used = cyclic_flats[largest_coatom].copy()

    iterations = 0
    while not done(cyclic_flats):
        iterations += 1
        previous_cyclic_flats = deepcopy(cyclic_flats)
        previous_excluded = deepcopy(excluded)

        for atom in atoms:
            # For all atoms, check if we can determine the cyclic flat
            # from the elements already excluded
            if len(groundset - excluded[atom]) == atom.size:
                cyclic_flats[atom].update(groundset - excluded[atom])

            # For every atom, add elements that are present
            # in all of its upper covers
            for x in groundset - excluded[atom]:
                if all(x in cyclic_flats[coatom]
                        for coatom in config.upper_covers(atom)):
                    if x not in cyclic_flats[atom]:
                        print("(a) Added {0} to {1} ({2.size}, {2.rank}): {3}"
                              .format(x, idx(atom), atom, cyclic_flats[atom] | {x}))
                    cyclic_flats[atom].add(x)
                    used.add(x)

            # Exclude elements from coatoms if their inclusion would
            # cause an atom (the intersection of its upper covers)
            # to be larger than its size.
            # TODO: replace this with a simpler application of
            # Lemma 6 (in the LCF paper).
            if is_filled(atom, cyclic_flats):
                excluded[atom].update(groundset - cyclic_flats[atom])
                for coatom in config.upper_covers(atom):
                    other_covers = config.upper_covers(atom)
                    other_covers.remove(coatom)
                    for x in groundset - cyclic_flats[coatom]:
                        candidate = cyclic_flats[coatom].copy()
                        candidate.add(x)
                        intersection = candidate & intersect(cyclic_flats, other_covers)
                        if cyclic_flats[atom] < intersection:
                            excluded[coatom].add(x)

        for coatom in coatoms:
            # Again, check if we can determine cyclic flats
            # from the elements already excluded (this time for coatoms)
            if len(groundset - excluded[coatom]) == coatom.size:
                cyclic_flats[coatom].update(groundset - excluded[coatom])

            # If the cyclic flat is filled, simply complete the exclusions
            if is_filled(coatom, cyclic_flats):
                excluded[coatom].update(groundset - cyclic_flats[coatom])

            for atom in config.lower_covers(coatom):
                # Exclusions in coatoms must also be excluded in their
                # lower covers
                excluded[atom].update(excluded[coatom])
                # Include all elements from a coatom's lower covers
                # in the coatom
                for x in cyclic_flats[atom]:
                    if x not in cyclic_flats[coatom]:
                        print("(b) Added {0} to {1} ({2.size}, {2.rank}): {3}"
                              .format(x, idx(coatom), coatom,
                                      cyclic_flats[coatom] | {x}))
                    cyclic_flats[coatom].add(x)
                    used.add(x)

        for vertex in config.elements:
            # If all elements chosen are either included or excluded
            # for the vertex, and the cyclic flat is not filled,
            # fill it with as of yet unused elements
            if used == cyclic_flats[vertex] | excluded[vertex]:
                while not is_filled(vertex, cyclic_flats):
                    x = max(used) + 1
                    if x not in cyclic_flats[vertex]:
                        print("(c) Added {0} to {1} ({2.size}, {2.rank}): {3}"
                              .format(x, idx(vertex), vertex,
                                      cyclic_flats[vertex] | {x}))
                    cyclic_flats[vertex].add(x)
                    used.add(x)

        # Find symmetric sets of elements; that is, 
        # elements that always appear together.
        symmetric_combinations = []
        for i in range(2, len(groundset) + 1):
            symmetric_combinations.extend(itertools.combinations(groundset, i))
        for vertex in config.elements:
            for combination in symmetric_combinations:
                if not (all(x in cyclic_flats[vertex] for x in combination)
                        or all(x not in cyclic_flats[vertex] for x in combination)):
                    symmetric_combinations.remove(combination)

        # Try to add an element from a symmetric combination
        # to a vertex where it has to be included
        # (based on already filled CFs and exclusions)
        break_loop = False
        for combination in symmetric_combinations:
            combination = set(combination)
            for vertex in config.elements:
                if (not is_filled(vertex, cyclic_flats)
                        and (cyclic_flats[vertex] | excluded[vertex]
                             == groundset - combination)):
                    elem = combination.pop()
                    cyclic_flats[vertex].add(elem)
                    print("(d) Added {0} to {1} ({2.size}, {2.rank}): {3}"
                          .format(elem, idx(vertex), vertex, cyclic_flats[vertex]))
                    break_loop = True
                    break
            if break_loop:
                break

        # Did we make any progress during this iteration?
        # The next step could be computationally expensive, so we do it
        # only if none of the previous steps yielded any improvement.
        if (cyclic_flats == previous_cyclic_flats
                and excluded == previous_excluded):
            # See if we can determine something about the sublattices
            # of height <= 3 (where the top element is filled,
            # but some others are not).
            for coatom in coatoms:
                if (is_filled(coatom, cyclic_flats)
                        and any(not is_filled(atom, cyclic_flats)
                        for atom in config.lower_covers(coatom))):
                    # Find elements found in either all or none of
                    # the unfilled cyclic flats
                    everywhere = cyclic_flats[coatom].copy()
                    nowhere = cyclic_flats[coatom].copy()

                    # Check permutations of the sublattice's groundset
                    # and see if the incomplete CFs can be filled to match
                    for permutation in itertools.permutations(cyclic_flats[coatom]):
                        restricted = config.restrict(coatom)
                        unfilled_cfs = {vertex: cyclic_flats[vertex]
                                        for vertex in restricted.elements}
                        candidate_cfs = cyclic_flats_height3(
                            config.restrict(coatom), groundset=permutation)

                        match, not_everywhere, somewhere = sublattice_check(
                            unfilled_cfs, candidate_cfs, cyclic_flats[coatom])
                        if match:
                            everywhere -= not_everywhere
                            nowhere -= somewhere
                            for vertex, cf in unfilled_cfs.items():
                                if not is_filled(vertex, cyclic_flats):
                                    cyclic_flats[vertex] |= everywhere
                                    used |= everywhere
                                    excluded[vertex] |= nowhere

                    # If there was improvement, restart the while-loop
                    # to go through the faster steps
                    if (cyclic_flats != previous_cyclic_flats
                            or excluded != previous_excluded):
                        break

            # If nothing has changed, we're stuck
            if (cyclic_flats == previous_cyclic_flats
                    and excluded == previous_excluded):
                print("Nothing changed on iteration #{}; terminating"
                      .format(iterations))
                show_progress(cyclic_flats)
                return None

    # Finished
    show_progress(cyclic_flats)
    return cyclic_flats


# Tests
if __name__ == '__main__':
    matrix11_4_5 = matrix(GF(2), [
        [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    ])
    matroid11_4_5 = BinaryMatroid2(matrix11_4_5)
    h4 = matroid11_4_5.restrict({0,2,4,5,6,7,8,9,10}) # (9,4,3)-matroid
    #h4 = matroid11_4_5.restrict({0,1,3,4,5,6,7,8,9,10}) # (10,4,4)-matroid
    print(h4)
    config = h4.cf_lattice_config()

    # Check correctness (up to isomorphism)
    try:
        cfs = sorted(sorted(cf) for cf in
            cyclic_flats_height4(len(h4), h4.cf_lattice_config()).values())
        for permutation in itertools.permutations(h4.groundset()):
            permuted = BinaryMatroid2.from_matroid(h4, groundset=permutation)
            if cfs == sorted(sorted(cf) for cf in permuted.cyclic_flats()):
                print("yay")
                break
        else:
            print("nope")
    except AttributeError:
        print("search failed")