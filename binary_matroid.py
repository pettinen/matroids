from __future__ import print_function
from collections import Counter
from six.moves import range
import itertools
import operator
import sys

from sage.all import *
from sage.matroids.advanced import BinaryMatroid

from configuration import Configuration, Element


# Various functions moved into a class.
# Most methods authored by Matthias Grezet.
class BinaryMatroid2(BinaryMatroid):
    # Arguments:
    #    ranks: A rank or a list of ranks to find cyclic flats of.
    #           Optional, defaults to checking all ranks.
    #    print_progress: Print in real time where the program is.
    # Returns: The list of cyclic flats of the matroid.
    def cyclic_flats(self, ranks=None):
        if ranks is None:
            ranks = range(self.full_rank() + 2)
        else:
            try:              # Did we get a list of ranks?
                iter(ranks)
            except TypeError: # Nope, assume we got a single integer
                ranks = [ranks]
        results = set()
        dual = self.dual()
        groundset = self.groundset()
        for rank in ranks:
            flats = self.flats(rank)
            for index, flat in enumerate(flats, start=1):
                if dual.is_closed(groundset.difference(flat)):
                    results.add(flat)
        return results

    # Returns: Boolean indicating whether a subset is cyclic.
    def is_cyclic(self, subset):
        return self.dual().is_closed(self.groundset().difference(subset))

    # Returns: Boolean indicating whether a subset
    #          is a cyclic flat of the matroid.
    def has_cyclic_flat(self, subset):
        return (self.is_closed(subset)
                and self.dual().is_closed(self.groundset().difference(subset)))

    # Returns: Flats of rank r; or all flats, if r is None.
    def flats(self, r=None):
        if r is not None:
            return super(BinaryMatroid2, self).flats(r)

        flats = []
        for rank in range(self.full_rank() + 1):
            flats.extend(self.flats(rank))
        # Return a SetSystem to be consistent with Sage's flats method.
        return sage.matroids.set_system.SetSystem(list(self.groundset()), flats)

    # Returns: The poset formed by the cyclic flats of the matroid
    #          and the inclusion operation.
    def cyclic_flats_poset(self):
        cyclic_flats = self.cyclic_flats()
        return Poset((cyclic_flats, operator.le))

    def cf_lattice(self):
        cyclic_flats = self.cyclic_flats()
        return LatticePoset(Poset((cyclic_flats, operator.le)))

    def cf_lattice_config(self):
        cyclic_flats = self.cyclic_flats()
        labels = {cf: Element(len(cf), self.rank(cf), i)
                  for i, cf in enumerate(cyclic_flats)}
        poset = Poset((cyclic_flats, operator.le), element_labels=labels)
        return Configuration(labels.values(), poset.cover_relations())

    # Returns: Number of atoms (in the lattice of cyclic flats) for each rank.
    def atoms(self):
        lattice = LatticePoset(self.cyclic_flats_poset())
        return Counter(self.rank(atom) for atom in lattice.atoms())

    def parallel_elements(self):
        return filter(lambda x: len(x) == 2, self.circuits())

    def minimum_distance(self):
        groundset = self.groundset()
        full_rank = self.full_rank()
        for d in range(len(groundset) + 1):
            for subset in itertools.combinations(groundset, d):
                if self.rank(groundset - set(subset)) < full_rank:
                    return d
        return None  # not sure what d is defined as when groundset has rank 0

    def __repr__(self):
        return "Binary ({}, {}, {})-matroid".format(
            len(self), self.full_rank(), self.minimum_distance())
