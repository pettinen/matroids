from __future__ import print_function
import random as py_random
import sys
import time

from sage.all import *

from binary_matroid import BinaryMatroid2


# Search for two non-isomorphic matroids with the same
# configurations of lattices of cyclic flats.
def find_collision(rows, cols, strategy='exhaustive', print_progress=True):
    global n
    n = 0
    results = {}

    def iteration(matrix_):
        global n
        n += 1
        if print_progress:
            print('\r{}'.format(n), end='', file=sys.stderr)

        matroid = BinaryMatroid2(matrix=matrix_)
        # Only consider simple matroids with no isthmuses
        if not matroid.is_simple() or matroid.coloops():
            return

        config = matroid.cf_lattice_config()
        if config in results:
            if not matroid.is_isomorphic(results[config][0]):
                print("\nFound:")
                print(matrix_)
                print('--------')
                print(results[config][1])
                config.show(index=True)
                return config
        else:
            results[config] = (matroid, matrix_)

    start = time.time()
    if strategy == 'exhaustive':
        print("Searching {}x{} matrices exhaustively...".format(ROWS, COLS),
            file=sys.stderr)
        for matrix_ in MatrixSpace(GF(2), rows, cols):
            if iteration(matrix_) is not None:
                break
    elif strategy == 'random':
        print("Searching {}x{} matrices at random...".format(ROWS, COLS),
            file=sys.stderr)
        while True:
            matrix_ = matrix(GF(2), [[py_random.randint(0, 1)
                                      for _ in range(cols)]
                                      for _ in range(rows)])
            if iteration(matrix_) is not None:
                break
    end = time.time()
    if print_progress:
        print()
    print('{} different configurations'.format(len(results)),
        file=sys.stderr)
    print("Completed in {} seconds".format(round(end - start, 2)),
        file=sys.stderr)
    for config in results:
        config.show(title="{}\n{}".format(*results[config]))


# Find a matroid with a given LCF configuration.
def find_matroid_by_config(rows, cols, search_config, print_progress=True):
    global n
    n = 0

    def iteration(matrix_):
        global n
        n += 1
        if print_progress:
            print('\r{}'.format(n), end='', file=sys.stderr)

        matroid = BinaryMatroid2(matrix=matrix_)
        # Only consider simple matroids with no isthmuses
        if not matroid.is_simple() or matroid.coloops():
            return

        candidate_config = matroid.cf_lattice_config()
        if candidate_config == search_config:
            print("\nFound:")
            print(matroid)
            print(matrix_)
            print('--------')
            return matroid

    start = time.time()
    print("Searching {}x{} matrices exhaustively...".format(ROWS, COLS),
        file=sys.stderr)
    for matrix_ in MatrixSpace(GF(2), rows, cols):
        if iteration(matrix_) is not None:
            break
    end = time.time()
    if print_progress:
        print()
    print("Completed in {} seconds".format(round(end - start, 2)),
        file=sys.stderr)


# Find a matroid with given cyclic_flats.
def find_matroid_by_cf(rows, cols, search_cf, print_progress=True):
    global n
    n = 0

    def iteration(matrix_):
        global n
        n += 1
        if print_progress:
            print('\r{}'.format(n), end='', file=sys.stderr)

        matroid = BinaryMatroid2(matrix=matrix_)
        # Only consider simple matroids with no isthmuses
        #if not matroid.is_simple() or matroid.coloops():
            #return

        candidate_cf = matroid.cyclic_flats()
        if candidate_cf == search_cf:
            print("\nFound:")
            print(matroid)
            print(matrix_)
            print('--------')
            return matroid

    start = time.time()
    print("Searching {}x{} matrices exhaustively...".format(ROWS, COLS),
        file=sys.stderr)
    for matrix_ in MatrixSpace(GF(2), rows, cols):
        if iteration(matrix_) is not None:
            break
    end = time.time()
    if print_progress:
        print()
    print("Completed in {} seconds".format(round(end - start, 2)),
        file=sys.stderr)


if __name__ == '__main__':
    search_cf = {
        frozenset({0,1,2,3,4,5,6}),
        frozenset({0,1,2,3,4}),
        frozenset({0,1,5,6}),
        frozenset({0,2,5,6}),
        frozenset({0,2,3}),
        frozenset({1,2,4}),
        frozenset({}),
    }
    ROWS = 4
    COLS = 7
    STRATEGY = 'exhaustive'
    PRINT_PROGRESS = True
    find_matroid_by_cf(ROWS, COLS, search_cf, PRINT_PROGRESS)
