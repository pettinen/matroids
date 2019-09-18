from __future__ import print_function
import random as py_random
import sys
import time

from sage.all import *

from binary_matroid import BinaryMatroid2
from configuration import Configuration


def search(rows, cols, strategy='exhaustive', print_progress=True):
    global n
    n = 0
    results = {}

    def iteration(matrix_):
        global n
        n += 1
        if print_progress:
            print('\r{}'.format(n), end='', file=sys.stderr)

        matroid = BinaryMatroid2(matrix=matrix_)
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


if __name__ == '__main__':
    ROWS = 4
    COLS = 5
    STRATEGY = 'exhaustive'
    PRINT_PROGRESS = True
    search(ROWS, COLS, STRATEGY, PRINT_PROGRESS)
