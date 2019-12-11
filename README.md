## Configuration of lattices of cyclic flats of binary matroids

Code related to my BSc thesis on the topic of binary matroids.
Specifically, the configurations of their lattices of cyclic flats
and the hypothesized uniqueness of said configurations.

Although configuration uniqueness is currently just a conjecture,
I am currently working on code to reconstruct the cyclic flats
from height-3 and height-4 configurations (and assuming that succeeds,
hopefully generalizing from there.)

A rough overview of what this repository contains:
- binary_matroid.py: a subclass of Sage&rsquo;s BinaryMatroid augmented with utility methods, most relating to cyclic flats and their lattice configurations
- configuration.py: a Configuration class (building upon Sage&rsquo;s LatticePoset) and work-in-progress algorithms for reconstructing cyclic flats
- search.py: brute-force search for non-isomorphic matroids with identical configurations (one of the first things I did)
- test scripts for whatever I&rsquo;m current working on
- a usage example which is (hopefully) self-evident

#### Benchmarks (exhaustive search for collisions through _m_&times;_n_ binary matrices on an i7-9700K):
* 4&times;4: 24&#8239;s
* 4&times;5: 7&#8239;min 37&#8239;s
* 4&times;6: 2&#8239;h 42&#8239;min
* 5&times;4: 6&#8239;min 20&#8239;s
* 5&times;5: 3&#8239;h 50&#8239;min
