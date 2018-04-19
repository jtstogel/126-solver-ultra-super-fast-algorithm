# 126-solver-ultra-super-fast-algorithm

## Notes
Our project is written in Python 3 and only been run using the 3.6.3 Anaconda version. It certainly will not work with any Python 2 version but has not been tested with other Python 3 versions.

We do have one global variable in our program, so please do not run any action calls or multiple simulations in parallel.

## Installation
We wrote a C extension for python to assist in computing hitting time.
To compile this, please cd into our directory and run `sh build_extension.sh`
It will compile the extension in `hittingtime_numpyextension` and move the created Python module back into the directory containing our catan code.

This extension is written only for Python 3.5 and above. If it does not compile please let us know, as it has compiled on every unix machine we have tried it on. The `build_extension.sh` function just cd's into `hittingtime_numpyextension` and builds the extension.

## Files
The `catanAction.py` file contains our `action` and `planBoard` functions. Please only import these two functions when testing our code. This file depends on our `hittingtime.py` file and the given `catan.py` file.

The `test_strat.py` file contains our testing code, which generates 100 boards and runs a single simulation on each.
