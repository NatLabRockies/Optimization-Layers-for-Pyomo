# PyomoLayers

## Installation
```bash
pip install '.[test]'
idaes get-extensions
```

## Regression test
```bash
pytest tests/regression_test.py
```

## Linear Solvers
#### For higher performance, you might want to install MUMPS or MA27. Here we show installation instructions for MUMPS.
```bash
conda install conda-forge::mumps-include
conda install conda-forge::mumps-mpi
pip install pymumps
```
