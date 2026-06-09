# PyomoLayers (SWR 25-132)
An optimization layer solves an optimization problem using Pyomo and IPOPT during the forward pass.
It computes the gradient of the optimal solution with respect to the parameters based on the KKT conditions in the backward pass.
## Documentation
- Source docs: [docs](docs/)
- Build locally:
```bash
cd docs
make html
```
- Open `docs/_build/html/index.html` in a browser.

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
