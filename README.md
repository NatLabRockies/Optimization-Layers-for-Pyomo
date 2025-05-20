# Opt_Layer
# Requirements
```bash
PyNumero, PyMUMPS, Homebrew
```
# Regression test
```bash
pytest Opt_Layer/tests/regression_test.py 
```
# Installation 
```bash
conda create --name OPT_layer_env python=3.11.0
conda activate OPT_layer_env
conda install conda-forge::mumps-include
conda install conda-forge::mumps-mpi
brew install mpich
pip install pymumps
conda install -c conda-forge ipopt
cd Opt_Layer
# Remove nvidia and triton packages in requirements.txt if installing on the local laptop 
# conda install ipopt on the local laptop 
pip install -e .
```


