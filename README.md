# Opt_Layer

# Regression test
```bash
pytest Opt_Layer/tests/regression_test.py 
```
# Installation 
```bash
# python 3.12 and 3.13 require scipy = 1.14 while scipy should use the 1.10 version because 1.14 version gives an error about singularity when running "spsolve(kkt.tocsc(), -kkt_rhs.tocsc())" for QCQP.py.
conda create --name OPT_layer_env python=3.11.0
conda activate OPT_layer_env
cd Opt_Layer
# Remove nvidia and triton packages in requirements.txt if installing on the local laptop 
# conda install ipopt on the local laptop 
pip install -e .
```

