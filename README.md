# Opt_Layer

## Regression test
# run pytest Opt_Layer/tests/regression_test.py 

## Installation 
```bash
conda create --name OPT_layer_env python=3.11.0
conda activate OPT_layer_env
cd Opt_Layer
# Remove nvidia and triton packages in requirements.txt if installing on the local laptop 
# conda install ipopt on the local laptop 
pip install -e .


