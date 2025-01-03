#!/bin/bash
#SBATCH --account=drl4dsr
#SBATCH --time=1:00:00
#SBATCH --job-name=job
#SBATCH --partition=debug

module load anaconda3
conda activate /projects/drl4dsr/kchen2/OPT_layer_env
module use /nopt/nrel/apps/cpu_stack/software/idaes_solvers/modules/test/
module load netlib-lapack
module load idaes_solvers

# python /projects/drl4dsr/kchen2/Opt_Layer/tests/QP_test.py
# python /projects/drl4dsr/kchen2/Opt_Layer/tests/QCQP_test.py
# python /projects/drl4dsr/kchen2/Opt_Layer/examples/ReLULayers.py
# python /projects/drl4dsr/kchen2/Opt_Layer/examples/Convex_approximate_dynamic_programming.py