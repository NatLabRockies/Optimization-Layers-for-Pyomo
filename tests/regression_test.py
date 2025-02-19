import pytest

import sys
import os
import numpy as np
import datetime
import subprocess

current_dir = os.getcwd()
sys.path.append(current_dir) # Add the parent directory to Python's search path
# Obtain the gradient ansa save them in the current_results folder.
# Paths to the folders
base_results = os.path.join(current_dir, "tests/results_12172024")
current_results = os.path.join(current_dir, "tests/results_" + datetime.date.today().strftime("%Y%m%d"))

subprocess.run(["python", os.path.join(current_dir, "tests/QP_test.py"), "1"], check=True)
subprocess.run(["python", os.path.join(current_dir, "tests/QCQP_test.py"), "1"], check=True)

# List files in both folders
base_result_files = set(os.listdir(base_results))
current_result_files = set(os.listdir(current_results))

# Identify common files
common_files = base_result_files.intersection(current_result_files)

print("Files only in Folder base_result:", base_result_files - current_result_files)
print("Files only in Folder current_result:", current_result_files - base_result_files)

@pytest.mark.parametrize("file", common_files)
def test_regression(file):
    base_array = np.load(os.path.join(base_results, file))
    current_array = np.load(os.path.join(current_results, file))
    
    if base_array.shape != current_array.shape:
        print(f"Shape mismatch in {file}: {base_array.shape} vs {current_array.shape}")
        assert not (base_array.shape[0] > 0 and current_array.shape[0] > 0)
        return
    # If the shapes are the same, compare arrays
    if np.array_equal(base_array, current_array):
        print(f"{file}: Arrays are identical.")
    assert np.allclose(base_array, current_array, atol=1e-5, equal_nan=False)

    # Compute difference summary
    diff = np.abs(base_array - current_array)
    print(f"{file}: Differences found. Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
