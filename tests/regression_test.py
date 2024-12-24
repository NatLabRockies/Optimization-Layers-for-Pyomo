import sys
import os
import numpy as np

current_dir = os.getcwd()
sys.path.append(current_dir) # Add the parent directory to Python's search path

# Paths to the folders
base_results = os.path.join(current_dir, "results_12172024")
current_results = os.path.join(current_dir, "results")

# List files in both folders
base_result_files = set(os.listdir(base_results))
current_result_files = set(os.listdir(current_results))

# Identify common files
common_files = base_result_files.intersection(current_result_files)

print("Files only in Folder base_result:", base_result_files - current_result_files)
print("Files only in Folder current_result:", current_result_files - base_result_files)

# Compare common files
global Pass, Pass_tol
Pass, Pass_tol = True, True

for file in common_files:
    base_array = np.load(os.path.join(base_results, file))
    current_array = np.load(os.path.join(current_results, file))
    
    if base_array.shape != current_array.shape:
        print(f"Shape mismatch in {file}: {base_array.shape} vs {current_array.shape}")
        if base_array.shape[0] > 0 and current_array.shape[0] > 0:
            Pass = False
            Pass_tol = False
    else:
        # Compare arrays
        if np.array_equal(base_array, current_array):
            print(f"{file}: Arrays are identical.")
        elif np.allclose(base_array, current_array, rtol=1e-05, atol=1e-08, equal_nan=False):
            Pass = False
            print(f"{file}: Arrays are close within relative tolerance 1e-5.")
        else:
            # Compute difference summary
            Pass = False
            Pass_tol = False
            diff = np.abs(base_array - current_array)
            print(f"{file}: Differences found. Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")

def test():
    assert Pass == True

def test_tol():
    assert Pass_tol == True