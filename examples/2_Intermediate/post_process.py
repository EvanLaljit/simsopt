
# post.py
import sys
import os
import numpy as np
from pathlib import Path

sq_flux_values = []
mean_perturbed_sq_flux_values = []
gradients = []
def postprocess(in_dir,out_dir,n_initial_perturbations):
    print(f"Running postprocessing.")
    # load data from out_dir
    # save postprocessed data to out_dir
    for i  in range(n_initial_perturbations):
        gradients.append(np.load(in_dir / f"gradient_{i}.npy"))
        sq_flux_values.append(np.load(in_dir / f"sq_flux_value_{i}.npy"))
        mean_perturbed_sq_flux_values.append(np.mean(np.load(in_dir / f"perturbed_sq_flux_data_{i}.npy")))
        
    header_string = 'Jf.J(), <Perturbed Jf.J()>, ||∇J||'
    combined_array = np.column_stack((np.array(sq_flux_values),np.array(mean_perturbed_sq_flux_values),np.array(gradients)))
    np.savetxt(out_dir / 'J_Values.txt', 
            combined_array, delimiter=',', header = header_string, comments='')
    print("Post processing done.")

if __name__ == "__main__":
    in_dir = Path(sys.argv[1])  # first argument passed in
    out_dir = Path(sys.argv[2])
    n_initial_perturbations = int(sys.argv[3])
    postprocess(in_dir,out_dir,n_initial_perturbations)
