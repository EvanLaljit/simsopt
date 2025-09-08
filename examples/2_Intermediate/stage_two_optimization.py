#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
import time 
from pathlib import Path
import numpy as np
import json
from numpy.random import PCG64DXSM, Generator
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation,
                         GaussianSampler, CurvePerturbed, PerturbationSample)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions
import seaborn as sns
from simsopt.util.famus_helpers import FocusPlasmaBnormal
import matplotlib.pyplot as plt


start_time = time.time()

# Each SLURM array job will process one initial guess perturbation
slurm_array_int = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
print(f"Running initial guess perturbation {slurm_array_int}")


# Number of Fourier modes describing each Cartesian component of each coil:
order = 24

# Number of times to perturb initial guess and run optimization for each
N_INITIAL_GUESS_PERTURBATIONS = 20
SIGMA_INITIAL_GUESS = 1e-2*0 # Standard deviation for the initial guess perturbation
L_INITIAL_GUESS = 0.15 # Length scale for the initial guess perturbation

# Number of samples for out-of-sample evaluation
N_OOS = 1000

#save pairs of sigma and L values to test for perturbing coils
sigma_values = np.linspace(1e-3, 3.5e-2,8)
L_values = np.linspace(0.4, 1.5,4)
sigma_and_L = [(sigma, L) for sigma in sigma_values for L in L_values]

# Standard deviation for the coil errors
# Length scale for the coil errors
SIGMA_OOS, L_OOS = sigma_and_L[slurm_array_int]

if SIGMA_INITIAL_GUESS != 0:
    SIGMA_OOS, L_OOS = 1e-2, 0.5

# Choose and load input parameters from configuration
CONFIG_NAME = "QA" 

# use fourier fitting
fourier_fit = False

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1000

#######################################################
# End of input parameters.
#######################################################

#load configuration
with open("input_parameters.json") as f:
    all_configs = json.load(f)

config = all_configs[CONFIG_NAME]

# Assign all keys as variables
globals().update(config)

# Write input parameters to file
info_txt = f"L init guess: {L_INITIAL_GUESS}, L for perturbing optimized coils = {L_OOS}\n" \
    + f"SIGMA init guess: {SIGMA_INITIAL_GUESS}, SIGMA for perturbing optimized coils = {SIGMA_OOS}" \

#specify what to label results for each run
#currently being saved and labeled:
#initial curves, perturbed initial curves, initial coils, 
#base curves optimized, coils optimized, 
loop_label = slurm_array_int
#label for numerical data, like arrays or floats
#unperturbed sq flux, gradient, perturbed sq flux distribution
loop_numerical_data_label = slurm_array_int

# File for the desired boundary magnetic surface:

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / config["surface_filename"]

# Directory for output
out_dir_path = f"output_stage_two_optimization_{CONFIG_NAME}_LW_{LENGTH_WEIGHT}_CCDW_{CC_WEIGHT}"
if fourier_fit == True:
    out_dir_path += "_ffit"
    
if SIGMA_INITIAL_GUESS == 0:
    out_dir_path += "_original_3.5cm_sigma_max"
elif SIGMA_INITIAL_GUESS != 0:
    out_dir_path += "_pert_init"
    
if CS_WEIGHT == 0:
    out_dir_path += "_CS_WEIGHT_0"
    
if MAXITER != 1000:
    out_dir_path += f"_{MAXITER/1000}kiter"
    
OUT_DIR = Path(out_dir_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)

#save sigma and L values used, for plotting 
# np.save(OUT_DIR / f"Sigma_{slurm_array_int}",SIGMA)
# np.save(OUT_DIR / f"L_{slurm_array_int}",L)

# Create the subdirectory
SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the boundary magnetic surface:
nphi = 64
ntheta = 16
# Pick the correct constructor dynamically
surface_constructor = getattr(SurfaceRZFourier, config["surface_method"])

s = surface_constructor(
    filename=filename,
    range="full torus",
    nphi=nphi,
    ntheta=ntheta
)

qphi = 2 * nphi
qtheta = 64
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = surface_constructor(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)


#write input parameters to file
with open(SUB_DIR / 'input_parameters.txt', 'w') as f:
            f.write(info_txt)
 
#for loop over initial guess perturbation

seed_initial_guess = slurm_array_int

# Create the initial coils:

base_curves_init = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
curves_to_vtk(base_curves_init, OUT_DIR / f"base_curves_init")
    
# Perturb coils
rg_initial_guess = Generator(PCG64DXSM(seed_initial_guess))
sampler_initial_guess = GaussianSampler(base_curves_init[0].quadpoints, SIGMA_INITIAL_GUESS, L_INITIAL_GUESS, n_derivs=2)
base_curves_pert = [CurvePerturbed(c, PerturbationSample(sampler_initial_guess, randomgen=rg_initial_guess)) for c in base_curves_init]

# show initial base coil after perturbation
curves_to_vtk(base_curves_pert, OUT_DIR / f"base_curves_init_perturbed_{loop_label}")


#take x,y,z coordinates from perturbed coil and fit fourier
if fourier_fit == True: 

    base_curves_fit = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)

    theta = np.array(base_curves_fit[0].quadpoints)  # This gives you 0 to 1 (not 0 to 2π)
    for c in range(ncoils):
        #for each coordinate, find coefficients
        coeffs_for_all_coords = []
        for coordinate in range(3):
            x = base_curves_pert[c].gamma()[:,coordinate]
            # x = np.append(x,x[0]) #enforce periodicity for lstsq solver
            basis = []
            for m in range(order+1):
                if m == 0:
                    basis.append(np.ones_like(theta))  # Constant term (m=0)
                else:
                    basis.append(np.sin(2*np.pi*m*theta))  # sin(2π*m*phi) 
                    basis.append(np.cos(2*np.pi*m*theta))  # cos(2π*m*phi)
            A = np.column_stack(basis)
            coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
            coeffs_for_all_coords = np.append(coeffs_for_all_coords,coeffs)
        base_curves_fit[c].x = coeffs_for_all_coords

    # Plot base curves obtained from fitted coefficients
    curves_to_vtk(base_curves_fit, OUT_DIR / f"base_curves_init_fit{loop_label}")
    #print rms error
    # More detailed error analysis
    fit_error = []
    for c in range(ncoils):
        original_points = base_curves_pert[c].gamma()
        fitted_points = base_curves_fit[c].gamma()
        
        # Interpolate fitted curve to match original points for fair comparison
        from scipy.interpolate import interp1d
        
        # Create parameterization for fitted curve
        theta_fitted = np.linspace(0, 2*np.pi, fitted_points.shape[0], endpoint=False)
        
        # Interpolate each coordinate
        fitted_interp = []
        for coord in range(3):
            f_interp = interp1d(theta_fitted, fitted_points[:, coord], kind='cubic', assume_sorted=True)
            theta_original = np.linspace(0, 2*np.pi, original_points.shape[0], endpoint=False)
            fitted_interp.append(f_interp(theta_original))
        
        fitted_interp = np.column_stack(fitted_interp)
        
        # Now calculate error with same number of points
        point_errors = np.sqrt(np.sum((original_points - fitted_interp)**2, axis=1))
        rms_error = np.sqrt(np.mean(point_errors**2))
        
        # Relative error
        curve_size = np.max(np.linalg.norm(original_points, axis=1)) - np.min(np.linalg.norm(original_points, axis=1))
        relative_error = rms_error / curve_size
        
        fit_error.append(rms_error)
        
        print(f"Coil {c}: RMS error: {rms_error:.6f}, Relative: {relative_error:.6f}")
        print(f"Max point error: {np.max(point_errors):.6f}, Min point error: {np.min(point_errors):.6f}")

    print(f"Overall fit errors: {fit_error}")
    print(f"Mean fit error: {np.mean(fit_error):.6f}")
    with open(SUB_DIR / 'fit_errors.txt', 'w') as f:
            f.write(f"{np.mean(fit_error)}")
    # define base_curves to be the fitted or not
    base_curves = base_curves_fit
elif fourier_fit == False:
    base_curves = base_curves_pert

base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR / f"curves_init_{loop_label}") 

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_init_{loop_label}", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:

JF = Jf \
    + LENGTH_WEIGHT * sum(Jls) \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + ARCLENGTH_WEIGHT * sum(Jals) \
    + CS_WEIGHT * Jcsdist \
    
# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize

iteration_counter = 0
def fun(dofs):
    global iteration_counter
    iteration_counter += 1
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"Iteration {iteration_counter}/{MAXITER}-----\n"
    outstr += f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    outstr += f"\n-----On {(slurm_array_int+1)}/{N_INITIAL_GUESS_PERTURBATIONS} Initial Guess Perturbations"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")

f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)
    
print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

# Reset counter before optimization starts
iteration_counter = 0

res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol = 1e-15)

for c in base_currents:
    print(f"-----------------Base current: {c.x}")
    
curves_to_vtk(curves, OUT_DIR / f"curves_opt_{loop_label}")
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_opt_{loop_label}", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))

# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller `B·n` on the surface.
# dofs = res.x
# LENGTH_WEIGHT *= 0.1
# res = minimize(fun, x0, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
# curves_to_vtk(curves, OUT_DIR / f"curves_opt_long_{j}")
# bs.set_points(s_plot.gamma().reshape((-1, 3)))
# pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
# s_plot.to_vtk(OUT_DIR / f"surf_opt_long_{j}", extra_data=pointData)

bs.set_points(s.gamma().reshape((-1, 3)))
curves_to_vtk(base_curves, OUT_DIR / f"base_curves_opt_{loop_label}")
# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
# bs.save(OUT_DIR / "biot_savart_opt.json")
sq_flux_unperturbed = Jf.J()

#Perturb coils
seed = 0
curves_pert = []
squared_flux_data = []
curves_pert_oos = []
rg = Generator(PCG64DXSM(seed+1))
sampler = GaussianSampler(curves[0].quadpoints, SIGMA_OOS, L_OOS, n_derivs=1)
for i in range(N_OOS):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    # Squared Flux calculation
    bs_pert = BiotSavart(coils_pert)
    bs_pert.set_points(s.gamma().reshape((-1, 3)))
    squared_flux_data.append(SquaredFlux(s, bs_pert).J())
    #only save first 15 samples, for first initial guess
    if slurm_array_int==0 and i<15: 
        curves_pert_oos.append([c.curve for c in coils_pert])
        curves_to_vtk(curves_pert_oos[i], OUT_DIR / f"curves_pert_oos_{loop_label}_sample_{i}")
    #print progress
    if (i+1) % (N_OOS/10) == 0:
        print(f"Finished {i+1}/{N_OOS} Out-of-Sample Evaluations")
        
print(f"Flux Objective for exact coils coils      : {sq_flux_unperturbed:.3e}")
print(f"Out-of-sample flux value                  : {np.mean(squared_flux_data):.3e}")
print(f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}")

np.save(OUT_DIR / f"perturbed_sq_flux_data_{loop_numerical_data_label}",squared_flux_data)
np.save(OUT_DIR / f"sq_flux_value_{loop_numerical_data_label}",Jf.J())
np.save(OUT_DIR / f"gradient_{loop_numerical_data_label}",np.linalg.norm(JF.dJ()))

# import subprocess
# if j==(N_INITIAL_GUESS_PERTURBATIONS-1):
#     subprocess.run(["python", "post_process.py", OUT_DIR, SUB_DIR, str(N_INITIAL_GUESS_PERTURBATIONS)])

end_time = time.time()
print(f"Took {end_time-start_time}s")
