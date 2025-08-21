#!/usr/bin/env python

r"""
In this example we solve a stochastic version of the FOCUS like Stage II coil
optimisation problem: the goal is to find coils that generate a specific target
normal field on a given surface.  In this particular case we consider a vacuum
field, so the target is just zero.

The objective is given by

    J = (1/2) Mean(\int |B dot n|^2 ds)
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + ARCLENGTH_WEIGHT * ArclengthVariation

where the Mean is approximated by a sample average over perturbed coils.

The target equilibrium is the QA configuration of arXiv:2108.03711.

The coil perturbations for each coil are the sum of a 'systematic error' and a
'statistical error'.  The former satisfies rotational and stellarator symmetry,
the latter is independent for each coil.
"""

import os
import time
from pathlib import Path
from numpy.random import PCG64DXSM, Generator
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.geo import (CurveLength, CurveCurveDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                         MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation, GaussianSampler, CurvePerturbed, PerturbationSample)
from simsopt.objectives import QuadraticPenalty, MPIObjective, SquaredFlux
from simsopt.util import in_github_actions, proc0_print, comm_world
import seaborn as sns
import matplotlib.pyplot as plt
import json 

# Each SLURM array job will process one initial guess perturbation
j = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
print(f"Running initial guess perturbation {j}")


start = time.time()



# Number of Fourier modes describing each Cartesian component of each coil:
order = 24

# Standard deviation for the coil errors
SIGMA = 1e-2

# Length scale for the coil errors
L = 0.5

# Number of samples to approximate the mean
N_SAMPLES = 4

# Out-of-sample evaluation parameters
N_OOS = 1000
N_OOS_SIGMA = SIGMA

# Initial guess perturbation parameters
N_INITIAL_GUESS_PERTURBATIONS = 20
SIGMA_INITIAL_GUESS = 2e-2 # Standard deviation for the initial guess perturbation
L_INITIAL_GUESS = 0.15 # Length scale for the initial guess perturbation

# Current Parameters
CURRENT_BASE = 1e5
SIGMA_COIL = 1e-2 * CURRENT_BASE

# Pick which configuration you want
CONFIG_NAME = "QH"   # or "NCSX"

with open("input_parameters.json") as f:
    all_configs = json.load(f)

config = all_configs[CONFIG_NAME]

# Assign all keys as variables
globals().update(config)

fourier_fit = False

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 2000

# Write input parameters to file
info_txt = f"L init guess: {L_INITIAL_GUESS}, L for perturbing optimized coils = {L}\n" \
    + f"SIGMA init guess: {SIGMA_INITIAL_GUESS}, SIGMA for perturbing optimized coils = {SIGMA}" \
        
# Seed for initial guess perturbation
seed_initial_guess = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / config["surface_filename"]

# Directory for output
OUT_DIR = Path(f"output_stage_two_optimization_{CONFIG_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Create the subdirectory
SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the boundary magnetic surface; errors break symmetries, so consider the full torus
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

#######################################################
# End of input parameters.
#######################################################

#write input parameters to file
with open(SUB_DIR / 'input_parameters.txt', 'w') as f:
            f.write(info_txt)
            

#for loop over initial guess perturbation

seed_initial_guess = j

# Create the initial coils:
#initial coils for comparison, left unbothered
base_curves_init = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
curves_to_vtk(base_curves_init, OUT_DIR / f"base_curves_init")

# Perturb coils
rg_initial_guess = Generator(PCG64DXSM(seed_initial_guess))
sampler_initial_guess = GaussianSampler(base_curves_init[0].quadpoints, SIGMA_INITIAL_GUESS, L_INITIAL_GUESS, n_derivs=2)
base_curves_pert = [CurvePerturbed(c, PerturbationSample(sampler_initial_guess, randomgen=rg_initial_guess)) for c in base_curves_init]

# show initial base coil after perturbation
curves_to_vtk(base_curves_pert, OUT_DIR / f"base_curves_init_perturbed_{j}")

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
    curves_to_vtk(base_curves_fit, OUT_DIR / f"base_curves_init_fit{j}")
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
    # define base_curves to be the fitted or not
    base_curves = base_curves_fit
    
elif fourier_fit == False:
    base_curves = base_curves_pert


base_currents = [Current(CURRENT_BASE) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all() 

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

curves = [c.curve for c in coils]
currents = [c.current for c in coils]
curves_to_vtk(curves, OUT_DIR / f"curves_init_{j}")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_init_{j}", extra_data=pointData)

bs.set_points(s.gamma().reshape((-1, 3)))
# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]

seed = 0
rg = Generator(PCG64DXSM(seed))
# rg = np.random.Generator(PCG64(seed, inc=0))
sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=1)
Jfs = []
curves_pert = []
for i in range(N_SAMPLES):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    
    # Perturbing current, commented out for now to avoid changing the code
    # base_currents_perturbed = [
    #     Current(CURRENT_BASE) if c.x.size==0 else Current((c.x) + np.random.normal(scale=SIGMA_COIL))
    #     for c in base_currents
    # ]
    
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # coils = coils_via_symmetries(base_curves_perturbed, base_currents_perturbed, s.nfp, True)
    
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    
    # coils_pert = [
    #     Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), 
    #     Current(CURRENT_BASE) if c.current.x.size==0 else Current((c.current.x) + np.random.normal(scale=SIGMA_COIL))) 
    #     for c in coils
    #     ]

    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    Jfs.append(SquaredFlux(s, bs_pert))
    
for k in range(len(curves_pert)):
    if k < 15:
        curves_to_vtk(curves_pert[k], OUT_DIR / f"curves_pert_n_sample_{k}")

Jmpi = MPIObjective(Jfs, comm_world, needs_splitting=True)

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jmpi \
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
    jf = Jmpi.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"Iteration {iteration_counter}/{MAXITER}-----\n"
    outstr += f"J={J:.1e}, ⟨Jf⟩={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L>=[{msc_string}], C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    outstr += f"\n-----On {(j+1)}/{N_INITIAL_GUESS_PERTURBATIONS} Initial Guess Perturbations"
    proc0_print(outstr, flush=True)
    return J, grad

proc0_print("""
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
    proc0_print("err", (J1-J2)/(2*eps) - dJh)

proc0_print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

# Reset counter before optimization starts
iteration_counter = 0

res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 400}, tol=1e-15)
print("--------------------------------JF.x shape after opt:", JF.x.shape)
alen_string = ", ".join([f"{np.max(c.incremental_arclength())/np.min(c.incremental_arclength())-1:.2e}" for c in base_curves])
proc0_print(f"Final arclength variation max(|ℓ|)/min(|ℓ|) - 1=[{alen_string}]")

proc0_print("""
################################################################################
### Evaluate the obtained coils ################################################
################################################################################
""")

curves_to_vtk(curves, OUT_DIR / f"curves_opt_{j}")
curves_to_vtk(base_curves, OUT_DIR / f"base_curves_opt_{j}")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_opt_{j}", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))
Jf.x = res.x

# now draw some fresh samples to evaluate the out-of-sample error
rg = Generator(PCG64DXSM(seed+1))
sampler = GaussianSampler(curves[0].quadpoints, N_OOS_SIGMA, L, n_derivs=1)
b_dot_n_pert = np.zeros((qphi, qtheta)) 
squared_flux_data = []
curves_pert_oos = []
for i in range(N_OOS):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    # curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    squared_flux_data.append(SquaredFlux(s, bs_pert).J())
    if j==0 and i<15: 
        curves_pert_oos.append([c.curve for c in coils_pert])
        curves_to_vtk(curves_pert_oos[i], OUT_DIR / f"curves_pert_oos_{j}_sample_{i}")
    if (i+1) % (N_OOS/10) == 0:
        print(f"Finished {i+1}/{N_OOS} Out-of-Sample Evaluations")
        
proc0_print(f"Mean Flux Objective across perturbed coils: {Jmpi.J():.3e}")
proc0_print(f"Flux Objective for exact coils coils      : {Jf.J():.3e}")     
proc0_print(f"Out-of-sample flux value                  : {np.mean(squared_flux_data):.3e}")
proc0_print(f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}")

np.save(OUT_DIR / f"perturbed_sq_flux_data_{j}",squared_flux_data)
np.save(OUT_DIR / f"sq_flux_value_{j}",Jf.J())
np.save(OUT_DIR / f"gradient_{j}",np.linalg.norm(JF.dJ()))

# import subprocess
# if j==(N_INITIAL_GUESS_PERTURBATIONS-1):
#     subprocess.run(["python", "post_process.py", OUT_DIR, SUB_DIR, str(N_INITIAL_GUESS_PERTURBATIONS)])

end = time.time()
print(f"Total time taken: {(end - start):.2f} seconds")