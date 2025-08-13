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

start = time.time()
# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 3

# Major radius for the initial circular coils:
R0 = 1.3

# Minor radius for the initial circular coils:
R1 = 0.8

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Weight on the curve lengths in the objective function:
LENGTH_WEIGHT = 8e-4

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
DISTANCE_THRESHOLD = 0.2
DISTANCE_WEIGHT = 100

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.2
CS_WEIGHT = 1e-4

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5.
MSC_WEIGHT = 1e-6

# Weight for the arclength variation penalty in the objective function:
ARCLENGTH_WEIGHT = 1e-2

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
N_INITIAL_GUESS_PERTURBATIONS = 8
SIGMA_INITIAL_GUESS = 1e-2 # Standard deviation for the initial guess perturbation
L_INITIAL_GUESS = 0.2 # Length scale for the initial guess perturbation

# Current Parameters
CURRENT_BASE = 1e5
SIGMA_COIL = 1e-2 * CURRENT_BASE

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1000

# Write input parameters to file
info_txt = f"L init guess: {L_INITIAL_GUESS}, L for perturbing optimized coils = {L}\n" \
    + f"SIGMA init guess: {SIGMA_INITIAL_GUESS}, SIGMA for perturbing optimized coils = {SIGMA}" \
        
# Seed for initial guess perturbation
seed_initial_guess = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.NCSX_c09r00_halfTeslaTF'

# Directory for output
OUT_DIR = Path("output_stage_two_optimization_stochastic_NCSX")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Create the subdirectory
SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the boundary magnetic surface; errors break symmetries, so consider the full torus
nphi = 64
ntheta = 16
s = SurfaceRZFourier.from_focus(filename, 
                                     range="full torus", nphi=nphi, ntheta=ntheta)

qphi = 2 * nphi
qtheta = 64
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
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
            
#define figure for kde plot
fig, ax = plt.subplots(figsize=(10,10))
#for loop over initial guess perturbation
sq_flux_values = []
mean_perturbed_sq_flux_values = []
gradients = []
for j in range(N_INITIAL_GUESS_PERTURBATIONS):
    seed_initial_guess += 1
    
    # Create the initial coils:
    #initial coils for comparison, left unbothered
    base_curves_init = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    #create coils to perturb
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    curves_to_vtk(base_curves, OUT_DIR / f"base_curves_init")
    
    # for i, c in enumerate(base_curves):
        
    #     dofs_per_coordinate = order*2 + 1
    #     indices = np.concatenate([[1,1], np.arange(2,dofs_per_coordinate + 1)])
    #     # indices = np.concatenate([[1,1], np.arange(2,dofs_per_coordinate + 1)])
    #     # v = c.x.reshape((-1, dofs_per_coordinate))
    #     # v = v/indices**2
        
    #     # Create mode numbers for Fourier coefficients
    #     # [0, 1, 1, 2, 2, 3, 3, ..., order, order]
    #     mode_numbers = np.zeros(dofs_per_coordinate)
    #     mode_numbers[0] = 0  # constant term
    #     for m in range(1, order + 1):
    #         mode_numbers[2*m-1] = m  # sine coefficient for mode m
    #         mode_numbers[2*m] = m    # cosine coefficient for mode m
            
    #     # Use different seed for each curve to ensure independent perturbations
    #     np.random.seed(seed_initial_guess + i)
    #     v = np.random.normal(scale=SIGMA_INITIAL_GUESS, size=(3,dofs_per_coordinate))
    #     # Avoid division by zero for constant term (m=0)
    #     mode_numbers_squared = np.where(mode_numbers == 0, 1, mode_numbers**2)
    #     v = v / mode_numbers_squared
    #     c.x += v.flatten()
    
    for i, c in enumerate(base_curves):
        np.random.seed(seed_initial_guess + i)
        c.x += np.random.normal(scale=SIGMA_INITIAL_GUESS, size=c.x.shape)
        
    # rg_initial_guess = Generator(PCG64DXSM(seed_initial_guess))
    # sampler_initial_guess = GaussianSampler(base_curves[0].quadpoints, SIGMA_INITIAL_GUESS, L_INITIAL_GUESS, n_derivs=2)
    # base_curves = [CurvePerturbed(c, PerturbationSample(sampler_initial_guess, randomgen=rg_initial_guess)) for c in base_curves]
    
    for i in range(ncoils):
        print(f"Coils Match: {np.array_equal(base_curves[i].x,base_curves_init[i].x)}" )
        
    # show initial base coil after perturbation
    curves_to_vtk(base_curves, OUT_DIR / f"base_curves_init_perturbed_{j}")
        
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
    Jdist = CurveCurveDistance(curves, DISTANCE_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]

    # for c in base_currents:
    #     print(f"-----------------Base current: {c.x}")
    #     print(f"-----------------Perturbed current: {c.x + np.random.normal(scale=SIGMA_COIL)}")
    
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
        + DISTANCE_WEIGHT * Jdist \
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
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L>=[{msc_string}], C-C-Sep={Jdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        outstr += f"\n-----On {(j+1)}/{N_INITIAL_GUESS_PERTURBATIONS} Initial Guess Perturbations"
        proc0_print(outstr, flush=True)
        return J, grad
    
    # Perturb initial guess
    # seed_initial_guess += 1
    # np.random.seed(seed_initial_guess)
    # base_x = JF.x.copy()
    # x0 = base_x + np.random.normal(scale=SIGMA_INITIAL_GUESS,size=base_x.shape)
    # JF.x = x0
    #show initial coil after perturbation
    # curves_to_vtk(curves, OUT_DIR + f"curves_init_perturbed_{j}")
    
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
    proc0_print(f"Mean Flux Objective across perturbed coils: {Jmpi.J():.3e}")
    proc0_print(f"Flux Objective for exact coils coils      : {Jf.J():.3e}")

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
            
    proc0_print(f"Out-of-sample flux value                  : {np.mean(squared_flux_data):.3e}")
    proc0_print(f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}")
    
    sns.kdeplot(squared_flux_data,fill=False,ax=ax, label=f'Initial Guess {j+1}')
    np.save(OUT_DIR / f"stochastic_perturbed_sq_flux_data_{j}",squared_flux_data)
    sq_flux_values.append(Jf.J())
    mean_perturbed_sq_flux_values.append(np.mean(squared_flux_data))
    gradients.append(np.linalg.norm(JF.dJ()))
    print(f"Finished {(j+1)}/{N_INITIAL_GUESS_PERTURBATIONS} Initial Guess Perturbations")
    
    time.sleep(5)
    
    
# Save squared flux data for analysis
# plt.hist(squared_flux_data, bins=50, density=True, edgecolor='black')
plt.xlabel("Squared Flux")
plt.ylabel("Count")
plt.title(f"Distribution of Squared Flux Values for {N_INITIAL_GUESS_PERTURBATIONS} Initial"
          " Guess Perturbations")
plt.savefig(SUB_DIR / "squared_flux_distribution.png")
plt.close()

# Save unperturbed and mean perturbed values
header_string = 'Jf.J(), <Perturbed Jf.J()>, ||∇J||'
combined_array = np.column_stack((np.array(sq_flux_values),np.array(mean_perturbed_sq_flux_values),np.array(gradients)))
np.savetxt(SUB_DIR / 'J_Values.txt', 
           combined_array, delimiter=',', header = header_string, comments='')

end = time.time()
print(f"Total time taken: {(end - start):.2f} seconds")