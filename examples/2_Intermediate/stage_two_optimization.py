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


#CHANGES--------------------------------
#cc_weight 1000 to 10
#cs weight irrelevant now, commented out Jcsdist and its contribution to JF
#added ArclengthVariation to the objective function, Jals
#---------------------------------------

start_time = time.time()
# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 24

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT_INPUT = 1e-7

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 10
 
# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 0

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# Weight for the arclength variation penalty in the objective function:
ARCLENGTH_WEIGHT = 1e-2

# Standard deviation for the coil errors
SIGMA = 1e-2

# Length scale for the coil errors
L = 0.5

# Number of samples for out-of-sample evaluation
N_OOS = 400

# Number of times to perturb initial guess and run optimization for each
N_INITIAL_GUESS_PERTURBATIONS = 14
SIGMA_INITIAL_GUESS = 1e-2 # Standard deviation for the initial guess perturbation
L_INITIAL_GUESS = 0.2 # Length scale for the initial guess perturbation

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1000

# Seed for initial guess perturbation
seed_initial_guess = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
# filename = TEST_DIR / 'input.NCSX_c09r00_halfTeslaTF'
filename = TEST_DIR / 'input.LandremanPaul2021_QA'
# bn_file = TEST_DIR / 'input.NCSX_c09r00_halfTeslaTF_Bn'

# Directory for output

OUT_DIR = Path("output_stage_two_optimization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the boundary magnetic surface:
nphi = 64
ntheta = 16
s = SurfaceRZFourier.from_vmec_input(filename, 
                                     range="full torus", nphi=nphi, ntheta=ntheta)

qphi = 2 * nphi
qtheta = 64
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

#######################################################
# End of input parameters.
#######################################################
 
#define figure for kde plot
fig, ax = plt.subplots(figsize=(10,10))
#for loop over initial guess perturbation
sq_flux_values = []
mean_perturbed_sq_flux_values = []
gradients = []
for j in range(N_INITIAL_GUESS_PERTURBATIONS):
    LENGTH_WEIGHT = Weight(LENGTH_WEIGHT_INPUT)
    seed_initial_guess += 1
    
    # Create the initial coils:
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    curves_to_vtk(base_curves, OUT_DIR / f"base_curves_init")
    
    # np.random.seed(seed_initial_guess)
    # for c in base_curves:
    #     c.x += np.random.normal(scale=SIGMA_INITIAL_GUESS, size=c.x.shape)
        
    rg_initial_guess = Generator(PCG64DXSM(seed_initial_guess))
    sampler_initial_guess = GaussianSampler(base_curves[0].quadpoints, SIGMA_INITIAL_GUESS, L_INITIAL_GUESS, n_derivs=2)
    base_curves = [CurvePerturbed(c, PerturbationSample(sampler_initial_guess, randomgen=rg_initial_guess)) for c in base_curves]
    
    # show initial base coil after perturbation
    curves_to_vtk(base_curves, OUT_DIR / f"base_curves_init_perturbed_{j}")
    
    base_currents = [Current(1e5) for i in range(ncoils)]
    # Since the target field is zero, one possible solution is just to set all
    # currents to 0. To avoid the minimizer finding that solution, we fix one
    # of the currents:
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)

    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR / f"curves_init_{j}") #need to fix this, init and init_perturbed are the same

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

    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    # + CS_WEIGHT * Jcsdist \
    JF = Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        + CC_WEIGHT * Jccdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
        + ARCLENGTH_WEIGHT * sum(Jals) \
        
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
        outstr += f"\n-----On {(j+1)}/{N_INITIAL_GUESS_PERTURBATIONS} Initial Guess Perturbations"
        print(outstr)
        return J, grad

    # Perturb initial guess
    # seed_initial_guess += 1
    # np.random.seed(seed_initial_guess)
    # base_x = JF.x.copy()
    # x0 = base_x + np.random.normal(scale=SIGMA_INITIAL_GUESS,size=base_x.shape)
    # JF.x = x0
    # #show initial coil after perturbation
    # curves_to_vtk(curves, OUT_DIR / f"curves_init_perturbed_{j}")

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
    
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    
    for c in base_currents:
        print(f"-----------------Base current: {c.x}")
        
    curves_to_vtk(curves, OUT_DIR / f"curves_opt_{j}")
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
    s_plot.to_vtk(OUT_DIR / f"surf_opt_{j}", extra_data=pointData)
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
    curves_to_vtk(base_curves, OUT_DIR / f"base_curves_opt_{j}")
    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    # bs.save(OUT_DIR / "biot_savart_opt.json")
    sq_flux_unperturbed = Jf.J()

    #Perturb coils
    seed = 0
    curves_pert = []
    squared_flux_data = []
    rg = Generator(PCG64DXSM(seed+1))
    sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=1)
    for i in range(N_OOS):
        # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
        base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
        coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
        # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
        coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
        curves_pert.append([c.curve for c in coils_pert])
        bs_pert = BiotSavart(coils_pert)
        bs_pert.set_points(s.gamma().reshape((-1, 3)))
        squared_flux_data.append(SquaredFlux(s, bs_pert).J())
    
    print(f"Flux Objective for exact coils coils      : {sq_flux_unperturbed:.3e}")
    print(f"Out-of-sample flux value                  : {np.mean(squared_flux_data):.3e}")
    print(f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}")
    
    sns.kdeplot(squared_flux_data,fill=False,ax=ax, label=f'Initial Guess {j+1}')
    np.save(OUT_DIR / f"deterministic_perturbed_sq_flux_data_{j}",squared_flux_data)
    sq_flux_values.append(sq_flux_unperturbed)
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

plt.savefig(OUT_DIR / "squared_flux_distribution.png")
plt.close()

# Save unperturbed and mean perturbed values
header_string = 'Jf.J(), <Perturbed Jf.J()>, ||∇J||'
combined_array = np.column_stack((np.array(sq_flux_values),np.array(mean_perturbed_sq_flux_values),np.array(gradients)))
np.savetxt(OUT_DIR / 'J_Values.txt', 
           combined_array, delimiter=',', header = header_string, comments='')

# "Unperturbed Value: {:.4e}\n"
# "Out Of Sample Mean Flux Value: {:.4e}".format(sq_flux_unperturbed, np.mean(squared_flux_data)))
end_time = time.time()
print(f"Took {end_time-start_time}s")
