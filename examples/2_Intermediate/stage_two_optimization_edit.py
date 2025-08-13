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

Updated Documentation--here the SIMSOpt example has been updated to given different initial
guesses by running multiple iterations and perturbing coil fourier coeffiecients to check the
robustness of the plasma compared to the original standard initial guess stochastic and deterministic
optimizations.

V1 - Uses Florians method using a normal distribution to perturb Fouier Coeffs, gives wriggly coils, somehwat infeasable.
V2 - Implmenting, attempting to use a perturbed coils set controled by user inputs as inital guesses 
instead of a random norm distibution that improptionally affects higher FSMs making the coils unbuildable.
"""

import os
import time
from pathlib import Path
import numpy as np
from randomgen import PCG64
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries, Coil
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation, CurvePerturbed,
                         PerturbationSample, GaussianSampler)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(1e-7)

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 10

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 10

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

#Added from Stoch
ARCLENGTH_WEIGHT=1e-2

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1000

#added for init pert compare
num_of_pert_inits = 2 #WHEN THIS IS 1, THEN JF.X ONLY HAS 4 CURVES
N_OOS = 500
SIGMA = 1
SIGMA_OOS = 1e-2
L = 0.5
seed = 0

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output_stage_two_optimization_edit/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################


# Initialize the boundary magnetic surface:
nphi = 64
ntheta = 16
s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
# Build Initial curvers to initialize the sampler to use to make initial guesses
control_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=False, R0=R0, R1=R1, order=order)
curves_to_vtk(control_curves, OUT_DIR + "curves_control") # just to compare to the initalg guesses while testing

#Current fixed stay outside loops
base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()

rg = np.random.Generator(PCG64(seed, inc=0))
sampler = GaussianSampler(control_curves[0].quadpoints, SIGMA, L, n_derivs=1)

### Build perturbed initial guesses to use for curves to optimize ###
pert_init_guesses = []
pert_coil_init_guesses = []
base_curves_pert = []
for i in range(num_of_pert_inits):
    # Create initial basic coils
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    pert_init_guesses.append([c.curve for c in coils_pert])
    pert_coil_init_guesses.append(coils_pert)
    base_curves_pert.append(base_curves_perturbed)

for j in range(num_of_pert_inits):
    
    coils = pert_coil_init_guesses[j]
    # coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
        
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR + f"curves_init_{j}")
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
    
    # Define the individual terms objective function:
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    #Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]


    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    # CS_WEIGHT * Jcsdist taken out
    JF = Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        + CC_WEIGHT * Jccdist \
        + ARCLENGTH_WEIGHT * sum(Jals)\
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
        

    # We don't have a general interface in SIMSOPT for optimisation problems that
    # are not in least-squares form, so we write a little wrapper function that we
    # pass directly to scipy.optimize.minimize

    # , C-S-Sep={Jcsdist.shortest_distance():.2f} taken out 
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
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
    for i in range(ncoils):
        print(np.array_equal(base_curves[i].x, base_curves_pert[j][i].x))
    
    
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        J1, _ = f(dofs + eps*h)
        J2, _ = f(dofs - eps*h)
        print("err", (J1-J2)/(2*eps) - dJh)
        

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    #res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    #curves_to_vtk(curves, OUT_DIR + "curves_opt_short")
    #pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    #s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)

    ############ Pert Init#######
    #Implmnting the actual init pert
    #base_x = JF.x.copy()

    seed += 1 
    np.random.seed(seed)

    # 1. Generate perturbed initial guess
    #x0 = base_x + np.random.normal(scale=1e-2, size=base_x.shape)

    # We now use the result from the optimization as the initial guess for a
    # subsequent optimization with reduced penalty for the coil length. This will
    # result in slightly longer coils but smaller `B·n` on the surface.
    #dofs = res.x
    #LENGTH_WEIGHT *= 0.1

    print(JF.dof_names)
    print(JF.x)

    exit()
        
    # After creating JF, add this debugging code:
    print("=== COMPARING FOURIER COEFFICIENTS ===")
    print(f"JF.x shape: {JF.x.shape}")
    print(f"First 3 elements (currents): {JF.x[:3]}")

    # Extract Fourier coefficients from JF.x (skip the first 3 current elements)
    fourier_coeffs_from_jf = JF.x[3:]
    print(f"Fourier coefficients from JF.x shape: {fourier_coeffs_from_jf.shape}")

    # Calculate DOFs per curve
    dofs_per_curve = 3 * (2 * order + 1)  # 33 DOFs per curve
    
    base_curves = base_curves_pert[j]
    print("\n=== FOURIER COEFFICIENTS COMPARISON ===")
    for i in range(ncoils):
        print(f"\n--- Base Curve {i} ---")
        
        # Get Fourier coefficients from JF.x for this curve
        start_idx = i * dofs_per_curve + 4*dofs_per_curve
        end_idx = start_idx + dofs_per_curve
        jf_curve_coeffs = fourier_coeffs_from_jf[start_idx:end_idx]
        
        # Get Fourier coefficients from the actual base_curves
        actual_curve_coeffs = base_curves[i].x
        
        print(f"JF.x Fourier coefficients for curve {i}:")
        print(f"  Shape: {jf_curve_coeffs.shape}")
        print(f"  First 10 values: {jf_curve_coeffs[:10]}")
        
        print(f"Actual base_curves[{i}] Fourier coefficients:")
        print(f"  Shape: {actual_curve_coeffs.shape}")
        print(f"  First 10 values: {actual_curve_coeffs[:10]}")
        
        # Compare the Fourier coefficients
        if np.array_equal(jf_curve_coeffs, actual_curve_coeffs):
            print(f"  ✓ MATCH: JF.x and base_curves[{i}] have identical Fourier coefficients")
        else:
            print(f"  ✗ MISMATCH: JF.x and base_curves[{i}] have different Fourier coefficients")
            diff = np.abs(jf_curve_coeffs - actual_curve_coeffs)
            print(f"  Max difference: {np.max(diff):.2e}")
            print(f"  Mean difference: {np.mean(diff):.2e}")
           
    exit()
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)

    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_long", extra_data=pointData)

    # 4. Save the optimized base coil curves to VTK
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_long_{j}")

    # 5. Recalculate OOS objective (taken for stoch)
    curves_pert = []
    sampler = GaussianSampler(curves[0].quadpoints, SIGMA_OOS, L, n_derivs=1)
    rg = np.random.Generator(PCG64(seed+j, inc=0))
    val = 0
    oos_flux = []

    for i in range(N_OOS):
        # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
        base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
        coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
        # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
        coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
        curves_pert.append([c.curve for c in coils_pert])
        bs_pert = BiotSavart(coils_pert)
        jn_oos = SquaredFlux(s, bs_pert).J() #added
        val += jn_oos 
        oos_flux.append(jn_oos) #added
    val /= N_OOS
    
    print(f"Flux Objective for exact coils coils      : {Jf.J():.3e}")
    print(f"Out-of-sample flux value                  : {val:.3e}")
    print(f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}")
    # 6. Save result
    np.save(f"{OUT_DIR}/det_flux_value_run_{j}.npy", oos_flux)


    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    #bs.save(OUT_DIR + "biot_savart_opt.json")