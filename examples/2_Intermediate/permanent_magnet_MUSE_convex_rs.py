#!/usr/bin/env 
r"""
This example script uses the GPMO
greedy algorithm for solving permanent 
magnet optimization on the MUSE grid. This 
algorithm is described in the following paper:
    A. A. Kaptanoglu, R. Conlin, and M. Landreman, 
    Greedy permanent magnet optimization, 
    Nuclear Fusion 63, 036016 (2023)

The script should be run as:
    mpirun -n 1 python permanent_magnet_MUSE.py
on a cluster machine but 
    python permanent_magnet_MUSE.py
is sufficient on other machines. Note that this code does not use MPI, but is 
parallelized via OpenMP and XSIMD, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

For high-resolution and more realistic designs, please see the script files at
https://github.com/akaptano/simsopt_permanent_magnet_advanced_scripts.git
"""

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import relax_and_split
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 2
    nIter_max = 100
    nBacktracking = 50
    max_nMagnets = 20
    downsample = 100  # downsample the FAMUS grid of magnets by this factor
else:
    nphi = 64  # >= 64 for high-resolution runs
    nIter_max = 30000 #increasing doesnt affect fB for RS, but does affect fB_s
    nBacktracking = 200
    max_nMagnets = 30000
    downsample = 8

#Poincare plot parameters
tol = 1e-10
nfieldlines = 30 #30
tmax_fl = 20000 #10k
degree = 4
n = 40 #20 1.9, 3.2e-5

#noise parameters
mean = 0.0
sigma_factor = 1
samples_after_opt = 1e3

#algorithm parameters
max_iter = 250 # Number of iterations to take in a convex step
max_iter_RS = 1  # Number of iterations to take in a relax-and-split step
reg_l0 = 0.0  # L0 regularization parameter
reg_l1 = 0.0  # L1 regularization parameter
relax_and_split_iteration = 1 # Number of relax-and-split iterations to perform, 

#10k magnets, ds=2,rs_iter = 1
#200 for 3 is 3.9e-8, 2 is 5e-8
#2000 iter took 1600s/26min wit 2.2e-8
#1000 iter took 1200s/20 with 3.1e-8
#500 iter too 683s/11.4min with 4.3e-8
#400 iter 5e-8
#300 oter 6/5e-8
#200 iter 1.03e-7
#100 iter 2.5e-7

ntheta = nphi  # same as above
dr = 0.01  # Radial extent in meters of the cylindrical permanent magnet bricks
input_name = 'input.muse'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / 'zot80.focus'
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory -- warning, saved data can get big!
# On NERSC, recommended to change this directory to point to SCRATCH!
out_dir = Path("output_permanent_magnet_RS_MUSE")
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# Set up correct Bnormal from TF coils
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# remove pol_vector bc using relax and split
kwargs_geo = {"downsample": downsample, "dr": dr,"coordinate_flag": 'cylindrical',}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs_geo)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs()
kwargs['max_iter'] = max_iter  # Number of iterations to take in a convex step
kwargs['max_iter_RS'] = max_iter_RS  # Number of iterations to take in a relax-and-split step
kwargs['reg_l0'] = reg_l0
kwargs['reg_l1'] = reg_l1

# Optimize the permanent magnets. This actually solves
# 2 full relax-and-split problems, and uses the result of each
# problem to initialize the next, increasing L0 threshold each time,
# until thresholding over all magnets with strengths < 50% the max.

m0 = np.zeros(pm_opt.ndipoles*3) #initial guess, random or zeros
total_m_history = []
total_mproxy_history = []
total_RS_history = []
for i in range(relax_and_split_iteration):
    print('Relax-and-split iteration ', i)
    RS_history, m_history, m_proxy_history = relax_and_split(pm_opt, m0=m0, **kwargs)
    total_m_history.append(m_history)
    total_mproxy_history.append(m_proxy_history)
    total_RS_history.append(RS_history)
    m0 = pm_opt.m
    
total_RS_history = np.ravel(np.array(total_RS_history))

print('Done optimizing the permanent magnet object')

#plot fB_s = 0.5 |A(m+e_s)-b|^2, perturbing after optimization to check for robustness
sigma = pm_opt.m_maxima * sigma_factor 
fB_s_data = []

for i in range(int(samples_after_opt)):
    e_s = np.random.normal(loc=mean,scale=sigma[:,None],size=(pm_opt.ndipoles,3))
    e_s = e_s.reshape(pm_opt.ndipoles*3)
    fB_s_data.append(0.5 * np.sum((pm_opt.A_obj@(pm_opt.m+e_s)-pm_opt.b_obj)**2))
    if i % int(samples_after_opt/10) == 0:
        print("Sample", i, "mean[fB_s] = ", np.mean(fB_s_data))

#plot fb_s data and label fB and E(fB_s)
total_fB = 0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2)
plt.figure(figsize=(10,10))
plt.hist(fB_s_data, bins=50)
plt.xlabel('fB_s')
plt.ylabel('Count')
plt.title(
    "Histogram of fB (Relax and Split Convex)\n"
    "$fB = 0.5 |Am-b|^2 = {:.4e}$\n"
    "$\\mathbb{{E}}[fB_s] = {:.4e}$".format(total_fB, np.mean(fB_s_data))
)
plt.tight_layout()
plt.savefig(out_dir / "fB_s_deterministic_histogram.png")
plt.close()
#plot m/m_max
plt.figure()
m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
m_mag = np.sqrt((np.sum(m ** 2, axis=1)))
plt.figure()
plt.hist(m_mag/pm_opt.m_maxima,bins=np.linspace(0.7,1.3,500))
plt.xlabel('m/m_max')
plt.ylabel('Count')
plt.title('Histogram of m/m_max')
plt.savefig(out_dir/ "m_over_m_max_histogram.png")
plt.close()

# Try to make a mp4 movie of the optimization progress
try:
    make_optimization_plots(total_RS_history, total_m_history, total_mproxy_history, pm_opt, out_dir)
except ValueError:
    print(
        'Attempted to make a mp4 of optimization progress but ValueError was raised. '
        'This is probably an indication that a mp4 python writer was not available for use.'
    )
    
# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Plot the solutions from SIMSOPT

b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / "Dipole_Fields")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(b_dipole, s_plot, out_dir, "only_m_optimized")

pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "m_optimized", extra_data=pointData)

# Print optimized f_B and other metrics--------------------------------------------------------------------------------------
ratio = m_mag / pm_opt.m_maxima
print("m/m_maxima statistics:")
print("Min:", np.min(ratio))
print("Max:", np.max(ratio))
print("Median:", np.median(ratio))
print("Mean:", np.mean(ratio))
print("Std:", np.std(ratio))

print("Total fB (Deterministic)= ",
      total_fB)
print("Expected Value of fB_s = ", np.mean(fB_s_data))

f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)

total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume for m = ', total_volume)

dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
num_nonzero_sparse = np.count_nonzero(np.sum(dipoles ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("% of sparse dipoles that are nonzero = ", num_nonzero_sparse)

# write solution to FAMUS-type file
pm_opt.write_to_famus(out_dir)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    mpi = MpiPartition(ngroups=1)

    # Make the QFM surface
    t1 = time.time()
    Bfield = bs + b_dipole
    Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    Bfield_proxy.set_points(s_plot.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s_plot, Bfield)
    qfm_surf = qfm_surf.surface

    # repeat QFM calculation for the proxy solution
    Bfield_proxy = bs + b_dipole_proxy
    qfm_surf_proxy = make_qfm(s, Bfield_proxy)
    qfm_surf_proxy = qfm_surf_proxy.surface
    qfm_surf_proxy.plot()
    qfm_surf_proxy = qfm_surf
    t2 = time.time()
    print("Making the two QFM surfaces took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()

    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

    ### Always use the QH VMEC file for the proxy solution and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QH_reactorScale_lowres"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf_proxy
    equil.run()


#-------------------------------------

t_end = time.time()
print('Total time = ', t_end - t_start)
#plt.show()
