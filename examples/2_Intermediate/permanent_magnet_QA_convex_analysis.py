import time
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import relax_and_split_stochastic
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *


#-------------------------------------------
t_start = time.time()
# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    dr = 0.05  # cylindrical bricks with radial extent 5 cm
else:
    nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = 64
    dr = 0.02  # cylindrical bricks with radial extent 2 cm

algo = 'GPMO'

m_deterministic_file = 'output_permanent_magnet_GPMO_QA/m.npy'
m_stochastic_file = 'output_permanent_magnet_GPMO_QA_stochastic/m.npy'

coff = 0.1  # PM grid starts offset ~ 10 cm from the plasma surface
poff = 0.05  # PM grid end offset ~ 15 cm from the plasma surface
input_name = 'input.LandremanPaul2021_QA_lowres'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_projected_normal(poff)
s_outer.extend_via_projected_normal(poff + coff)

# Make the output directory
out_dir = Path("output_permanent_magnet_QA_stoch_and_det")
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils
bs = coil_optimization(s, bs, base_curves, curves, out_dir)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# check after-optimization average on-axis magnetic field strength
calculate_modB_on_major_radius(bs, s)

# Set up correct Bnormal from TF coils
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Finally, initialize the permanent magnet class
kwargs_geo = {"dr": dr, "coordinate_flag": "cylindrical"}
pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
    s, Bnormal, s_inner, s_outer, **kwargs_geo
)

# Retrieve optimized magnet dipoles
m_deterministic = np.load(m_deterministic_file)
m_stochastic = np.load(m_stochastic_file)

m_total = m_stochastic - m_deterministic
pm_opt.m = m_total

# Plot dipole field, bn from coils, bn from dipoles, and bn from coils and dipoles
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima
)
#plot dipole field
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(out_dir / "Dipole_Fields")
#plot bn from dipoles
make_Bnormal_plots(b_dipole, s_plot, out_dir, "only_m_optimized")
#plot bn from coils
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")
#plot bn from coils and dipoles
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(out_dir / "m_optimized", extra_data=pointData)

t_end = time.time()
print(f"Time taken: {t_end - t_start:.2f} seconds")
