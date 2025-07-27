import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()


#-------------------------------------------
# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 2
    downsample = 100  # downsample the FAMUS grid of magnets by this factor
else:
    nphi = 64  # >= 64 for high-resolution runs
    downsample = 2

m_deterministic_file = 'output_permanent_magnet_GPMO_MUSE/m.npy'
m_stochastic_file = 'output_permanent_magnet_GPMO_MUSE_stochastic/m.npy'

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
out_dir = Path("output_permanent_magnet_MUSE_stoch_and_det")
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

# Load a downsampled version of the magnet grid from a FAMUS file
mag_data = FocusData(famus_filename, downsample=downsample)

# Set the allowable orientations of the magnets to be face-aligned
# i.e. the local x, y, or z directions
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

# Optionally add additional types of allowed orientations
PM4Stell_orientations = False
if PM4Stell_orientations:
    pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
    ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
    pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
    pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
    pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fe_ftri))
    pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
    ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
    pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
    pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
    pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fc_ftri))

ophi = np.arctan2(mag_data.oy, mag_data.ox)
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z
print('pol_vectors_shape = ', pol_vectors.shape)

# pol_vectors is only used for the greedy algorithms with cartesian coordinate_flag
# which is the default, so no need to specify it here.
kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

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
