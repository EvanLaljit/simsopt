"""
This module contains the a number of useful functions for using 
the coil functionality in the SIMSOPT code.
"""
__all__ = ['curve_fourier_fit', 'run_mode'
           ]

from simsopt.geo import (create_equally_spaced_curves,)
import numpy as np

#some values are different between stochastic and deterministic
#namely sigma, l vs sigma_oos, l_oos
#and loop_label for sigma_l_scan
def run_mode(run_mode, slurm_array_int):
    """Setup parameters based on run mode and SLURM array ID"""
    if run_mode == 'pert_init':
        return {
            'SIGMA_INITIAL_GUESS': 1e-2,
            'L_INITIAL_GUESS': 0.15,
            'fourier_fit': False,
            'loop_label': slurm_array_int
        }
    elif run_mode == 'sigma_l_scan':
        sigma_values = np.linspace(1e-3, 1e-2, 8)
        L_values = np.linspace(0.5, 1.0, 4)
        sigma_and_L = [(sigma, L) for sigma in sigma_values for L in L_values]
        SIGMA, L = sigma_and_L[slurm_array_int]
        return {
            'sigma_values': sigma_values,
            'L_values': L_values,
            'sigma_and_L': sigma_and_L,
            'SIGMA': SIGMA,
            'L': L,
            'loop_label': f"Sigma={SIGMA:.3f};L={L:.3f}"
        }
    elif run_mode == 'order_scan':
        order_values = [int(i) for i in range(5,85,10)]
        order = order_values[slurm_array_int]
        return {
            'order_values': order_values,
            'order': order,
            'loop_label': f"Order:{order}"
        }
    else:
        raise ValueError(f"Unknown run mode: {run_mode}")
    """
    ## Usage in your main code:
    params = setup_run_mode(RUN_MODE, slurm_array_int)
    globals().update(params)
    """
def curve_fourier_fit(base_curves_pert,s,order):
    
    ncoils = len(base_curves_pert)
    base_curves_fit = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=0.5, R1=1.0, order=order)

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
    # curves_to_vtk(base_curves_fit, OUT_DIR / f"base_curves_init_fit")
    
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

    # return fitted curves and the mean fit error
    return base_curves_fit, np.mean(fit_error)