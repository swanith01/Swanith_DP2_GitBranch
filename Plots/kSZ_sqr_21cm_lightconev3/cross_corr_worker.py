# =============================================================================
# cross_corr_worker.py
# Parallel worker for computing kSZ²-21cm cross-correlations per seed
# FIX: Load LightCone from HDF5 inside worker (don't pickle CFFI objects)
# =============================================================================

import numpy as np
import os
import time


def compute_cross_corr_for_seed(args):
    """
    Compute cross-correlation power spectra for ONE seed across all node redshifts.
    
    Parameters
    ----------
    args : tuple
        (seed, cache_file, kSZ_map, z_obs, main_cache_dir,
         npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
         dk, kgrid, k_bins, k_centers)
    
    Returns
    -------
    tuple : (seed, cross_corr_results_dict or None, status_msg)
    """
    
    import py21cmfast as p21c
    
    (seed, cache_file, kSZ_map, z_obs, main_cache_dir,
     npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
     dk, kgrid, k_bins, k_centers) = args
    
    # Load LightCone from HDF5 file INSIDE the worker (don't pickle CFFI objects)
    try:
        lc = p21c.LightCone.read(cache_file)
    except Exception as e:
        return (seed, None, f"failed to load lightcone: {e}")
    
    seed_cache_dir = os.path.join(main_cache_dir, f"seed_{seed}")
    os.makedirs(seed_cache_dir, exist_ok=True)
    cc_cache = os.path.join(seed_cache_dir, f"cross_corr_seed{seed}.npy")
    
    # Check if already cached
    if os.path.exists(cc_cache):
        try:
            result = np.load(cc_cache, allow_pickle=True).item()
            return (seed, result, f"cached ({len(result)} redshifts)")
        except Exception as e:
            return (seed, None, f"cache load failed: {e}")
    
    # Validate kSZ map
    if kSZ_map is None:
        return (seed, None, "no kSZ map available")
    
    try:
        # Compute kSZ² power spectrum (same for all redshifts)
        kSZ2_map = kSZ_map**2
        kSZ2_map_centered = kSZ2_map - np.mean(kSZ2_map)
        fft_kSZ2_shifted = np.fft.fftshift(np.fft.fft2(kSZ2_map_centered))
        auto_kSZ2_ps2d = (np.abs(fft_kSZ2_shifted)**2 * pix_area / npix_side**2)
        
        # Loop over node redshifts
        node_redshifts = np.asarray(lc.node_redshifts[::-1])
        cross_corr_results = {}
        
        lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
        
        for i, z_21cm in enumerate(node_redshifts):
            idx_closest = int(np.argmin(np.abs(lc_redshifts - z_21cm)))
            z_actual = float(lc_redshifts[idx_closest])
            
            # Get 21cm slice
            T21_slice = np.asarray(lc.brightness_temp[:, :, idx_closest])
            T21_slice_centered = T21_slice - np.mean(T21_slice)
            fft_T21_shifted = np.fft.fftshift(np.fft.fft2(T21_slice_centered))
            
            # Compute cross and auto power spectra
            cross_ps2d = (np.real(np.conj(fft_kSZ2_shifted) * fft_T21_shifted)
                         * pix_area / npix_side**2)
            auto_T21_ps2d = (np.abs(fft_T21_shifted)**2 * pix_area / npix_side**2)
            
            # Bin in k-space
            C_cross_1d = np.zeros(len(k_centers))
            C_cross_1d_err_sample = np.zeros(len(k_centers))
            C_cross_1d_err_cosmic = np.zeros(len(k_centers))
            C_cross_1d_err_total = np.zeros(len(k_centers))
            P_kSZ2_1d = np.zeros(len(k_centers))
            P_T21_1d = np.zeros(len(k_centers))
            n_modes = np.zeros(len(k_centers))
            
            for j in range(len(k_centers)):
                mask = (kgrid >= k_bins[j]) & (kgrid < k_bins[j+1])
                n_pix = np.sum(mask)
                
                if n_pix > 0:
                    cross_values = cross_ps2d[mask]
                    C_cross_1d[j] = np.mean(cross_values)
                    C_cross_1d_err_sample[j] = np.std(cross_values) / np.sqrt(n_pix)
                    P_kSZ2_1d[j] = np.mean(auto_kSZ2_ps2d[mask])
                    P_T21_1d[j] = np.mean(auto_T21_ps2d[mask])
                    
                    k_volume = (box_size_Mpc / (2*np.pi))**3
                    n_modes[j] = (4 * np.pi * k_centers[j]**2 * k_volume
                                 * (k_bins[j+1] - k_bins[j]))
                    
                    if n_modes[j] > 0:
                        C_cross_1d_err_cosmic[j] = (
                            np.sqrt(P_kSZ2_1d[j] * P_T21_1d[j]
                                   + C_cross_1d[j]**2)
                            / np.sqrt(n_modes[j])
                        )
                    else:
                        C_cross_1d_err_cosmic[j] = np.nan
                    
                    C_cross_1d_err_total[j] = np.sqrt(
                        C_cross_1d_err_sample[j]**2 + C_cross_1d_err_cosmic[j]**2
                    )
                else:
                    C_cross_1d[j] = np.nan
                    C_cross_1d_err_sample[j] = np.nan
                    C_cross_1d_err_cosmic[j] = np.nan
                    C_cross_1d_err_total[j] = np.nan
                    P_kSZ2_1d[j] = np.nan
                    P_T21_1d[j] = np.nan
            
            cross_corr_results[z_21cm] = {
                'k_centers': k_centers,
                'C_cross_1d': C_cross_1d,
                'C_cross_1d_err_sample': C_cross_1d_err_sample,
                'C_cross_1d_err_cosmic': C_cross_1d_err_cosmic,
                'C_cross_1d_err_total': C_cross_1d_err_total,
                'n_modes': n_modes,
                'P_kSZ2_1d': P_kSZ2_1d,
                'P_T21_1d': P_T21_1d,
                'z_actual': z_actual,
                'idx_closest': idx_closest,
                'kSZ2_rms': float(np.sqrt(np.mean(kSZ2_map**2))),
                'T21_rms': float(np.sqrt(np.mean(T21_slice**2))),
                'T21_mean': float(np.mean(T21_slice)),
            }
        
        # Save cache
        np.save(cc_cache, cross_corr_results)
        
        return (seed, cross_corr_results, f"computed ({len(cross_corr_results)} redshifts)")
    
    except Exception as e:
        return (seed, None, f"failed: {str(e)}")
