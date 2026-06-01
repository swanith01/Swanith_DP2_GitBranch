# lightcone_worker_helper.py
# =============================================================================
# Helper module for lightcone operations
# Optional: Import this in your notebook or use inline (Cell 2 is already inline)
# =============================================================================

import os
import glob
import time as _time
import py21cmfast as _p21c


def run_or_load_lightcone_tvir(seed, tvir, cache_base_dir, z_min, z_max, 
                                user_params, hii_eff_factor_fixed):
    """
    Run or load one (seed, ION_Tvir_MIN) lightcone combination.
    
    This is a standalone version suitable for use in external scripts.
    Cell 2 uses an inline version for better error handling.
    
    Parameters
    ----------
    seed : int
        Random seed for simulation
    tvir : float
        ION_Tvir_MIN parameter value
    cache_base_dir : str
        Base directory for caching HDF5 files
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    user_params : py21cmfast.UserParams
        User parameters (HII_DIM, BOX_LEN, etc.)
    hii_eff_factor_fixed : float
        Fixed HII_EFF_FACTOR value
    
    Returns
    -------
    tuple : (seed, tvir, lightcone, status, sim_time)
        - seed : int
        - tvir : float
        - lightcone : py21cmfast.LightCone or None
        - status : str ("cached", "computed", or "failed: <msg>")
        - sim_time : float (seconds)
    """
    
    # Create unique subdirectory for this (seed, tvir) combination
    cache_subdir = os.path.join(
        cache_base_dir,
        f"seed_{seed:03d}_Tvir{tvir:.4f}"
    )
    os.makedirs(cache_subdir, exist_ok=True)

    # Build astrophysical parameters
    astro_params = _p21c.AstroParams(
        HII_EFF_FACTOR=hii_eff_factor_fixed,
        ION_Tvir_MIN=tvir
    )

    # =========================================================================
    # CACHE CHECK
    # =========================================================================
    cached_files = sorted(glob.glob(os.path.join(cache_subdir, "LightCone_*.h5")))
    valid_cached = [(f, os.path.getsize(f) / 1e6)
                    for f in cached_files if os.path.getsize(f) / 1e6 > 1.0]

    if valid_cached:
        cache_file, size_mb = valid_cached[0]
        try:
            lc = _p21c.run_lightcone(
                redshift=z_min,
                max_redshift=z_max,
                lightcone_quantities=('brightness_temp', 'density', 'xH_box', 'velocity'),
                user_params=user_params,
                astro_params=astro_params,
                random_seed=seed,
                direc=cache_subdir,
                write=False,
            )
            return (seed, tvir, lc, "cached", 0.0)
        except Exception:
            pass  # Cache invalid, fall through to recompute

    # =========================================================================
    # RUN NEW SIMULATION
    # =========================================================================
    sim_start = _time.time()
    try:
        lc = _p21c.run_lightcone(
            redshift=z_min,
            max_redshift=z_max,
            lightcone_quantities=('brightness_temp', 'density', 'xH_box', 'velocity'),
            user_params=user_params,
            astro_params=astro_params,
            random_seed=seed,
            direc=cache_subdir,
            write=True
        )
        
        # Ensure persistence
        try:
            lc.save(direc=cache_subdir)
        except Exception:
            pass

        sim_time = _time.time() - sim_start
        return (seed, tvir, lc, "computed", sim_time)

    except Exception as e:
        sim_time = _time.time() - sim_start
        return (seed, tvir, None, f"failed: {str(e)}", sim_time)


def load_lightcone_from_cache(seed, tvir, cache_base_dir):
    """
    Load a previously computed lightcone from cache.
    
    Parameters
    ----------
    seed : int
        Random seed
    tvir : float
        ION_Tvir_MIN value
    cache_base_dir : str
        Base cache directory
    
    Returns
    -------
    py21cmfast.LightCone or None
        Loaded lightcone, or None if not found/failed to load
    """
    cache_subdir = os.path.join(
        cache_base_dir,
        f"seed_{seed:03d}_Tvir{tvir:.4f}"
    )
    
    cached_files = sorted(glob.glob(os.path.join(cache_subdir, "LightCone_*.h5")))
    if not cached_files:
        return None
    
    try:
        lc = _p21c.LightCone.read(cached_files[0])
        return lc
    except Exception:
        return None
