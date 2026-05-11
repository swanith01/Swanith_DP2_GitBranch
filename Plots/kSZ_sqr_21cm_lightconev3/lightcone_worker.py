# =============================================================================
# lightcone_worker.py
# Worker function for concurrent lightcone simulations
# Runs in spawned subprocesses — fully self-contained with all imports
# =============================================================================

import os
import glob
import time as _time
import py21cmfast as _p21c


def run_or_load_seed(seed, seed_cache_dir, z_min, z_max,
                     hii_dim, box_len, n_threads):
    """
    Run or load one lightcone simulation for a given random seed.
    
    Checks for cached HDF5 files first. If valid cache exists, loads from disk.
    Otherwise, runs new simulation and saves it.
    
    Parameters
    ----------
    seed : int
        Random seed for the simulation
    seed_cache_dir : str
        Directory to store/retrieve cached HDF5 files
    z_min : float
        Minimum redshift (lower bound of lightcone)
    z_max : float
        Maximum redshift (upper bound of lightcone)
    hii_dim : int
        HII_DIM parameter (resolution of ionization field)
    box_len : float
        BOX_LEN parameter (comoving box size in Mpc)
    n_threads : int
        Number of threads for this worker
    
    Returns
    -------
    tuple : (seed, cache_file_path, sim_time, status)
        - seed : int — the input seed
        - cache_file_path : str or None — full path to cached HDF5 file
        - sim_time : float — wall time in seconds (0 if cached)
        - status : str — "cached", "computed", or "failed: <error_msg>"
    """
    
    os.makedirs(seed_cache_dir, exist_ok=True)

    # =========================================================================
    # Build parameters inside subprocess (CFFI objects don't survive spawn)
    # =========================================================================
    up = _p21c.UserParams(
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        USE_INTERPOLATION_TABLES=True,
        N_THREADS=n_threads,
    )
    ap = _p21c.AstroParams()

    # =========================================================================
    # CACHE CHECK (native py21cmfast HDF5)
    # =========================================================================
    cached = sorted(glob.glob(os.path.join(seed_cache_dir, "LightCone_*.h5")))
    valid_cached = [(f, os.path.getsize(f) / 1e6)
                    for f in cached if os.path.getsize(f) / 1e6 > 1.0]

    if valid_cached:
        cache_file, size_mb = valid_cached[0]
        # Validate it's loadable by re-running with write=False (uses cache)
        try:
            _ = _p21c.run_lightcone(
                redshift=z_min,
                max_redshift=z_max,
                lightcone_quantities=('brightness_temp', 'density',
                                      'xH_box', 'velocity'),
                user_params=up,
                astro_params=ap,
                random_seed=seed,
                direc=seed_cache_dir,
                write=False,
            )
            return (seed, cache_file, 0.0, "cached")
        except Exception:
            # Cache file present but invalid — fall through to recompute
            pass

    # =========================================================================
    # RUN NEW SIMULATION
    # =========================================================================
    sim_start = _time.time()
    try:
        lc = _p21c.run_lightcone(
            redshift=z_min,
            max_redshift=z_max,
            lightcone_quantities=('brightness_temp', 'density',
                                  'xH_box', 'velocity'),
            user_params=up,
            astro_params=ap,
            random_seed=seed,
            direc=seed_cache_dir,
            write=True,
        )
        
        # Make sure it's persisted (write=True usually does this, but be explicit)
        try:
            lc.save(direc=seed_cache_dir)
        except Exception:
            pass

        # Find the saved file
        cached = sorted(glob.glob(os.path.join(seed_cache_dir, "LightCone_*.h5")))
        cache_file = cached[0] if cached else None
        
        sim_time = _time.time() - sim_start
        return (seed, cache_file, sim_time, "computed")

    except Exception as e:
        sim_time = _time.time() - sim_start
        return (seed, None, sim_time, f"failed: {str(e)}")
