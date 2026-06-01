# =============================================================================
# run_config.py
# Central configuration imported at the top of kSZ_Squared_21cm_QMaStyle.py
#
# TWO MODES — set RUN_MODE below:
#   "test"      → tiny box, 2 seeds, narrow z, fast (~10 min on cluster)
#   "production" → full science run, 20 seeds, full z range
# =============================================================================

import os

# ──────────────────────────────────────────────────────────────────────────────
# SET THIS BEFORE SUBMITTING
# ──────────────────────────────────────────────────────────────────────────────
RUN_MODE = os.environ.get("RUN_MODE", "test")   # override via PBS env var
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATE_TAG  = "1Jun2026_kSZ_sqr_21cm_sqr"

if RUN_MODE == "test":
    # ── tiny smoke-test ──────────────────────────────────────────────────────
    HII_DIM          = 32          # 32³ cells  (vs 128³ in production)
    BOX_LEN          = 200.0       # 200 Mpc    (vs 800 Mpc)
    N_THREADS        = 4           # threads per seed worker
    RANDOM_SEEDS     = [1, 2]      # 2 seeds only
    Z_MIN            = 5.0         # lightcone lower z
    Z_MAX            = 15.0        # lightcone upper z  (narrower range)
    Z_OBS            = 5.0         # kSZ integration endpoint
    DELTA_Z          = 2.0         # chunk width
    Z_CHUNK_CENTRES  = [7.0, 9.0]  # only 2 chunks
    K_PAR_MIN        = 0.01        # foreground filter

    CACHE_DIR = os.path.join(BASE_DIR, DATE_TAG + "_TEST", "cache")
    PLOT_DIR  = os.path.join(BASE_DIR, DATE_TAG + "_TEST", "plots")

    print("=" * 60)
    print("  RUN_MODE = test")
    print(f"  HII_DIM  = {HII_DIM}   BOX_LEN = {BOX_LEN} Mpc")
    print(f"  Seeds    = {RANDOM_SEEDS}")
    print(f"  z range  = [{Z_MIN}, {Z_MAX}]")
    print(f"  Chunks   = {Z_CHUNK_CENTRES}")
    print(f"  Cache    → {CACHE_DIR}")
    print(f"  Plots    → {PLOT_DIR}")
    print("=" * 60)

elif RUN_MODE == "production":
    # ── full science run ─────────────────────────────────────────────────────
    HII_DIM          = 128
    BOX_LEN          = 800.0
    N_THREADS        = 8           # 8 threads/seed × 4 workers = 32 cores
    RANDOM_SEEDS     = list(range(1, 21))   # seeds 1..20
    Z_MIN            = 0.001
    Z_MAX            = 20.0
    Z_OBS            = 5.0
    DELTA_Z          = 2.0
    Z_CHUNK_CENTRES  = [7.0, 8.0, 9.0, 10.0, 11.0]
    K_PAR_MIN        = 0.01

    CACHE_DIR = os.path.join(BASE_DIR, DATE_TAG, "cache")
    PLOT_DIR  = os.path.join(BASE_DIR, DATE_TAG, "plots")

    print("=" * 60)
    print("  RUN_MODE = production")
    print(f"  HII_DIM  = {HII_DIM}   BOX_LEN = {BOX_LEN} Mpc")
    print(f"  Seeds    = {RANDOM_SEEDS[0]}..{RANDOM_SEEDS[-1]}")
    print(f"  z range  = [{Z_MIN}, {Z_MAX}]")
    print(f"  Chunks   = {Z_CHUNK_CENTRES}")
    print(f"  Cache    → {CACHE_DIR}")
    print(f"  Plots    → {PLOT_DIR}")
    print("=" * 60)

else:
    raise ValueError(f"Unknown RUN_MODE='{RUN_MODE}'. Use 'test' or 'production'.")

# Make directories immediately on import so workers never race on mkdir
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)
