# %%
# %%
# =============================================================================
# LIGHTCONE GENERATION - MULTIPLE HII_EFF_FACTOR VALUES
# py21cmfast v3.4
# Adapted for running with HII_EFF_FACTOR = [30.0, 50.0, 70.0]
# =============================================================================

# =============================================================================
# CELL 1: Imports and Setup
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import py21cmfast as p21c
from py21cmfast import plotting

import os
import glob
import time
from datetime import datetime

# PBS vs desktop backend
if os.environ.get('PBS_JOBID'):
    matplotlib.use('Agg')
    print("✓ Using Agg backend (PBS/server mode)")
else:
    matplotlib.use('Agg')   # change to TkAgg for interactive desktop display
    print("✓ Using Agg backend")

print(f"py21cmfast version: {p21c.__version__}")

# --- Skewed LOS parameters (defined early, used in CELL 5) ---
angle_deg = 10    # Rotation angle in degrees
Nlos      = None  # None → full Ndim² box face

# =============================================================================
# CELL 1a: Output Directory
# =============================================================================

plot_dir      = "/user1/swanith/HII_EFF_FACTOR_SCAN_6May2026/plots"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
else:
    print(f"Directory already exists: {plot_dir}")

print(f"All plots will be saved to: {os.path.abspath(plot_dir)}")

# =============================================================================
# CELL 1b: Global Plot Settings
# DO NOT override font sizes, grid, or tick settings in any downstream cell.
# All plots rely entirely on these settings + PDF_STYLE / PNG_STYLE contexts.
# =============================================================================

plt.rcParams.update({
    # Font
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 20,
    'axes.labelsize'     : 28,
    'axes.titlesize'     : 22,
    'xtick.labelsize'    : 22,
    'ytick.labelsize'    : 22,
    'legend.fontsize'    : 18,
    'figure.titlesize'   : 20,
    # Ticks
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.major.size'   : 6,
    'ytick.major.size'   : 6,
    'xtick.minor.size'   : 3,
    'ytick.minor.size'   : 3,
    'xtick.major.width'  : 1.0,
    'ytick.major.width'  : 1.0,
    'xtick.minor.width'  : 0.8,
    'ytick.minor.width'  : 0.8,
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    # Lines / axes
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.8,
    'lines.markersize'   : 5,
    # Grid — OFF everywhere, no exceptions
    'axes.grid'          : False,
    'grid.linewidth'     : 0.5,
    'grid.alpha'         : 0.3,
    # Figure / save
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
})

print("✓ Global plot settings applied (grid OFF, no downstream overrides needed)")

# =============================================================================
# PDF / PNG style contexts — used ONLY inside save_pdf_png
# =============================================================================

PDF_STYLE = {
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 28,
    'axes.labelsize'     : 28,
    'axes.titlesize'     : 32,
    'xtick.labelsize'    : 26,
    'ytick.labelsize'    : 26,
    'legend.fontsize'    : 22,
    'figure.titlesize'   : 28,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size'   : 6,
    'ytick.major.size'   : 6,
    'xtick.minor.size'   : 3,
    'ytick.minor.size'   : 3,
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.8,
    'axes.grid'          : False,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
}

PNG_STYLE = {
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 16,
    'axes.labelsize'     : 22,
    'axes.titlesize'     : 18,
    'xtick.labelsize'    : 20,
    'ytick.labelsize'    : 20,
    'legend.fontsize'    : 18,
    'figure.titlesize'   : 16,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.5,
    'axes.grid'          : False,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
}

print("✓ PDF and PNG style contexts defined")


def save_pdf_png(plot_func, plot_dir, plot_name, title=None):
    """
    Save a plot as both PDF and PNG.

    Parameters
    ----------
    plot_func : callable
        f(ax) — draws onto the provided Axes.
        Do NOT set font sizes or grid inside plot_func.
    plot_dir  : str
    plot_name : str  (no extension)
    title     : str or None  — PNG-only title (PDF has no title)
    """
    with mpl.rc_context(PDF_STYLE):
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        ax.set_title("")
        ax.grid(False)
        fig.savefig(f"{plot_dir}/{plot_name}.pdf")
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        ax.grid(False)
        if title is not None:
            ax.set_title(title, fontweight='bold')
        fig.savefig(f"{plot_dir}/{plot_name}.png")
        plt.close(fig)


print("✓ save_pdf_png defined (plot_func pattern, grid always OFF)")


# =============================================================================
# CELL 1c: Define Parameters
# =============================================================================

user_params = p21c.UserParams(
    HII_DIM=128,
    BOX_LEN=800.0,
    USE_INTERPOLATION_TABLES=True,
    N_THREADS=64
)

z_min = 5.0
z_max = 20.0

HII_EFF_FACTORS = np.linspace(25.0, 70.0, 20).tolist()

# --- Skewed LOS geometry ---
angle_rad = np.deg2rad(angle_deg)
sin_a     = float(np.sin(angle_rad))
cos_a     = float(np.cos(angle_rad))

print(f"\n=== PARAMETER SETUP ===")
print(f"HII_DIM        = {user_params.HII_DIM}")
print(f"BOX_LEN        = {user_params.BOX_LEN:.0f} Mpc")
print(f"z range        = [{z_min}, {z_max}]")
print(f"HII_EFF_FACTORS = {HII_EFF_FACTORS}")

print(f"\n=== SKEWED LOS PARAMETERS ===")
print(f"  Rotation angle : {angle_deg}°")
print(f"  sin(θ)         : {sin_a:.4f}")
print(f"  cos(θ)         : {cos_a:.4f}")
print(f"  Nlos mode      : FULL BOX FACE (Ndim² = {user_params.HII_DIM**2})")

print("\n=== DEFAULT COSMOLOGY ===")
print(p21c.CosmoParams())

print("\n=== DEFAULT ASTROPHYSICS ===")
print(p21c.AstroParams())

print("\n=== DEFAULT FLAGS ===")
print(p21c.FlagOptions())

# %%
# =============================================================================
# CELL 2: Run Lightcone Simulations - HII_EFF_FACTOR Scan
# WITH ROBUST CACHE CHECKING AND RETRIEVAL
# =============================================================================

import time
import os
import glob

print("\n" + "="*70)
print("RUNNING HII_EFF_FACTOR SCAN")
print("="*70)

# =============================================================================
# CACHE DIRECTORY
# =============================================================================
try:
    # Running as .py script via PBS
    main_cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache"
    )
except NameError:
    # Running in Jupyter notebook
    main_cache_dir = os.path.join(
        os.getcwd(), "HII_EFF_FACTOR_SCAN_6May2026", "cache"
    )

os.makedirs(main_cache_dir, exist_ok=True)
print(f"Cache directory: {main_cache_dir}", flush=True)

lightcones      = {}
scan_start_time = time.time()

for idx, hii_factor in enumerate(HII_EFF_FACTORS):
    sim_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"SIMULATION {idx+1}/{len(HII_EFF_FACTORS)}")
    print(f"HII_EFF_FACTOR = {hii_factor:.2f}")
    print(f"{'='*70}", flush=True)

    astro_params = p21c.AstroParams(HII_EFF_FACTOR=hii_factor)
    cache_subdir = os.path.join(main_cache_dir, f"EFF{hii_factor:.2f}")
    os.makedirs(cache_subdir, exist_ok=True)

    print(f"Redshift range: z = {z_min} → {z_max}")
    print(f"Box size: {user_params.BOX_LEN:.0f} Mpc")
    print(f"Resolution: {user_params.HII_DIM}³ cells", flush=True)

    # ========================================================================
    # CACHE CHECK
    # ========================================================================
    cached_files = glob.glob(
        os.path.join(cache_subdir, f"LightCone_*z{z_min:.2f}*.h5")
    )
    valid_cache = [(f, os.path.getsize(f) / 1e6)
                   for f in cached_files if os.path.getsize(f) / 1e6 > 1.0]

    if valid_cache:
        cache_file, size_mb = valid_cache[0]
        print(f"\n✓ Found valid cached lightcone ({size_mb:.1f} MB)")
        print(f"  {cache_file}", flush=True)
        try:
            lightcone = p21c.run_lightcone(
                redshift=z_min,
                max_redshift=z_max,
                lightcone_quantities=('brightness_temp', 'density',
                                      'xH_box', 'velocity'),
                user_params=user_params,
                astro_params=astro_params,
                random_seed=37,
                direc=cache_subdir,
                write=False
            )
            lightcones[hii_factor] = lightcone
            load_time = time.time() - sim_start_time
            print(f"  ✓ Loaded in {load_time:.1f} s  "
                  f"shape={lightcone.brightness_temp.shape}", flush=True)
            continue
        except Exception as e:
            print(f"  ✗ Load failed: {e} — recomputing...", flush=True)

    # ========================================================================
    # RUN NEW SIMULATION
    # ========================================================================
    print(f"\n⚙ Running new simulation...", flush=True)
    try:
        lightcone = p21c.run_lightcone(
            redshift=z_min,
            max_redshift=z_max,
            lightcone_quantities=('brightness_temp', 'density',
                                  'xH_box', 'velocity'),
            user_params=user_params,
            astro_params=astro_params,
            random_seed=37,
            direc=cache_subdir,
            write=True
        )
        lightcones[hii_factor] = lightcone

        # Save immediately
        try:
            lightcone.save(direc=cache_subdir)
            print(f"  ✓ Lightcone saved to {cache_subdir}", flush=True)
        except Exception as e:
            print(f"  ⚠ Could not save: {e}", flush=True)

        sim_time = time.time() - sim_start_time
        print(f"\n✓ Simulation complete! ({sim_time/60:.2f} min)")
        print(f"  Shape: {lightcone.brightness_temp.shape}")
        print(f"  z range: [{lightcone.lightcone_redshifts.min():.2f}, "
              f"{lightcone.lightcone_redshifts.max():.2f}]")

        z_nodes   = lightcone.node_redshifts[::-1]
        x_e_nodes = 1.0 - lightcone.global_xH[::-1]
        try:
            z_10 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.1))]
            z_50 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.5))]
            z_90 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.9))]
            print(f"  z(10%)={z_10:.2f}  z(50%)={z_50:.2f}  "
                  f"z(90%)={z_90:.2f}  Δz={z_10-z_90:.2f}")
        except:
            pass

        elapsed  = time.time() - scan_start_time
        avg_time = elapsed / (idx + 1)
        eta_min  = (len(HII_EFF_FACTORS) - idx - 1) * avg_time / 60
        print(f"  Progress: {idx+1}/{len(HII_EFF_FACTORS)} "
              f"({100*(idx+1)/len(HII_EFF_FACTORS):.0f}%)  "
              f"ETA: {eta_min:.1f} min", flush=True)

    except Exception as e:
        print(f"\n✗ Simulation FAILED: {e}", flush=True)
        lightcones[hii_factor] = None

total_time = time.time() - scan_start_time
successful = sum(lc is not None for lc in lightcones.values())
print(f"\n{'='*70}")
print(f"ALL {len(HII_EFF_FACTORS)} SIMULATIONS COMPLETE  —  {successful} successful")
print(f"Total time: {total_time/60:.2f} min ({total_time/3600:.2f} h)")
print(f"Cache: {main_cache_dir}")
print("="*70, flush=True)

# =============================================================================
# CELL 2b: Plot Lightcones (stacked, one panel per HII_EFF_FACTOR)
# =============================================================================

print("\n" + "="*70)
print("GENERATING LIGHTCONE PLOTS")
print("="*70)

cmap_reion = mpl.cm.plasma
norm_reion = mpl.colors.Normalize(
    vmin=min(HII_EFF_FACTORS), vmax=max(HII_EFF_FACTORS)
)

custom_cmaps = {
    'brightness_temp': 'EoR',
    'xH_box'         : 'viridis',
    'density'        : 'magma',
    'velocity'       : 'RdBu_r',
}

fields_to_plot = [
    ('brightness_temp', '21cm Brightness Temperature [mK]'),
    ('xH_box',          'Neutral Fraction $x_{\rm HI}$'),
    ('density',         'Overdensity $\delta$'),
    ('velocity',        'LOS Velocity [Mpc/s]'),
]

for field_name, field_title in fields_to_plot:
    print(f"\nPlotting {field_name}...")

    fig, axes = plt.subplots(len(HII_EFF_FACTORS), 1,
                             figsize=(12, 5*len(HII_EFF_FACTORS)),
                             constrained_layout=True)
    if len(HII_EFF_FACTORS) == 1:
        axes = [axes]

    for idx, (hii_factor, ax) in enumerate(zip(HII_EFF_FACTORS, axes)):
        if hii_factor not in lightcones or lightcones[hii_factor] is None:
            ax.text(0.5, 0.5, f'HII_EFF={hii_factor:.1f}\nNo data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, color='red')
            continue

        lightcone = lightcones[hii_factor]
        plotting.lightcone_sliceplot(lightcone, field_name, ax=ax, fig=fig)

        im = ax.images[0]
        im.set_cmap(custom_cmaps[field_name])

        color_label = cmap_reion(norm_reion(hii_factor))
        ax.text(0.02, 0.98, f'HII\_EFF\_FACTOR = {hii_factor:.1f}',
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color_label,
                        alpha=0.8, edgecolor='black', linewidth=1.5))

    plot_name = f"{field_name}_lightcone_multi_HII"
    fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
    fig.suptitle(f'{field_title} — HII\_EFF\_FACTOR Scan',
                fontsize=20, fontweight='bold', y=0.995)
    fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_name}")
    plt.close(fig)

print("\n✓ LIGHTCONE PLOTTING COMPLETE!")

# %%
# =============================================================================
# CELL 4: Reionization History + Optical Depth Analysis
# =============================================================================

print("\n" + "="*70)
print("REIONIZATION HISTORY + OPTICAL DEPTH ANALYSIS")
print("="*70)

# Colormap — rainbow over HII_EFF_FACTORS
cmap_eff = mpl.cm.plasma
norm_eff  = mpl.colors.Normalize(vmin=min(HII_EFF_FACTORS),
                                   vmax=max(HII_EFF_FACTORS))

# =============================================================================
# PLOT 4a: Ionization Fraction vs Redshift
# =============================================================================

def plot_xe(ax):
    for hii_factor in HII_EFF_FACTORS:
        if hii_factor not in lightcones or lightcones[hii_factor] is None:
            continue
        lc = lightcones[hii_factor]
        ax.plot(lc.node_redshifts[::-1], 1.0 - lc.global_xH[::-1],
                color=cmap_eff(norm_eff(hii_factor)),
                linewidth=2.5, marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Ionization Fraction $x_e$')
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    sm = mpl.cm.ScalarMappable(cmap=cmap_eff, norm=norm_eff)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'HII\_EFF\_FACTOR')

save_pdf_png(plot_xe, plot_dir,
             "reionization_history_xe_multi_HII",
             title='Reionization History: Ionization Fraction')
print("✓ Saved: reionization_history_xe_multi_HII")

# =============================================================================
# PLOT 4b: Neutral Fraction vs Redshift
# =============================================================================

def plot_xHI(ax):
    for hii_factor in HII_EFF_FACTORS:
        if hii_factor not in lightcones or lightcones[hii_factor] is None:
            continue
        lc = lightcones[hii_factor]
        ax.plot(lc.node_redshifts[::-1], lc.global_xH[::-1],
                color=cmap_eff(norm_eff(hii_factor)),
                linewidth=2.5, marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Neutral Fraction $x_{\rm HI}$')
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    sm = mpl.cm.ScalarMappable(cmap=cmap_eff, norm=norm_eff)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'HII\_EFF\_FACTOR')

save_pdf_png(plot_xHI, plot_dir,
             "reionization_history_xHI_multi_HII",
             title='Reionization History: Neutral Fraction')
print("✓ Saved: reionization_history_xHI_multi_HII")

# =============================================================================
# Reionization Milestones
# =============================================================================

print("\nReionization Milestones:")
print("-" * 70)

milestones_data = {}

for hii_factor in HII_EFF_FACTORS:
    if hii_factor not in lightcones or lightcones[hii_factor] is None:
        continue
    lc      = lightcones[hii_factor]
    z_nodes = lc.node_redshifts[::-1]
    x_e     = 1.0 - lc.global_xH[::-1]

    z_10 = z_nodes[np.argmin(np.abs(x_e - 0.1))]
    z_50 = z_nodes[np.argmin(np.abs(x_e - 0.5))]
    z_90 = z_nodes[np.argmin(np.abs(x_e - 0.9))]

    milestones_data[hii_factor] = dict(
        z_10=z_10, z_50=z_50, z_90=z_90, delta_z=z_10 - z_90
    )
    print(f"HII_EFF={hii_factor:.2f}: "
          f"z10={z_10:.2f}  z50={z_50:.2f}  z90={z_90:.2f}  "
          f"Δz={z_10-z_90:.2f}")

hii_vals     = sorted(milestones_data.keys())
z_10_vals    = [milestones_data[h]['z_10']    for h in hii_vals]
z_50_vals    = [milestones_data[h]['z_50']    for h in hii_vals]
z_90_vals    = [milestones_data[h]['z_90']    for h in hii_vals]
delta_z_vals = [milestones_data[h]['delta_z'] for h in hii_vals]

# =============================================================================
# PLOT 4c: Milestones vs HII_EFF_FACTOR
# =============================================================================

def plot_milestones(ax):
    ax.plot(hii_vals, z_10_vals, 'o-', linewidth=2.5, label='10% ionized')
    ax.plot(hii_vals, z_50_vals, 's-', linewidth=2.5, label='50% ionized')
    ax.plot(hii_vals, z_90_vals, '^-', linewidth=2.5, label='90% ionized')
    ax.set_xlabel(r'HII\_EFF\_FACTOR')
    ax.set_ylabel('Redshift')
    ax.legend(loc='best')

save_pdf_png(plot_milestones, plot_dir,
             "reionization_milestones_multi_HII",
             title='Reionization Milestones')
print("✓ Saved: reionization_milestones_multi_HII")

# =============================================================================
# PLOT 4d: Duration Δz vs HII_EFF_FACTOR
# =============================================================================

def plot_duration(ax):
    ax.plot(hii_vals, delta_z_vals, 'o-', linewidth=3)
    ax.set_xlabel(r'HII\_EFF\_FACTOR')
    ax.set_ylabel(r'Reionization Duration $\Delta z$ (10%→90%)')

save_pdf_png(plot_duration, plot_dir,
             "reionization_duration_multi_HII",
             title='Reionization Duration')
print("✓ Saved: reionization_duration_multi_HII")

# =============================================================================
# CELL 4f: Optical Depth Calculations
# =============================================================================

print("\n" + "="*70)
print("OPTICAL DEPTH CALCULATIONS")
print("="*70)

h              = 0.6766
Omega_b        = 0.04897468161869667
rho_crit_p_cm3 = 1.88e-29 * h**2 / (1.67e-24)
n_H0_cm3       = Omega_b * rho_crit_p_cm3
sigma_T_cm2    = 6.65e-25
cm_per_Mpc     = 3.086e24
n_e0_Mpc3      = n_H0_cm3 * cm_per_Mpc**3
sigma_T_Mpc2   = sigma_T_cm2 / cm_per_Mpc**2
prefactor      = n_e0_Mpc3 * sigma_T_Mpc2

print(f"  n_H0      = {n_H0_cm3:.6e} cm^-3")
print(f"  σ_T       = {sigma_T_cm2:.6e} cm^2")
print(f"  Prefactor = {prefactor:.6e} Mpc^-1")

tau_results = {}

for hii_factor in HII_EFF_FACTORS:
    if hii_factor not in lightcones or lightcones[hii_factor] is None:
        continue
    lc       = lightcones[hii_factor]
    red_axis = lc.lightcone_redshifts
    pos_axis = lc.lightcone_distances
    ind_z    = np.where(red_axis <= z_max)[0]
    red_axis = red_axis[ind_z]
    pos_axis = pos_axis[ind_z]

    z_nodes    = lc.node_redshifts[::-1]
    x_e_nodes  = 1.0 - lc.global_xH[::-1]
    x_e_interp = np.interp(red_axis, z_nodes, x_e_nodes)

    s    = pos_axis
    ds   = np.diff(s)
    z_mid   = 0.5 * (red_axis[:-1] + red_axis[1:])
    x_e_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])
    dtau    = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds
    tau     = np.cumsum(dtau)

    tau_results[hii_factor] = dict(
        red_axis=red_axis, pos_axis=pos_axis, s=s, ds=ds,
        z_mid=z_mid, x_e_interp=x_e_interp, x_e_mid=x_e_mid,
        dtau=dtau, tau=tau, tau_total=tau[-1], ind_z=ind_z
    )
    print(f"  HII_EFF={hii_factor:.2f}: τ_total={tau[-1]:.6f}  "
          f"<ds>={ds.mean():.3f} Mpc")

# =============================================================================
# PLOT 4e: Comoving Distance s vs z
# =============================================================================

def plot_s_vs_z(ax):
    for hii_factor in sorted(tau_results.keys()):
        r = tau_results[hii_factor]
        ax.plot(np.asarray(r['red_axis']), np.asarray(r['s']),
                color=cmap_eff(norm_eff(hii_factor)),
                linewidth=2.0, alpha=0.8)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Comoving Distance $s$ [Mpc]')
    ax.invert_xaxis()
    sm = mpl.cm.ScalarMappable(cmap=cmap_eff, norm=norm_eff)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'HII\_EFF\_FACTOR')

save_pdf_png(plot_s_vs_z, plot_dir, "s_vs_z_multi_HII",
             title='Comoving Distance vs Redshift')
print("✓ Saved: s_vs_z_multi_HII")

# =============================================================================
# PLOT 4f: dτ vs z
# =============================================================================

def plot_dtau(ax):
    for hii_factor in sorted(tau_results.keys()):
        r = tau_results[hii_factor]
        ax.plot(np.asarray(r['z_mid']), np.asarray(r['dtau']),
                color=cmap_eff(norm_eff(hii_factor)),
                linewidth=2.0, marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Optical Depth Element $d\tau$')
    ax.invert_xaxis()
    sm = mpl.cm.ScalarMappable(cmap=cmap_eff, norm=norm_eff)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'HII\_EFF\_FACTOR')

save_pdf_png(plot_dtau, plot_dir, "dtau_vs_z_multi_HII",
             title=r'Optical Depth Element $d\tau$ vs Redshift')
print("✓ Saved: dtau_vs_z_multi_HII")

# =============================================================================
# PLOT 4g: Cumulative τ vs z
# =============================================================================

def plot_tau(ax):
    for hii_factor in sorted(tau_results.keys()):
        r = tau_results[hii_factor]
        tau_total = float(np.asarray(r['tau_total']))
        ax.plot(np.asarray(r['z_mid']), np.asarray(r['tau']),
                color=cmap_eff(norm_eff(hii_factor)),
                linewidth=2.0, marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$')
    ax.invert_xaxis()
    sm = mpl.cm.ScalarMappable(cmap=cmap_eff, norm=norm_eff)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'HII\_EFF\_FACTOR')

save_pdf_png(plot_tau, plot_dir, "tau_vs_z_multi_HII",
             title='Cumulative Optical Depth vs Redshift')
print("✓ Saved: tau_vs_z_multi_HII")

# =============================================================================
# PLOT 4h: Total τ vs HII_EFF_FACTOR
# =============================================================================

tau_total_vals = [float(np.asarray(tau_results[h]['tau_total']))
                  for h in hii_vals if h in tau_results]

def plot_tau_total(ax):
    ax.plot(hii_vals, tau_total_vals, 'o-', linewidth=3)
    ax.set_xlabel(r'HII\_EFF\_FACTOR')
    ax.set_ylabel(r'Total Optical Depth $\tau$')

save_pdf_png(plot_tau_total, plot_dir,
             "tau_total_vs_HII_EFF",
             title='Total Optical Depth vs HII\_EFF\_FACTOR')
print("✓ Saved: tau_total_vs_HII_EFF")

print("\n" + "="*70)
print("REIONIZATION HISTORY + OPTICAL DEPTH COMPLETE!")
print("="*70)

# %%
# =============================================================================
# CELL 5: Compute kSZ Integrand — Rotated Skewed LOS only (θ=angle_deg°)
# kSZ integrand = (1 + δ) × x_e × v_z / c × e^(-τ(z))
# Full box face: Nlos_dim = Ndim² per HII_EFF_FACTOR value.
# Results cached to .npz for fast reload.
# =============================================================================

print("\n" + "="*70)
print(f"COMPUTING kSZ INTEGRAND — ROTATED SKEWED LOS (θ={angle_deg}°)")
print("="*70)

c_Mpc_s = 299792.458 / 3.08567758e19
print(f"Speed of light: c = {c_Mpc_s:.6e} Mpc/s")

Ndim      = int(user_params.HII_DIM)
Lbox      = float(user_params.BOX_LEN)
cell_size = Lbox / Ndim

kSZ_results = {}

for hii_factor in HII_EFF_FACTORS:
    if hii_factor not in lightcones or lightcones[hii_factor] is None:
        continue
    if hii_factor not in tau_results:
        continue

    print(f"\n{'='*70}")
    print(f"HII_EFF_FACTOR = {hii_factor:.2f}", flush=True)
    print(f"{'='*70}")

    # ==========================================================================
    # Cache check
    # ==========================================================================
    cache_subdir = os.path.join(main_cache_dir, f"EFF{hii_factor:.2f}")
    cache_file   = os.path.join(cache_subdir,
                                 f"kSZ_integrand_rot_angle{angle_deg}.npz")

    if os.path.exists(cache_file):
        print(f"  ✓ Loading cached rotated integrand...", flush=True)
        cached = np.load(cache_file)
        pos_ax = np.asarray(cached['pos_axis'])
        kSZ_results[hii_factor] = {
            'kSZ_integrand_rot' : cached['kSZ_integrand_rot'],
            'visibility'        : cached['visibility'],
            'tau_at_lightcone'  : cached['tau_at_lightcone'],
            'red_axis'          : cached['red_axis'],
            'pos_axis'          : pos_ax,
            'ds'                : cached['ds'],
            's_axis'            : cached['s_axis'],
            'ind_z'             : cached['ind_z'],
            'Nlos_dim'          : Ndim * Ndim,
            'Ndim'              : Ndim,
            'delta_z'           : float(pos_ax[1] - pos_ax[0]),
            'cell_size'         : cell_size,
        }
        r = kSZ_results[hii_factor]
        print(f"  Rotated RMS={np.sqrt(np.mean(r['kSZ_integrand_rot']**2)):.4e}")
        continue

    # ==========================================================================
    # Extract fields
    # ==========================================================================
    lightcone = lightcones[hii_factor]
    res       = tau_results[hii_factor]

    red_axis  = np.asarray(res['red_axis'])
    z_mid     = np.asarray(res['z_mid'])
    tau       = np.asarray(res['tau'])
    pos_axis  = np.asarray(res['pos_axis'])
    ds        = np.asarray(res['ds'])

    Nbins   = len(red_axis)
    delta_z = float(pos_axis[1] - pos_axis[0])
    s_axis  = pos_axis - pos_axis[0]

    red_axis_full = np.asarray(lightcone.lightcone_redshifts)
    ind_z = np.where(red_axis_full <= z_max)[0]

    _Delta_3d = np.array(lightcone.density[:, :, ind_z],  dtype=np.float32) + 1.0
    _xHI_3d   = np.array(lightcone.xH_box[:, :, ind_z],   dtype=np.float32)
    _vel_3d   = np.array(lightcone.velocity[:, :, ind_z],  dtype=np.float32)

    print(f"3D field shape: {_Delta_3d.shape}  (Ndim x Ndim x Nbins)")

    Nlos_dim = Ndim * Ndim
    LOS_ind  = np.array([[i, j] for i in range(Ndim) for j in range(Ndim)])
    print(f"LOS grid: {Ndim}×{Ndim} = {Nlos_dim} skewers (full box face)", flush=True)

    # ==========================================================================
    # Rotated skewer extraction
    # ==========================================================================
    def _periodic(n, ngrid):
        return int(round(float(n))) % int(ngrid)

    def rotated_skewer(field, x_start, y_idx):
        skewer = np.empty(Nbins, dtype=np.float32)
        for i in range(Nbins):
            s      = s_axis[i]
            z_cont = s * cos_a / delta_z
            x_cont = float(x_start) + s * sin_a / cell_size
            z0 = int(np.floor(z_cont))
            z1 = z0 + 1
            fz = z_cont - z0
            z0 = min(max(z0, 0), Nbins - 1)
            z1 = min(max(z1, 0), Nbins - 1)
            x  = _periodic(x_cont, Ndim)
            skewer[i] = (field[x, y_idx, z0] * (1 - fz) +
                         field[x, y_idx, z1] * fz)
        return skewer

    density_rot  = np.zeros((Nlos_dim, Nbins), dtype=np.float32)
    xH_box_rot   = np.zeros((Nlos_dim, Nbins), dtype=np.float32)
    velocity_rot = np.zeros((Nlos_dim, Nbins), dtype=np.float32)

    for k, (x0, y0) in enumerate(LOS_ind):
        density_rot[k]  = rotated_skewer(_Delta_3d, int(x0), int(y0))
        xH_box_rot[k]   = rotated_skewer(_xHI_3d,   int(x0), int(y0))
        velocity_rot[k] = rotated_skewer(_vel_3d,    int(x0), int(y0))
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{Nlos_dim} skewers done", flush=True)

    print(f"  Extraction complete.")
    print(f"  <1+δ>={density_rot.mean():.4f}  "
          f"<xHI>={xH_box_rot.mean():.4f}  <v>={velocity_rot.mean():.4e}")

    # Visibility function
    tau_extended     = np.concatenate([[0.0], tau])
    tau_at_lightcone = np.interp(red_axis,
                                  np.concatenate([[red_axis[0]], z_mid]),
                                  tau_extended)
    visibility = np.exp(-tau_at_lightcone)
    vis_broad  = visibility[np.newaxis, :]

    print(f"τ range      : [{tau_at_lightcone.min():.6f}, {tau_at_lightcone.max():.6f}]")
    print(f"e^(-τ) range : [{visibility.min():.6f}, {visibility.max():.6f}]")

    # kSZ integrand
    x_e_rot         = 1.0 - xH_box_rot
    kSZ_integrand_rot = density_rot * x_e_rot * velocity_rot / c_Mpc_s * vis_broad

    print(f"  Rotated — RMS={np.sqrt(np.mean(kSZ_integrand_rot**2)):.4e}  "
          f"mean={kSZ_integrand_rot.mean():.4e}", flush=True)

    # Save to cache
    os.makedirs(cache_subdir, exist_ok=True)
    np.savez_compressed(
        cache_file,
        kSZ_integrand_rot = kSZ_integrand_rot,
        visibility        = visibility,
        tau_at_lightcone  = tau_at_lightcone,
        red_axis          = red_axis,
        pos_axis          = pos_axis,
        ds                = ds,
        s_axis            = s_axis,
        ind_z             = ind_z,
    )
    print(f"  ✓ Cached to {cache_file}", flush=True)

    kSZ_results[hii_factor] = {
        'kSZ_integrand_rot' : kSZ_integrand_rot,
        'visibility'        : visibility,
        'tau_at_lightcone'  : tau_at_lightcone,
        'red_axis'          : red_axis,
        'pos_axis'          : pos_axis,
        'ds'                : ds,
        'ind_z'             : ind_z,
        'Nlos_dim'          : Nlos_dim,
        'Ndim'              : Ndim,
        's_axis'            : s_axis,
        'delta_z'           : delta_z,
        'cell_size'         : cell_size,
        'LOS_ind'           : LOS_ind,
    }

print("\n" + "="*70)
print(f"kSZ INTEGRAND COMPLETE — ROTATED (θ={angle_deg}°)")
print(f"Computed for {len(kSZ_results)} HII_EFF_FACTOR values")
print("="*70)

# Summary
print(f"\n{'HII_EFF':<10} {'RMS_rot':<14} {'vis_min':<10} {'vis_max':<10}")
print("-" * 48)
for hii_factor in sorted(kSZ_results.keys()):
    r     = kSZ_results[hii_factor]
    rms_r = np.sqrt(np.mean(r['kSZ_integrand_rot']**2))
    print(f"{hii_factor:<10.2f} {rms_r:<14.4e} "
          f"{r['visibility'].min():<10.4f} {r['visibility'].max():<10.4f}")

# =============================================================================
# PLOT: Visibility Function e^(-τ) vs z
# =============================================================================

def plot_visibility(ax):
    for hii_factor in sorted(kSZ_results.keys()):
        r = kSZ_results[hii_factor]
        ax.plot(np.asarray(r['red_axis']), r['visibility'],
                color=cmap_eff(norm_eff(hii_factor)),
                linewidth=2.5, alpha=0.8)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Visibility Function $e^{-\tau(z)}$')
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    sm = mpl.cm.ScalarMappable(cmap=cmap_eff, norm=norm_eff)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(r'HII\_EFF\_FACTOR')

save_pdf_png(plot_visibility, plot_dir,
             "visibility_function_multi_HII",
             title=r'Visibility Function $e^{-\tau(z)}$')
print("✓ Saved: visibility_function_multi_HII")

# %%
# =============================================================================
# CELL 6: Line-of-Sight Integrated kSZ Maps — Rotated Skewers only
# kSZ(z=5) = ∫ n_e0 σ_T (1/a²) (1+δ) x_e v_z/c e^(-τ) ds
# =============================================================================

print("\n" + "="*70)
print(f"LINE-OF-SIGHT kSZ MAP INTEGRATION — ROTATED (θ={angle_deg}°)")
print("="*70)

c_cm_s        = 3.0e10
sigma_T_cm2   = 6.6525e-25
n_e0_cm3      = 2.06e-7
Mpc_to_cm     = 3.0857e24
prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s
c_Mpc_s_cell6 = 299792.458 / 3.08567758e19

print(f"Prefactor n_e0 × σ_T × c = {prefactor_cgs:.4e} s⁻¹")

kSZ_map_results = {}

for hii_factor in HII_EFF_FACTORS:
    if hii_factor not in kSZ_results or hii_factor not in tau_results:
        continue

    print(f"\n{'='*70}")
    print(f"HII_EFF_FACTOR = {hii_factor:.2f}", flush=True)
    print(f"{'='*70}")

    res      = kSZ_results[hii_factor]
    red_axis = np.asarray(res['red_axis'])
    ds_Mpc   = np.asarray(res['ds'])
    ds_cm    = ds_Mpc * Mpc_to_cm

    # Scale factor midpoints
    a             = 1.0 / (1.0 + red_axis)
    a_squared_mid = 0.5 * (a[:-1]**2 + a[1:]**2)
    weight        = (prefactor_cgs / a_squared_mid) * (ds_cm / c_cm_s)

    # Rotated skewers
    ki_rot     = np.asarray(res['kSZ_integrand_rot'], dtype=np.float64)
    ki_rot_mid = 0.5 * (ki_rot[:, :-1] + ki_rot[:, 1:])
    kSZ_1d_rot = np.sum(weight[None, :] * ki_rot_mid, axis=1)

    # Reshape to 2D map
    Nlos_dim    = res['Nlos_dim']
    npix_map    = int(np.floor(np.sqrt(Nlos_dim)))
    n_use       = npix_map * npix_map
    kSZ_map_rot = kSZ_1d_rot[:n_use].reshape(npix_map, npix_map)

    print(f"Map shape: {kSZ_map_rot.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map_rot**2)):.4e}  "
          f"Std={kSZ_map_rot.std():.4e}")

    kSZ_map_results[hii_factor] = {
        'kSZ_map'     : kSZ_map_rot,
        'kSZ_1d'      : kSZ_1d_rot,
        'npix_map'    : npix_map,
        'pix_size_Mpc': Lbox / npix_map,
    }

print("\n" + "="*70)
print(f"kSZ MAP INTEGRATION COMPLETE — {len(kSZ_map_results)} histories")
print("="*70)

# Summary
print(f"\n{'HII_EFF':<10} {'npix':<8} {'pix[Mpc]':<10} {'RMS':<14} {'Std':<14}")
print("-" * 60)
for hii_factor in sorted(kSZ_map_results.keys()):
    r = kSZ_map_results[hii_factor]
    m = r['kSZ_map']
    print(f"{hii_factor:<10.2f} {r['npix_map']:<8} "
          f"{r['pix_size_Mpc']:<10.3f} "
          f"{np.sqrt(np.mean(m**2)):<14.4e} "
          f"{m.std():<14.4e}")

# =============================================================================
# PLOT: kSZ Maps colored by z50 — stacked
# =============================================================================

# Use z50 for ordering and color
z50_map = {
    hii_factor: tau_results[hii_factor]['z_mid'][
        np.argmin(np.abs(tau_results[hii_factor]['x_e_mid'] - 0.5))
    ]
    for hii_factor in tau_results
}

# Global colour scale across all maps
vmax_global = np.percentile(
    [np.percentile(np.abs(kSZ_map_results[h]['kSZ_map']), 99)
     for h in kSZ_map_results], 90
)

fig, axes = plt.subplots(len(kSZ_map_results), 1,
                         figsize=(10, 4*len(kSZ_map_results)),
                         constrained_layout=True)
if len(kSZ_map_results) == 1:
    axes = [axes]

for ax, hii_factor in zip(axes, sorted(kSZ_map_results.keys())):
    r   = kSZ_map_results[hii_factor]
    im  = ax.imshow(r['kSZ_map'].T, cmap='seismic', origin='lower',
                    extent=[0, Lbox, 0, Lbox], aspect='equal',
                    vmin=-vmax_global, vmax=vmax_global)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('x [Mpc]')
    ax.set_ylabel('y [Mpc]')
    color = cmap_eff(norm_eff(hii_factor))
    ax.text(0.02, 0.98,
            f'HII\_EFF={hii_factor:.1f}  z₅₀={z50_map.get(hii_factor, 0):.2f}',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8,
                     edgecolor='black', linewidth=1.5))

plot_name = f"kSZ_maps_rot_angle{angle_deg}_multi_HII"
fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
fig.suptitle(rf'kSZ Maps — Rotated ($\theta={angle_deg}°$)',
             fontsize=18, fontweight='bold', y=0.995)
fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot_name}")
plt.close(fig)


# %%
# =============================================================================
# CELL 7: kSZ Power Spectrum — P(k), C_ℓ, D_ℓ colored by z50
# =============================================================================

print("\n" + "="*70)
print("COMPUTING kSZ POWER SPECTRA — COLORED BY z50")
print("="*70)

T_CMB_0_K        = 2.725
z_obs            = 5.0
T_CMB_z5_uK      = T_CMB_0_K * 1e6
D_A_Mpc          = 1300.0
chi_comoving_Mpc = D_A_Mpc * (1 + z_obs)

print(f"T_CMB(z=0) = {T_CMB_0_K:.3f} K  →  {T_CMB_z5_uK:.2f} μK")
print(f"χ(z=5)     = {chi_comoving_Mpc:.1f} Mpc")


def compute_ps(ksz_map_2d, box_size_Mpc, n_kbins=35):
    npix     = ksz_map_2d.shape[0]
    pix_size = box_size_Mpc / npix
    m        = ksz_map_2d - ksz_map_2d.mean()

    fft_shift = np.fft.fftshift(np.fft.fft2(m))
    ps2d      = (pix_size / npix)**2 * np.abs(fft_shift)**2

    dk    = 2 * np.pi / (npix * pix_size)
    kx    = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    ky    = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), n_kbins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    P1d       = np.full(n_kbins, np.nan)
    P1d_err   = np.full(n_kbins, np.nan)
    n_modes   = np.zeros(n_kbins)

    for i in range(n_kbins):
        mask = (kgrid >= k_bins[i]) & (kgrid < k_bins[i+1])
        n_modes[i] = mask.sum()
        if n_modes[i] > 0:
            vals      = ps2d[mask]
            P1d[i]    = vals.mean()
            P1d_err[i] = vals.std() / np.sqrt(n_modes[i])

    # Cosmic variance
    k_volume       = (box_size_Mpc / (2 * np.pi))**2
    n_modes_cosmic = 2 * np.pi * k_centers * k_volume * (k_bins[1:] - k_bins[:-1])
    cv_frac        = np.where(n_modes_cosmic > 0,
                              1.0 / np.sqrt(n_modes_cosmic), np.nan)
    P1d_err_cosmic = P1d * cv_frac
    P1d_err_total  = np.sqrt(P1d_err**2 + P1d_err_cosmic**2)

    ell    = k_centers * chi_comoving_Mpc / 0.67
    Cl     = P1d * 0.67**2 / D_A_Mpc**2
    Dl     = ell * (ell + 1) * Cl / (2 * np.pi)
    Dl_uK2 = Dl * T_CMB_z5_uK**2

    def _prop_err(P_err):
        Cl_e = P_err * 0.67**2 / D_A_Mpc**2
        return ell * (ell + 1) * Cl_e / (2 * np.pi) * T_CMB_z5_uK**2

    return dict(
        k_centers        = k_centers,
        P1d              = P1d,
        P1d_err_total    = P1d_err_total,
        n_modes          = n_modes,
        ell              = ell,
        Dl_uK2           = Dl_uK2,
        Dl_uK2_err_total = _prop_err(P1d_err_total),
        dk               = dk,
        npix             = npix,
        pix_size         = pix_size,
    )


# Colormap by z50
cmap_z50 = mpl.cm.plasma
norm_z50  = mpl.colors.Normalize(
    vmin=min(z50_map.values()),
    vmax=max(z50_map.values())
)

power_spectrum_results = {}

for hii_factor in HII_EFF_FACTORS:
    if hii_factor not in kSZ_map_results:
        continue
    r  = kSZ_map_results[hii_factor]
    ps = compute_ps(r['kSZ_map'], Lbox)
    power_spectrum_results[hii_factor] = ps
    valid = ~np.isnan(ps['P1d']) & (ps['P1d'] > 0)
    print(f"  HII_EFF={hii_factor:.2f}  z50={z50_map.get(hii_factor,0):.2f}  "
          f"npix={ps['npix']}  dk={ps['dk']:.5f} Mpc⁻¹  "
          f"valid bins={valid.sum()}")

print(f"\n✓ Power spectra computed for {len(power_spectrum_results)} histories")

# =============================================================================
# PLOT 1: P(k) colored by z50
# =============================================================================

def plot_Pk(ax):
    for hii_factor, ps in power_spectrum_results.items():
        color = cmap_z50(norm_z50(z50_map[hii_factor]))
        valid = ~np.isnan(ps['P1d']) & (ps['P1d'] > 0)
        ax.loglog(ps['k_centers'][valid], ps['P1d'][valid],
                  color=color, linewidth=2.0, alpha=0.85)
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax.set_ylabel(r'$P(k)$ [Mpc$^{2}$]')
    sm = mpl.cm.ScalarMappable(cmap=cmap_z50, norm=norm_z50)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(
        r'Reionization Midpoint $z_{50}$')

save_pdf_png(plot_Pk, plot_dir,
             f"kSZ_Pk_rot_angle{angle_deg}_vs_z50",
             title=r'kSZ Power Spectrum $P(k)$')
print(f"✓ Saved: kSZ_Pk_rot_angle{angle_deg}_vs_z50")

# =============================================================================
# PLOT 2: D_ℓ colored by z50  *** MAIN RESULT ***
# =============================================================================

def plot_Dl(ax):
    for hii_factor, ps in power_spectrum_results.items():
        color = cmap_z50(norm_z50(z50_map[hii_factor]))
        valid = (~np.isnan(ps['Dl_uK2']) & (ps['Dl_uK2'] > 0)
                 & (ps['ell'] > 10))
        ax.errorbar(ps['ell'][valid], ps['Dl_uK2'][valid],
                    yerr=ps['Dl_uK2_err_total'][valid],
                    color=color, linewidth=2.5, alpha=0.85,
                    capsize=2, capthick=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    sm = mpl.cm.ScalarMappable(cmap=cmap_z50, norm=norm_z50)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.02).set_label(
        r'Reionization Midpoint $z_{50}$')

save_pdf_png(plot_Dl, plot_dir,
             f"kSZ_Dl_rot_angle{angle_deg}_vs_z50",
             title=rf'kSZ $D_\ell$ — Rotated ($\theta={angle_deg}°$)  *** MAIN RESULT ***')
print(f"✓ Saved: kSZ_Dl_rot_angle{angle_deg}_vs_z50  *** MAIN RESULT ***")

# =============================================================================
# PLOT 3: Peak D_ℓ vs z50
# =============================================================================

def plot_peak_Dl(ax):
    z50_list  = []
    peak_list = []
    for hii_factor, ps in power_spectrum_results.items():
        valid = (ps['Dl_uK2'] > 0) & np.isfinite(ps['Dl_uK2'])
        if np.any(valid):
            z50_list.append(z50_map[hii_factor])
            peak_list.append(np.max(ps['Dl_uK2'][valid]))
    order = np.argsort(z50_list)
    ax.plot(np.array(z50_list)[order], np.array(peak_list)[order],
            'o-', linewidth=3)
    ax.set_xlabel(r'Reionization Midpoint $z_{50}$')
    ax.set_ylabel(r'Peak $D_\ell$ [$\mu$K$^2$]')

save_pdf_png(plot_peak_Dl, plot_dir,
             f"kSZ_peak_Dl_rot_angle{angle_deg}_vs_z50",
             title='Peak kSZ Power vs Reionization Timing')
print(f"✓ Saved: kSZ_peak_Dl_rot_angle{angle_deg}_vs_z50")

print("\n" + "="*70)
print("ALL kSZ POWER SPECTRUM ANALYSIS COMPLETE!")
print("="*70)


