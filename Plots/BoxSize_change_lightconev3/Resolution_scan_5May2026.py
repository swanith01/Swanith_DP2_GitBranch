# %%
# =============================================================================
# LIGHTCONE GENERATION - RESOLUTION SCAN
# py21cmfast v3.4
# Fixed Box Size (BOX_LEN=1000 Mpc), varying HII_DIM from 128 to 800
# =============================================================================

# =============================================================================
# CELL 1: Imports and Setup
# =============================================================================

import numpy as np
import matplotlib as mpl

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt

import py21cmfast as p21c
from py21cmfast import plotting

import os
from datetime import datetime

print(f"py21cmfast version: {p21c.__version__}")

# --- Skewed LOS parameters (defined early, used in CELL 5) ---
angle_deg = 10    # Rotation angle in degrees
Nlos      = None  # None → full Ndim² box face per resolution

# =============================================================================
# CELL 1a: Create Output Directory for Plots
# =============================================================================

plot_dir = "RESOLUTION_SCAN_5May2026/plots"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
else:
    print(f"Directory already exists: {plot_dir}")

print(f"All plots will be saved to: {os.path.abspath(plot_dir)}")

# =============================================================================
# CELL 1b: Standardized Plot Settings
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 20,
    'axes.labelsize': 30,
    'axes.titlesize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['axes.grid'] = False

print("✓ Plot settings applied")

PDF_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 30,
    'axes.labelsize': 29,
    'axes.titlesize': 40,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 20,
    'figure.titlesize': 28,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}

PNG_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 15,
    'axes.labelsize': 25,
    'axes.titlesize': 18,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'figure.titlesize': 15,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}
print("✓ PDF and PNG styles defined")

def save_pdf_png(plot_func, plot_dir, plot_name, title=None):
    """plot_func(ax) draws onto provided Axes — matches Box_Size_scan convention."""
    with mpl.rc_context(PDF_STYLE):
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        ax.set_title("")
        fig.savefig(f"{plot_dir}/{plot_name}.pdf")
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        if title is not None:
            ax.set_title(title, fontweight='bold')
        fig.savefig(f"{plot_dir}/{plot_name}.png")
        plt.close(fig)

print("✓ save_pdf_png function defined (plot_func pattern)")


# =============================================================================
# CELL 1c: Define Parameters
# =============================================================================

# **FIXED: BOX SIZE**
BOX_LEN_FIXED = 800.0  # Mpc

# **SCAN: HII_DIM (Resolution: 128, 256, 512, 800)**
HII_DIM_VALUES = np.array([128, 256, 512, 1024])

# Redshift range for the lightcone
z_min = 5.0
z_max = 20.0

print(f"\n=== PARAMETER SCAN SETUP ===")
print(f"Fixed BOX_LEN = {BOX_LEN_FIXED:.0f} Mpc")
print(f"\nScanning HII_DIM (Resolution):")
print(f"  Number of values: {len(HII_DIM_VALUES)}")
print(f"  Range: {HII_DIM_VALUES.min():.0f} → {HII_DIM_VALUES.max():.0f}")
print(f"  Values: {HII_DIM_VALUES}")
print(f"\nTotal simulations: {len(HII_DIM_VALUES)}")
print(f"\nRedshift range: z = {z_min} → {z_max}")

print("\n=== DEFAULT COSMOLOGY ===")
print(p21c.CosmoParams())

print("\n=== DEFAULT ASTROPHYSICS (REIONIZATION KEPT DEFAULT) ===")
default_astro = p21c.AstroParams()
print(default_astro)

print("\n=== DEFAULT FLAGS ===")
print(p21c.FlagOptions())

# Print resolution info for each HII_DIM
print("\n=== RESOLUTION DETAILS ===")
print(f"{'HII_DIM':<12} {'BOX_LEN [Mpc]':<15} {'Cell Size [Mpc]':<20} {'Cell Size [kpc]':<15}")
print("-" * 75)
for hii_dim in HII_DIM_VALUES:
    cell_size_mpc = BOX_LEN_FIXED / hii_dim
    cell_size_kpc = cell_size_mpc * 1000
    print(f"{hii_dim:<12} {BOX_LEN_FIXED:<15.0f} {cell_size_mpc:<20.3f} {cell_size_kpc:<15.1f}")

# --- Skewed LOS geometry ---
angle_rad = np.deg2rad(angle_deg)
sin_a     = float(np.sin(angle_rad))
cos_a     = float(np.cos(angle_rad))

print(f"\n=== SKEWED LOS PARAMETERS ===")
print(f"  Rotation angle : {angle_deg}°")
print(f"  sin(θ)         : {sin_a:.4f}")
print(f"  cos(θ)         : {cos_a:.4f}")
print(f"  Nlos mode      : FULL BOX FACE (Ndim² per resolution)")
print(f"\n  Skewers and artefact suppression by resolution:")
print(f"  {'HII_DIM':<10} {'Cell[Mpc]':<12} {'Nlos=Ndim²':<12} "
      f"{'k_box [Mpc⁻¹]':<16} {'k_box_rot [Mpc⁻¹]':<20} {'Suppression':<12}")
print(f"  {'-'*85}")
for hii_dim in HII_DIM_VALUES:
    k_box     = 2 * np.pi / BOX_LEN_FIXED           # constant — box fixed
    k_box_rot = 2 * np.pi / (BOX_LEN_FIXED / sin_a) # constant
    cell_mpc  = BOX_LEN_FIXED / hii_dim
    print(f"  {hii_dim:<10} {cell_mpc:<12.3f} {hii_dim**2:<12} "
          f"{k_box:<16.5f} {k_box_rot:<20.5f} {1/sin_a:.1f}×")

# %%
# =============================================================================
# CELL 2: Run Lightcone Simulations - HII_DIM (Resolution) Scan
# WITH ROBUST CACHE CHECKING AND RETRIEVAL
# =============================================================================

import time
import os
import glob

print("\n" + "="*70)
print("RUNNING HII_DIM (RESOLUTION) SCAN")
print("="*70)

# =============================================================================
# CACHE DIRECTORY
# =============================================================================
try:
    # Running as .py script via PBS
    main_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "cache")
except NameError:
    # Running in Jupyter — explicitly set path next to notebook
    main_cache_dir = os.path.join(
        os.path.dirname(os.path.abspath("Resolution_change_lightconeV3.ipynb")),
        "RESOLUTION_SCAN_5May2026", "cache"
    )

if not os.path.exists(main_cache_dir):
    os.makedirs(main_cache_dir)
    print(f"Created cache directory: {main_cache_dir}")
else:
    print(f"Cache directory exists: {main_cache_dir}")

print(f"Absolute cache path: {main_cache_dir}", flush=True)

lightcones      = {}
scan_start_time = time.time()

for idx, hii_dim in enumerate(HII_DIM_VALUES):
    sim_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"SIMULATION {idx+1}/{len(HII_DIM_VALUES)}")
    print(f"HII_DIM = {hii_dim}")
    print(f"BOX_LEN = {BOX_LEN_FIXED:.0f} Mpc")
    print(f"Cell size = {BOX_LEN_FIXED/hii_dim:.3f} Mpc = {BOX_LEN_FIXED/hii_dim*1000:.1f} kpc")
    print(f"{'='*70}", flush=True)

    user_params = p21c.UserParams(
        HII_DIM=hii_dim,
        BOX_LEN=BOX_LEN_FIXED,
        USE_INTERPOLATION_TABLES=True,
        N_THREADS=32
    )

    print(f"Redshift range: z = {z_min} → {z_max}")
    print(f"Resolution: {user_params.HII_DIM}³ cells")
    print(f"Total cells: {user_params.HII_DIM**3:,}", flush=True)

    cache_subdir = os.path.join(main_cache_dir,
                                 f"DIM{hii_dim}_BOX{BOX_LEN_FIXED:.0f}")
    os.makedirs(cache_subdir, exist_ok=True)

    # ========================================================================
    # CACHE CHECK — look for complete LightCone file
    # ========================================================================
    cached_files = glob.glob(
        os.path.join(cache_subdir, f"LightCone_*z{z_min:.2f}*.h5")
    )

    # Verify file is not empty/corrupted
    valid_cache = []
    for f in cached_files:
        size_mb = os.path.getsize(f) / 1e6
        if size_mb > 1.0:
            valid_cache.append((f, size_mb))
        else:
            print(f"  ⚠ Suspicious cache file ({size_mb:.2f} MB) — ignoring: {f}")

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
                random_seed=37,
                direc=cache_subdir,
                write=False
            )

            lightcones[hii_dim] = lightcone
            load_time = time.time() - sim_start_time

            print(f"  ✓ Loaded in {load_time:.1f} s")
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
                      f"z(90%)={z_90:.2f}  Δz={z_10-z_90:.2f}", flush=True)
            except:
                pass
            continue

        except Exception as e:
            print(f"  ✗ Load failed: {e}")
            print(f"  Will recompute...", flush=True)

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
            random_seed=37,
            direc=cache_subdir,
            write=True
        )

        lightcones[hii_dim] = lightcone
        sim_time = time.time() - sim_start_time

        # ← Save lightcone immediately to disk
        try:
            lightcone.save(direc=cache_subdir)
            print(f"  ✓ Lightcone saved to disk: {cache_subdir}", flush=True)
        except Exception as e:
            print(f"  ⚠ Could not save lightcone: {e}", flush=True)

        print(f"\n✓ Simulation complete! ({sim_time/60:.2f} min)")
        print(f"  Cache: {cache_subdir}")
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
        eta_min  = (len(HII_DIM_VALUES) - idx - 1) * avg_time / 60
        print(f"  Progress: {idx+1}/{len(HII_DIM_VALUES)} "
              f"({100*(idx+1)/len(HII_DIM_VALUES):.0f}%)  "
              f"ETA: {eta_min:.1f} min", flush=True)

    except Exception as e:
        print(f"\n✗ Simulation FAILED: {e}", flush=True)
        lightcones[hii_dim] = None

# =============================================================================
# Summary
# =============================================================================
total_time = time.time() - scan_start_time
successful = sum(1 for lc in lightcones.values() if lc is not None)

print(f"\n{'='*70}")
print("ALL SIMULATIONS COMPLETE")
print(f"Total time  : {total_time/60:.2f} min ({total_time/3600:.2f} h)")
print(f"Successful  : {successful}/{len(HII_DIM_VALUES)}")
print(f"Cache path  : {main_cache_dir}")
print("="*70, flush=True)

# %%
# # =============================================================================
# # NOT FOR REPORTS --- IGNORE ---
# # CELL 3: 2D SLICES AT z = 8 (ALL RESOLUTIONS)
# # =============================================================================

# print("\n" + "="*70)
# print("GENERATING 2D SLICES AT z = 8.0")
# print("="*70)

# target_z = 8.0

# # Use all resolutions
# print(f"Using all {len(HII_DIM_VALUES)} HII_DIM values for slices")

# fields_info = [
#     ('brightness_temp', '21cm Brightness Temperature [mK]', 'EoR'),
#     ('xH_box', 'Neutral Fraction (xHI)', 'viridis'),
#     ('density', 'Overdensity δ', 'magma'),
#     ('velocity', 'Line-of-Sight Velocity [km/s]', 'RdBu_r'),
# ]

# for field_name, field_label, cmap_name in fields_info:
#     print(f"\nProcessing {field_name}...")
    
#     # Create figure with all resolutions
#     fig, axes = plt.subplots(1, len(HII_DIM_VALUES), 
#                             figsize=(5*len(HII_DIM_VALUES), 5.5), 
#                             constrained_layout=True)
    
#     if len(HII_DIM_VALUES) == 1:
#         axes = [axes]
    
#     # First pass: find global min/max for consistent colorbar
#     vmin_global = np.inf
#     vmax_global = -np.inf
#     slices_data = []
    
#     for hii_dim in HII_DIM_VALUES:
#         if hii_dim not in lightcones or lightcones[hii_dim] is None:
#             slices_data.append((None, None, hii_dim))
#             continue
            
#         lightcone = lightcones[hii_dim]
#         z_values = lightcone.lightcone_redshifts
#         closest_idx = np.argmin(np.abs(z_values - target_z))
#         actual_z = z_values[closest_idx]
        
#         field_data = getattr(lightcone, field_name)
#         slice_2d = field_data[:, :, closest_idx]
        
#         slices_data.append((slice_2d, actual_z, hii_dim))
        
#         vmin_global = min(vmin_global, slice_2d.min())
#         vmax_global = max(vmax_global, slice_2d.max())
    
#     # Create rainbow colormap for resolution labels
#     cmap_rainbow = mpl.cm.rainbow
#     norm_rainbow = mpl.colors.Normalize(vmin=HII_DIM_VALUES.min(), vmax=HII_DIM_VALUES.max())
    
#     # Second pass: plot with consistent colorbar
#     for idx, (data_tuple, ax) in enumerate(zip(slices_data, axes)):
#         slice_2d, actual_z, hii_dim = data_tuple
        
#         if slice_2d is None:
#             ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
#                    transform=ax.transAxes, fontsize=16)
#             ax.set_title(f'DIM={hii_dim}', fontsize=14)
#             continue
        
#         im = ax.imshow(slice_2d.T, origin='lower', cmap=cmap_name,
#                       vmin=vmin_global, vmax=vmax_global,
#                       extent=[0, BOX_LEN_FIXED, 0, BOX_LEN_FIXED],
#                       aspect='auto')
        
#         ax.set_xlabel('Comoving Distance [Mpc]', fontsize=12)
#         ax.set_ylabel('Comoving Distance [Mpc]', fontsize=12)
        
#         # Get rainbow color for this resolution
#         color_label = cmap_rainbow(norm_rainbow(hii_dim))
        
#         # Calculate cell size
#         cell_size_mpc = BOX_LEN_FIXED / hii_dim
#         cell_size_kpc = cell_size_mpc * 1000
        
#         ax.set_title(f'DIM={hii_dim}\nCell={cell_size_mpc:.2f} Mpc ({cell_size_kpc:.1f} kpc)\nz={actual_z:.2f}', 
#                     fontsize=11, fontweight='bold',
#                     bbox=dict(boxstyle='round', facecolor=color_label, alpha=0.6, 
#                              edgecolor='black', linewidth=1.5))
        
#         cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         cbar.ax.tick_params(labelsize=10)
    
#     # Overall title
#     fig.suptitle(f'{field_label} at z ≈ {target_z} (BOX_LEN={BOX_LEN_FIXED:.0f} Mpc)', 
#                 fontsize=18, fontweight='bold')
    
#     # Save
#     plot_name = f"{field_name}_slice_z{int(target_z)}_resolution_all"
#     fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
#     fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
    
#     print(f"  ✓ Saved: {plot_name}")
#     print(f"    Range: [{vmin_global:.3f}, {vmax_global:.3f}]")
    
#     plt.close(fig)

# print("\n✓ 2D SLICE PLOTTING COMPLETE!")

# # =============================================================================
# # CELL 3b: Print Statistics for 2D Slices
# # =============================================================================

# print("\n" + "="*70)
# print("2D SLICE STATISTICS AT z = 8")
# print("="*70)

# for hii_dim in HII_DIM_VALUES:
#     if hii_dim not in lightcones or lightcones[hii_dim] is None:
#         continue
        
#     lightcone = lightcones[hii_dim]
#     z_values = lightcone.lightcone_redshifts
#     closest_idx = np.argmin(np.abs(z_values - target_z))
#     actual_z = z_values[closest_idx]
    
#     cell_size_mpc = BOX_LEN_FIXED / hii_dim
#     cell_size_kpc = cell_size_mpc * 1000
    
#     print(f"\nHII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc = {cell_size_kpc:.1f} kpc) at z = {actual_z:.2f}:")
    
#     for field_name, field_label, _ in fields_info:
#         field_data = getattr(lightcone, field_name)
#         slice_2d = field_data[:, :, closest_idx]
        
#         print(f"  {field_name:15s}: min={slice_2d.min():10.3e}, "
#               f"max={slice_2d.max():10.3e}, mean={slice_2d.mean():10.3e}")

# print("\n" + "="*70)

# # =============================================================================
# # CELL 3c: Summary Statistics for 2D Slices at z=8
# # =============================================================================

# print("\n" + "="*70)
# print(f"SUMMARY STATISTICS FOR 2D SLICES AT z ≈ {target_z}")
# print("="*70)

# for hii_dim in HII_DIM_VALUES:
#     if hii_dim not in lightcones or lightcones[hii_dim] is None:
#         continue
        
#     lightcone = lightcones[hii_dim]
    
#     # Find the closest redshift slice
#     z_values = lightcone.lightcone_redshifts
#     closest_idx = np.argmin(np.abs(z_values - target_z))
#     actual_z = z_values[closest_idx]
    
#     # Extract 2D slices
#     brightness_slice = lightcone.brightness_temp[:, :, closest_idx]
#     xHI_slice = lightcone.xH_box[:, :, closest_idx]
#     density_slice = lightcone.density[:, :, closest_idx]  # δ
#     velocity_slice = lightcone.velocity[:, :, closest_idx]
    
#     cell_size_mpc = BOX_LEN_FIXED / hii_dim
#     cell_size_kpc = cell_size_mpc * 1000
    
#     print(f"\nHII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc = {cell_size_kpc:.1f} kpc) at z = {actual_z:.2f}:")
#     print(f"  Brightness temp [mK]: min={brightness_slice.min():.2f}, "
#           f"max={brightness_slice.max():.2f}, mean={brightness_slice.mean():.2f}")
#     print(f"  Neutral fraction:     min={xHI_slice.min():.4f}, "
#           f"max={xHI_slice.max():.4f}, mean={xHI_slice.mean():.4f}")
#     print(f"  Overdensity δ:        min={density_slice.min():.3f}, "
#           f"max={density_slice.max():.3f}, mean={density_slice.mean():.3f}")
#     print(f"  Velocity [km/s]:      min={velocity_slice.min():.2f}, "
#           f"max={velocity_slice.max():.2f}, mean={velocity_slice.mean():.2f}")

# print("\n" + "="*70)

# # =============================================================================
# # CELL 3d: Summary Statistics for Full Lightcones (All Resolutions)
# # =============================================================================

# print("\n" + "="*70)
# print("SUMMARY STATISTICS FOR FULL LIGHTCONES (ALL RESOLUTIONS)")
# print("="*70)
# print(f"Fixed: BOX_LEN = {BOX_LEN_FIXED:.0f} Mpc")
# print(f"Total simulations: {len(HII_DIM_VALUES)}")

# for idx, hii_dim in enumerate(HII_DIM_VALUES):
#     if hii_dim not in lightcones or lightcones[hii_dim] is None:
#         print(f"\n[{idx+1}/{len(HII_DIM_VALUES)}] HII_DIM = {hii_dim}: FAILED")
#         continue
        
#     lightcone = lightcones[hii_dim]
#     cell_size_mpc = BOX_LEN_FIXED / hii_dim
#     cell_size_kpc = cell_size_mpc * 1000
    
#     print(f"\n[{idx+1}/{len(HII_DIM_VALUES)}] HII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc = {cell_size_kpc:.1f} kpc):")
#     print(f"  Redshift range: [{lightcone.lightcone_redshifts.min():.2f}, "
#           f"{lightcone.lightcone_redshifts.max():.2f}]")
    
#     # Full 3D field statistics
#     print(f"  Brightness temp [mK]: min={lightcone.brightness_temp.min():.2f}, "
#           f"max={lightcone.brightness_temp.max():.2f}, "
#           f"mean={lightcone.brightness_temp.mean():.2f}")
#     print(f"  Neutral fraction:     min={lightcone.xH_box.min():.4f}, "
#           f"max={lightcone.xH_box.max():.4f}, "
#           f"mean={lightcone.xH_box.mean():.4f}")
#     print(f"  Overdensity δ:        min={lightcone.density.min():.3f}, "
#           f"max={lightcone.density.max():.3f}, "
#           f"mean={lightcone.density.mean():.3f}")
#     print(f"  Velocity [km/s]:      min={lightcone.velocity.min():.2f}, "
#           f"max={lightcone.velocity.max():.2f}, "
#           f"mean={lightcone.velocity.mean():.2f}")

# print("\n" + "="*70)

# # =============================================================================
# # CELL 3e: Compact Summary Table
# # =============================================================================

# print("\n" + "="*70)
# print("COMPACT SUMMARY: REIONIZATION PROGRESS BY RESOLUTION")
# print("="*70)
# print(f"{'HII_DIM':<12} {'Cell[kpc]':<12} {'<xHI>':<10} {'<Tb>[mK]':<12} {'z_range':<15}")
# print("-" * 70)

# for hii_dim in HII_DIM_VALUES:
#     if hii_dim not in lightcones or lightcones[hii_dim] is None:
#         continue
        
#     lightcone = lightcones[hii_dim]
#     cell_size_kpc = (BOX_LEN_FIXED / hii_dim) * 1000
#     mean_xHI = lightcone.xH_box.mean()
#     mean_Tb = lightcone.brightness_temp.mean()
#     z_min_actual = lightcone.lightcone_redshifts.min()
#     z_max_actual = lightcone.lightcone_redshifts.max()
    
#     print(f"{hii_dim:<12} {cell_size_kpc:<12.1f} {mean_xHI:<10.4f} {mean_Tb:<12.2f} "
#           f"[{z_min_actual:.2f}, {z_max_actual:.2f}]")

# print("="*70)

# %%
# =============================================================================
# CELL 4: Reionization History Analysis - Resolution Scan
# =============================================================================
print("\n" + "="*70)
print("GENERATING REIONIZATION HISTORY COMPARISON")
print("="*70)

cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=HII_DIM_VALUES.min(), vmax=HII_DIM_VALUES.max())

# =============================================================================
# PLOT 4a: Ionization Fraction vs Redshift
# =============================================================================

def plot_reionization_xe(ax):
    for hii_dim in HII_DIM_VALUES:
        if hii_dim not in lightcones or lightcones[hii_dim] is None:
            continue
        lightcone = lightcones[hii_dim]
        z_nodes   = lightcone.node_redshifts[::-1]
        x_e_nodes = 1.0 - lightcone.global_xH[::-1]
        ax.plot(z_nodes, x_e_nodes, linewidth=2.5, color=cmap(norm(hii_dim)),
                marker='o', markersize=3, alpha=0.8, label=f'{hii_dim}³')
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Ionization Fraction $x_e$')
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    ax.legend(loc='best', ncol=1, title='Resolution')

save_pdf_png(plot_reionization_xe, plot_dir,
             "reionization_history_xe_resolution_all",
             title=rf'Reionization History: Ionization Fraction (BOX\_LEN={BOX_LEN_FIXED:.0f} Mpc)')
print("✓ Saved: reionization_history_xe_resolution_all")


# =============================================================================
# CELL 4f: Optical Depth Calculations for All Resolutions
# =============================================================================

print("\n" + "="*70)
print("OPTICAL DEPTH CALCULATIONS")
print("="*70)

c_km_s          = 2.998e5
h               = 0.6766
H0              = 100 * h
Omega_b         = 0.04897468161869667
Omega_m         = 0.30964144154550644
rho_crit_p_cm3  = 1.88e-29 * h**2 / (1.67e-24)
n_H0_cm3        = Omega_b * rho_crit_p_cm3
sigma_T_cm2     = 6.65e-25
cm_per_Mpc      = 3.086e24
n_e0_Mpc3       = n_H0_cm3 * cm_per_Mpc**3
sigma_T_Mpc2    = sigma_T_cm2 / cm_per_Mpc**2
prefactor       = n_e0_Mpc3 * sigma_T_Mpc2

print(f"  n_H0      = {n_H0_cm3:.6e} cm^-3")
print(f"  σ_T       = {sigma_T_cm2:.6e} cm^2")
print(f"  Prefactor = {prefactor:.6e} Mpc^-1")

tau_results = {}

for hii_dim in HII_DIM_VALUES:
    if hii_dim not in lightcones or lightcones[hii_dim] is None:
        continue

    lightcone = lightcones[hii_dim]
    red_axis  = lightcone.lightcone_redshifts
    pos_axis  = lightcone.lightcone_distances

    ind_z    = np.where(red_axis <= z_max)[0]
    red_axis = red_axis[ind_z]
    pos_axis = pos_axis[ind_z]

    z_nodes_sorted  = lightcone.node_redshifts[::-1]
    xHI_nodes_sorted = lightcone.global_xH[::-1]
    x_e_nodes_sorted = 1.0 - xHI_nodes_sorted

    x_e_interp = np.interp(red_axis, z_nodes_sorted, x_e_nodes_sorted)

    s  = pos_axis
    ds = np.diff(s)

    z_mid   = 0.5 * (red_axis[:-1] + red_axis[1:])
    x_e_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])

    dtau      = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds
    tau       = np.cumsum(dtau)
    tau_total = tau[-1]

    tau_results[hii_dim] = {
        'red_axis'  : red_axis,
        'pos_axis'  : pos_axis,
        's'         : s,
        'ds'        : ds,
        'z_mid'     : z_mid,
        'x_e_interp': x_e_interp,
        'x_e_mid'   : x_e_mid,
        'dtau'      : dtau,
        'tau'       : tau,
        'tau_total' : tau_total,
    }

print(f"\n{'HII_DIM':<12} {'Cell[kpc]':<12} {'z_range':<15} {'<ds>[Mpc]':<12} {'τ_total':<10}")
print("-" * 75)
for hii_dim in sorted(tau_results.keys()):
    r          = tau_results[hii_dim]
    red_axis   = np.asarray(r['red_axis'])
    ds         = np.asarray(r['ds'])
    tau_total  = float(np.asarray(r['tau_total']))
    cell_kpc   = (BOX_LEN_FIXED / hii_dim) * 1000
    print(f"{hii_dim:<12} {cell_kpc:<12.1f} [{red_axis.min():.2f}, {red_axis.max():.2f}] "
          f"{ds.mean():<12.3f} {tau_total:<10.6f}")

# =============================================================================
# PLOT: Cumulative τ vs z
# =============================================================================

def plot_tau_vs_z(ax):
    for hii_dim in sorted(tau_results.keys()):
        r = tau_results[hii_dim]
        ax.plot(np.asarray(r['z_mid']), np.asarray(r['tau']),
                linewidth=2.5, color=cmap(norm(hii_dim)),
                marker='o', markersize=3, alpha=0.8, label=f'{hii_dim}³')
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$')
    ax.invert_xaxis()
    ax.legend(loc='best', ncol=1, title='Resolution')

save_pdf_png(plot_tau_vs_z, plot_dir, "tau_vs_z_resolution_all",
             title=rf'Cumulative Optical Depth vs Redshift (BOX\_LEN={BOX_LEN_FIXED:.0f} Mpc)')
print("✓ Saved: tau_vs_z_resolution_all")

# =============================================================================
# PLOT: dτ vs z
# =============================================================================

def plot_dtau_vs_z(ax):
    for hii_dim in sorted(tau_results.keys()):
        r = tau_results[hii_dim]
        ax.plot(np.asarray(r['z_mid']), np.asarray(r['dtau']),
                linewidth=2.5, color=cmap(norm(hii_dim)),
                marker='o', markersize=3, alpha=0.8, label=f'{hii_dim}³')
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Optical Depth Element $d\tau$')
    ax.invert_xaxis()
    ax.legend(loc='best', ncol=1, title='Resolution')

save_pdf_png(plot_dtau_vs_z, plot_dir, "dtau_vs_z_resolution_all",
             title=r'Optical Depth Element $d\tau$ vs Redshift')
print("✓ Saved: dtau_vs_z_resolution_all")

# =============================================================================
# PLOT: Total τ vs Resolution
# =============================================================================

dim_vals       = sorted(tau_results.keys())
tau_total_vals = [float(np.asarray(tau_results[d]['tau_total'])) for d in dim_vals]
cell_sizes_kpc = [(BOX_LEN_FIXED / d) * 1000 for d in dim_vals]

def plot_tau_total_vs_resolution(ax):
    ax.plot(dim_vals, tau_total_vals, 'o-', linewidth=3, markersize=10,
            color='darkblue', label=r'Total $\tau$')
    ax.set_xlabel(r'HII\_DIM (Resolution)')
    ax.set_ylabel(r'Total Optical Depth $\tau$')
    ax.legend(loc='best')
    ax.text(0.05, 0.95, rf'BOX\_LEN = {BOX_LEN_FIXED:.0f} Mpc',
            transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

save_pdf_png(plot_tau_total_vs_resolution, plot_dir, "tau_total_vs_resolution",
             title='Total Optical Depth vs Resolution')
print("✓ Saved: tau_total_vs_resolution")

# =============================================================================
# PLOT: Total τ vs Cell Size
# =============================================================================

def plot_tau_total_vs_cellsize(ax):
    ax.plot(cell_sizes_kpc, tau_total_vals, 's-', linewidth=3, markersize=10,
            color='darkred', label=r'Total $\tau$')
    ax.set_xlabel(r'Cell Size [kpc]')
    ax.set_ylabel(r'Total Optical Depth $\tau$')
    ax.legend(loc='best')
    ax.invert_xaxis()
    ax.text(0.05, 0.95, rf'BOX\_LEN = {BOX_LEN_FIXED:.0f} Mpc',
            transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

save_pdf_png(plot_tau_total_vs_cellsize, plot_dir, "tau_total_vs_cellsize",
             title='Total Optical Depth vs Cell Size (Should be constant)')
print("✓ Saved: tau_total_vs_cellsize")

# =============================================================================
# PLOT: Comoving Distance s vs z
# =============================================================================

def plot_s_vs_z(ax):
    for hii_dim in sorted(tau_results.keys()):
        r = tau_results[hii_dim]
        ax.plot(np.asarray(r['red_axis']), np.asarray(r['s']),
                linewidth=2.0, color=cmap(norm(hii_dim)), alpha=0.7,
                label=f'{hii_dim}³')
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Comoving Distance $s$ [Mpc]')
    ax.invert_xaxis()
    ax.legend(loc='best', title='Resolution')

save_pdf_png(plot_s_vs_z, plot_dir, "s_vs_z_resolution",
             title='Comoving Distance vs Redshift (Independent of Resolution)')
print("✓ Saved: s_vs_z_resolution")

print("\n" + "="*70)
print("REIONIZATION HISTORY AND OPTICAL DEPTH COMPLETE!")
print("="*70)

# %%
# =============================================================================
# CELL 5: Compute kSZ Integrand with Visibility Function — Skewed LOS
# kSZ integrand = (1 + δ) × x_e × v_z / c × e^(-τ(z))
# Stores both unrotated (θ=0°) and rotated (θ=angle_deg°) skewers per resolution
# Full box face sampling: Nlos_dim = Ndim² for every resolution
# Results cached to .npz for fast reload
# =============================================================================

print("\n" + "="*70)
print("COMPUTING kSZ INTEGRAND WITH VISIBILITY FUNCTION (SKEWED LOS)")
print("="*70)

c_Mpc_s = 299792.458 / 3.08567758e19   # Speed of light in Mpc/s
print(f"Speed of light: c = {c_Mpc_s:.6e} Mpc/s")

kSZ_results = {}

for hii_dim in HII_DIM_VALUES:
    if hii_dim not in lightcones or lightcones[hii_dim] is None:
        continue
    if hii_dim not in tau_results:
        continue

    cell_size_mpc = BOX_LEN_FIXED / hii_dim
    Ndim          = int(hii_dim)

    print(f"\n{'='*70}")
    print(f"HII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc)")
    print(f"{'='*70}")

    # ==========================================================================
    # CHECK CACHE FIRST
    # ==========================================================================

    cache_file = (f"{main_cache_dir}/DIM{hii_dim}_BOX{BOX_LEN_FIXED:.0f}"
                  f"/kSZ_integrand_skewed_angle{angle_deg}.npz")

    if os.path.exists(cache_file):
        print(f"  ✓ Loading cached skewer integrand...")
        print(f"    {cache_file}")
        cached = np.load(cache_file)

        res      = tau_results[hii_dim]
        pos_axis = np.asarray(res['pos_axis'])

        kSZ_results[hii_dim] = {
            # --- unrotated ---
            'kSZ_integrand_unrot' : cached['kSZ_integrand_unrot'],
            # --- rotated ---
            'kSZ_integrand_rot'   : cached['kSZ_integrand_rot'],
            # --- shared ---
            'visibility'          : cached['visibility'],
            'tau_at_lightcone'    : cached['tau_at_lightcone'],
            'red_axis'            : cached['red_axis'],
            'pos_axis'            : cached['pos_axis'],
            'ds'                  : cached['ds'],
            's_axis'              : cached['s_axis'],
            'ind_z'               : cached['ind_z'],
            'Nlos_dim'            : Ndim * Ndim,
            'Ndim'                : Ndim,
            'hii_dim'             : hii_dim,
            'delta_z'             : float(pos_axis[1] - pos_axis[0]),
            'cell_size'           : BOX_LEN_FIXED / Ndim,
            'LOS_ind'             : np.array([[i, j]
                                    for i in range(Ndim)
                                    for j in range(Ndim)]),
        }

        r     = kSZ_results[hii_dim]
        rms_u = np.sqrt(np.mean(r['kSZ_integrand_unrot']**2))
        rms_r = np.sqrt(np.mean(r['kSZ_integrand_rot']**2))
        print(f"  Unrotated — RMS={rms_u:.4e}")
        print(f"  Rotated   — RMS={rms_r:.4e}")
        continue

    # ==========================================================================
    # NO CACHE — RUN SKEWER EXTRACTION
    # ==========================================================================

    lightcone = lightcones[hii_dim]
    res       = tau_results[hii_dim]

    red_axis  = np.asarray(res['red_axis'])
    z_mid     = np.asarray(res['z_mid'])
    tau       = np.asarray(res['tau'])
    pos_axis  = np.asarray(res['pos_axis'])
    ds        = np.asarray(res['ds'])

    Nbins     = len(red_axis)
    cell_size = BOX_LEN_FIXED / Ndim
    delta_z   = float(pos_axis[1] - pos_axis[0])
    s_axis    = pos_axis - pos_axis[0]

    # Extract 3D fields
    red_axis_full = np.asarray(lightcone.lightcone_redshifts)
    ind_z = np.where(red_axis_full <= z_max)[0]

    _Delta_3d = np.array(lightcone.density[:, :, ind_z],  dtype=np.float32) + 1.0
    _xHI_3d   = np.array(lightcone.xH_box[:, :, ind_z],   dtype=np.float32)
    _vel_3d   = np.array(lightcone.velocity[:, :, ind_z],  dtype=np.float32)/67.4

    print(f"3D field shape: {_Delta_3d.shape}  (Ndim x Ndim x Nbins)")

    # Build LOS grid
    Nlos_dim = Ndim * Ndim
    LOS_ind  = np.array([[i, j] for i in range(Ndim) for j in range(Ndim)])
    print(f"LOS grid: {Ndim}×{Ndim} = {Nlos_dim} skewers (full box face)")

    # Skewer extraction functions
    def _periodic(n, ngrid):
        return int(round(float(n))) % int(ngrid)

    def rotated_skewer(field, x_start, y_idx):
        """Rotated LOS with linear z-interpolation."""
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

    # Allocate arrays
    density_unrot  = np.zeros((Nlos_dim, Nbins), dtype=np.float32)
    xH_box_unrot   = np.zeros((Nlos_dim, Nbins), dtype=np.float32)
    velocity_unrot = np.zeros((Nlos_dim, Nbins), dtype=np.float32)
    density_rot    = np.zeros((Nlos_dim, Nbins), dtype=np.float32)
    xH_box_rot     = np.zeros((Nlos_dim, Nbins), dtype=np.float32)
    velocity_rot   = np.zeros((Nlos_dim, Nbins), dtype=np.float32)

    for k, (x0, y0) in enumerate(LOS_ind):
        ix, iy = int(x0) % Ndim, int(y0) % Ndim

        density_unrot[k]  = _Delta_3d[ix, iy, :]
        xH_box_unrot[k]   = _xHI_3d[ix,  iy, :]
        velocity_unrot[k] = _vel_3d[ix,   iy, :]

        density_rot[k]    = rotated_skewer(_Delta_3d, int(x0), int(y0))
        xH_box_rot[k]     = rotated_skewer(_xHI_3d,   int(x0), int(y0))
        velocity_rot[k]   = rotated_skewer(_vel_3d,    int(x0), int(y0))

        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{Nlos_dim} skewers done", flush=True)

    print(f"  Extraction complete.")
    print(f"  Unrotated — <1+δ>={density_unrot.mean():.4f}  "
          f"<xHI>={xH_box_unrot.mean():.4f}  <v>={velocity_unrot.mean():.4e}")
    print(f"  Rotated   — <1+δ>={density_rot.mean():.4f}  "
          f"<xHI>={xH_box_rot.mean():.4f}  <v>={velocity_rot.mean():.4e}")

    # Visibility function
    tau_extended     = np.concatenate([[0], tau])
    tau_at_lightcone = np.interp(red_axis,
                                  np.concatenate([[red_axis[0]], z_mid]),
                                  tau_extended)
    visibility = np.exp(-tau_at_lightcone)

    print(f"τ range      : [{tau_at_lightcone.min():.6f}, {tau_at_lightcone.max():.6f}]")
    print(f"e^(-τ) range : [{visibility.min():.6f}, {visibility.max():.6f}]")

    # kSZ integrand
    x_e_unrot  = 1.0 - xH_box_unrot
    x_e_rot    = 1.0 - xH_box_rot
    vis_broad  = visibility[np.newaxis, :]

    kSZ_integrand_unrot = density_unrot * x_e_unrot * velocity_unrot / c_Mpc_s * vis_broad
    kSZ_integrand_rot   = density_rot   * x_e_rot   * velocity_rot   / c_Mpc_s * vis_broad

    print(f"\nkSZ integrand shape : {kSZ_integrand_unrot.shape}")
    print(f"  Unrotated — RMS={np.sqrt(np.mean(kSZ_integrand_unrot**2)):.4e}  "
          f"mean={kSZ_integrand_unrot.mean():.4e}")
    print(f"  Rotated   — RMS={np.sqrt(np.mean(kSZ_integrand_rot**2)):.4e}  "
          f"mean={kSZ_integrand_rot.mean():.4e}")

    # ==========================================================================
    # SAVE TO CACHE
    # ==========================================================================

    np.savez_compressed(
        cache_file,
        kSZ_integrand_unrot = kSZ_integrand_unrot,
        kSZ_integrand_rot   = kSZ_integrand_rot,
        visibility          = visibility,
        tau_at_lightcone    = tau_at_lightcone,
        red_axis            = red_axis,
        pos_axis            = pos_axis,
        ds                  = ds,
        s_axis              = s_axis,
        ind_z               = ind_z,
    )
    print(f"  ✓ Cached to {cache_file}")

    # Store in memory
    kSZ_results[hii_dim] = {
        # --- unrotated ---
        'density_unrot'       : density_unrot,
        'xH_box_unrot'        : xH_box_unrot,
        'velocity_unrot'      : velocity_unrot,
        'kSZ_integrand_unrot' : kSZ_integrand_unrot,
        # --- rotated ---
        'density_rot'         : density_rot,
        'xH_box_rot'          : xH_box_rot,
        'velocity_rot'        : velocity_rot,
        'kSZ_integrand_rot'   : kSZ_integrand_rot,
        # --- shared ---
        'visibility'          : visibility,
        'tau_at_lightcone'    : tau_at_lightcone,
        'red_axis'            : red_axis,
        'pos_axis'            : pos_axis,
        'ds'                  : ds,
        'ind_z'               : ind_z,
        'LOS_ind'             : LOS_ind,
        'Nlos_dim'            : Nlos_dim,
        'Ndim'                : Ndim,
        'hii_dim'             : hii_dim,
        's_axis'              : s_axis,
        'delta_z'             : delta_z,
        'cell_size'           : cell_size,
    }

print("\n" + "="*70)
print("kSZ INTEGRAND CALCULATION COMPLETE (UNROTATED + ROTATED SKEWERS)")
print(f"Computed for {len(kSZ_results)} HII_DIM values")
print("="*70)

# =============================================================================
# Summary Statistics Table
# =============================================================================

print("\n" + "="*70)
print("kSZ INTEGRAND SUMMARY STATISTICS")
print("="*70)
print(f"{'HII_DIM':<12} {'Ndim':<8} {'Nlos=Ndim²':<12} "
      f"{'RMS_unrot':<14} {'RMS_rot':<14} {'Cell[Mpc]':<12}")
print("-" * 75)

for hii_dim in sorted(kSZ_results.keys()):
    r     = kSZ_results[hii_dim]
    rms_u = np.sqrt(np.mean(r['kSZ_integrand_unrot']**2))
    rms_r = np.sqrt(np.mean(r['kSZ_integrand_rot']**2))
    print(f"{hii_dim:<12} {r['Ndim']:<8} {r['Nlos_dim']:<12} "
          f"{rms_u:<14.4e} {rms_r:<14.4e} {BOX_LEN_FIXED/hii_dim:<12.3f}")

# %%
# # =============================================================================
# # NOT FOR REPORTs
# # CELL 5B:PLOT: kSZ Integrand - All Resolutions Stacked
# # =============================================================================

# print("\n" + "="*70)
# print("GENERATING kSZ INTEGRAND PLOTS")
# print("="*70)

# # Stacked plots for all resolutions
# fig, axes = plt.subplots(len(HII_DIM_VALUES), 1, 
#                          figsize=(14, 4*len(HII_DIM_VALUES)), 
#                          constrained_layout=True)

# if len(HII_DIM_VALUES) == 1:
#     axes = [axes]

# # Rainbow colormap for labels
# cmap = mpl.cm.rainbow
# norm = mpl.colors.Normalize(vmin=HII_DIM_VALUES.min(), vmax=HII_DIM_VALUES.max())

# for idx, (hii_dim, ax) in enumerate(zip(HII_DIM_VALUES, axes)):
#     if hii_dim not in kSZ_results:
#         ax.text(0.5, 0.5, f'HII_DIM = {hii_dim}\nNo data', 
#                ha='center', va='center',
#                transform=ax.transAxes, fontsize=16, color='red')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         continue
        
#     lightcone = lightcones[hii_dim]
    
#     plotting.lightcone_sliceplot(lightcone, 'kSZ_integrand', ax=ax, fig=fig)
    
#     # Change colormap to seismic (diverging colormap for positive/negative)
#     im = ax.images[0]
#     im.set_cmap('seismic')
    
#     # Set symmetric color limits
#     kSZ_data = kSZ_results[hii_dim]['kSZ_integrand']
#     vmax = np.percentile(np.abs(kSZ_data), 99)
#     im.set_clim(-vmax, vmax)
    
#     # Get rainbow color for label
#     color = cmap(norm(hii_dim))
#     cell_size_mpc = BOX_LEN_FIXED / hii_dim
#     cell_size_kpc = cell_size_mpc * 1000
    
#     # Add label with resolution info
#     ax.text(0.02, 0.98, 
#            f'DIM = {hii_dim}³  |  Cell = {cell_size_mpc:.3f} Mpc ({cell_size_kpc:.1f} kpc)', 
#            transform=ax.transAxes, 
#            fontsize=13, fontweight='bold',
#            verticalalignment='top',
#            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, 
#                     edgecolor='black', linewidth=2))
    
#     # Add resolution on right
#     ax.text(0.98, 0.98, 
#            f'{hii_dim}³', 
#            transform=ax.transAxes, fontsize=14, fontweight='bold',
#            verticalalignment='top', horizontalalignment='right',
#            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# # Save
# plot_name = "kSZ_integrand_with_visibility_resolution_stack"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

# fig.suptitle(r'kSZ Integrand: $(1+\delta) \times x_e \times v_z/c \times e^{-\tau(z)}$ (BOX_LEN=' + f'{BOX_LEN_FIXED:.0f} Mpc)', 
#              fontsize=20, fontweight='bold', y=0.995)
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

# print(f"✓ Saved: {plot_name}")
# plt.close(fig)

# # =============================================================================
# # PLOT: Visibility Function e^(-τ) vs z (All Resolutions - Rainbow)
# # =============================================================================

# fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

# for hii_dim in sorted(kSZ_results.keys()):
#     results = kSZ_results[hii_dim]
#     red_axis_plot = results['red_axis']
#     visibility_plot = results['visibility']
    
#     color = cmap(norm(hii_dim))
    
#     ax.plot(red_axis_plot, visibility_plot, 
#            linewidth=2.5, color=color,
#            marker='o', markersize=3, alpha=0.8,
#            label=f'{hii_dim}³')

# ax.set_xlabel('Redshift $z$', fontsize=20)
# ax.set_ylabel(r'Visibility Function $e^{-\tau(z)}$', fontsize=20)
# ax.set_ylim(0, 1.05)
# ax.invert_xaxis()
# ax.legend(fontsize=12, loc='best', ncol=1, title='Resolution')
# #ax.grid(True, alpha=0.3, linestyle='--')

# # Add colorbar
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, pad=0.02)
# cbar.set_label(r'HII\_DIM', fontsize=16)

# plot_name = "visibility_function_vs_z_resolution_all"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

# ax.set_title(r'Visibility Function $e^{-\tau(z)}$ vs Redshift', 
#             fontsize=20, fontweight='bold')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

# print(f"✓ Saved: {plot_name}")
# plt.close(fig)


# print("\n" + "="*70)
# print("kSZ INTEGRAND PLOTTING COMPLETE!")
# print("="*70)

# %%
# =============================================================================
# CELL 6: Compute Line-of-Sight Integrated kSZ Maps for All Resolutions
# Three cases: Original (full 3D), Unrotated skewers, Rotated skewers
# kSZ(z=5) = ∫ n_e0 σ_T (1/a²) (1+δ) x_e v_z/c e^(-τ) ds
# =============================================================================

print("\n" + "="*70)
print("LINE-OF-SIGHT kSZ MAP INTEGRATION — THREE CASES")
print("="*70)

# Physical constants in CGS
c_cm_s        = 3.0e10       # cm/s
sigma_T_cm2   = 6.6525e-25   # cm²
n_e0_cm3      = 2.06e-7      # cm⁻³
Mpc_to_cm     = 3.0857e24    # cm/Mpc
prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s   # [s⁻¹]
c_Mpc_s       = 299792.458 / 3.08567758e19         # Mpc/s

print(f"Prefactor n_e0 × σ_T × c = {prefactor_cgs:.4e} s⁻¹")

kSZ_map_results = {}

for hii_dim in HII_DIM_VALUES:
    if hii_dim not in kSZ_results or hii_dim not in tau_results:
        continue

    cell_size_mpc = BOX_LEN_FIXED / hii_dim

    print(f"\n{'='*70}")
    print(f"HII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc)")
    print(f"{'='*70}")

    lightcone = lightcones[hii_dim]
    res       = kSZ_results[hii_dim]

    red_axis  = np.asarray(res['red_axis'])
    ind_z     = res['ind_z']
    ds_Mpc    = np.asarray(res['ds'])
    ds_cm     = ds_Mpc * Mpc_to_cm

    # Scale factor midpoints — shared across all three cases
    a             = 1.0 / (1.0 + red_axis)
    a_squared     = a**2
    a_squared_mid = 0.5 * (a_squared[:-1] + a_squared[1:])   # (Nbins-1,)

    # Weight per slice: prefactor × ds/c / a²
    weight = (prefactor_cgs / a_squared_mid) * (ds_cm / c_cm_s)   # (Nbins-1,)

    # ===========================================================================
    # CASE 1: Original — full 3D LOS integral  (Ndim, Ndim, Nbins)
    # ===========================================================================

    density_orig = np.array(lightcone.density[:, :, ind_z],  dtype=np.float64) + 1.0
    x_e_orig     = 1.0 - np.array(lightcone.xH_box[:, :, ind_z],  dtype=np.float64)
    vel_orig     = np.array(lightcone.velocity[:, :, ind_z], dtype=np.float64)/67.4
    vis_orig     = res['visibility']   # (Nbins,)

    integrand_orig     = (density_orig * x_e_orig *
                          (vel_orig / c_Mpc_s) *
                          vis_orig[None, None, :])
    integrand_orig_mid = 0.5 * (integrand_orig[:, :, :-1] +
                                 integrand_orig[:, :, 1:])   # (Ndim, Ndim, Nbins-1)

    kSZ_map_orig = np.sum(weight[None, None, :] * integrand_orig_mid, axis=2)

    print(f"Original   — shape {kSZ_map_orig.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map_orig**2)):.4e}")

    # ===========================================================================
    # CASE 2: Unrotated skewers  (Nlos_dim, Nbins)
    # ===========================================================================

    ki_unrot     = np.asarray(res['kSZ_integrand_unrot'], dtype=np.float64)
    ki_unrot_mid = 0.5 * (ki_unrot[:, :-1] + ki_unrot[:, 1:])   # (Nlos, Nbins-1)

    kSZ_1d_unrot  = np.sum(weight[None, :] * ki_unrot_mid, axis=1)   # (Nlos,)

    # Reshape to 2D map — Nlos_dim = Ndim² so reshape is exact
    Nlos_dim      = res['Nlos_dim']
    npix_map      = int(np.floor(np.sqrt(Nlos_dim)))
    n_use         = npix_map * npix_map
    kSZ_map_unrot = kSZ_1d_unrot[:n_use].reshape(npix_map, npix_map)

    print(f"Unrotated  — 1D shape {kSZ_1d_unrot.shape}  "
          f"map {kSZ_map_unrot.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map_unrot**2)):.4e}")

    # ===========================================================================
    # CASE 3: Rotated skewers  (Nlos_dim, Nbins)
    # ===========================================================================

    ki_rot     = np.asarray(res['kSZ_integrand_rot'], dtype=np.float64)
    ki_rot_mid = 0.5 * (ki_rot[:, :-1] + ki_rot[:, 1:])   # (Nlos, Nbins-1)

    kSZ_1d_rot  = np.sum(weight[None, :] * ki_rot_mid, axis=1)   # (Nlos,)
    kSZ_map_rot = kSZ_1d_rot[:n_use].reshape(npix_map, npix_map)

    print(f"Rotated    — 1D shape {kSZ_1d_rot.shape}  "
          f"map {kSZ_map_rot.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map_rot**2)):.4e}")

    # ===========================================================================
    # Store all three
    # ===========================================================================

    kSZ_map_results[hii_dim] = {
        # original
        'kSZ_map_orig'  : kSZ_map_orig,
        # unrotated skewers
        'kSZ_1d_unrot'  : kSZ_1d_unrot,
        'kSZ_map_unrot' : kSZ_map_unrot,
        # rotated skewers
        'kSZ_1d_rot'    : kSZ_1d_rot,
        'kSZ_map_rot'   : kSZ_map_rot,
        # metadata
        'npix_map'      : npix_map,
        'n_use'         : n_use,
        'pix_size_Mpc'  : BOX_LEN_FIXED / npix_map,
        'pix_size_orig' : BOX_LEN_FIXED / hii_dim,
        'hii_dim'       : hii_dim,
        'Nlos_dim'      : Nlos_dim,
    }

    lightcone.kSZ_map = kSZ_map_orig

print("\n" + "="*70)
print("kSZ MAP INTEGRATION COMPLETE — THREE CASES")
print(f"Computed maps for {len(kSZ_map_results)} HII_DIM values")
print("="*70)

# =============================================================================
# Summary Statistics Table
# =============================================================================

print("\n" + "="*70)
print("kSZ MAP SUMMARY STATISTICS")
print("="*70)
print(f"{'HII_DIM':<10} {'Cell[Mpc]':<10} {'npix_skew':<12} "
      f"{'RMS_orig':<14} {'RMS_unrot':<14} {'RMS_rot':<14}")
print("-" * 80)

for hii_dim in sorted(kSZ_map_results.keys()):
    r     = kSZ_map_results[hii_dim]
    rms_o = np.sqrt(np.mean(r['kSZ_map_orig']**2))
    rms_u = np.sqrt(np.mean(r['kSZ_map_unrot']**2))
    rms_r = np.sqrt(np.mean(r['kSZ_map_rot']**2))
    print(f"{hii_dim:<10} {BOX_LEN_FIXED/hii_dim:<10.3f} "
          f"{r['npix_map']:<12} "
          f"{rms_o:<14.4e} {rms_u:<14.4e} {rms_r:<14.4e}")

# %%
# # =============================================================================
# #NOT FOR REPORTs
# #  CELL 6B:PLOT: kSZ Maps - All Resolutions Side-by-Side
# # =============================================================================

# print("\n" + "="*70)
# print("GENERATING kSZ MAP PLOTS")
# print("="*70)

# fig, axes = plt.subplots(1, len(HII_DIM_VALUES), 
#                          figsize=(5*len(HII_DIM_VALUES), 5.5), 
#                          constrained_layout=True)

# if len(HII_DIM_VALUES) == 1:
#     axes = [axes]

# # Rainbow colormap for labels
# cmap = mpl.cm.rainbow
# norm = mpl.colors.Normalize(vmin=HII_DIM_VALUES.min(), vmax=HII_DIM_VALUES.max())

# # Find global vmax for consistent color scale
# vmax_global = 0
# for hii_dim in kSZ_map_results.keys():
#     kSZ_map = kSZ_map_results[hii_dim]['kSZ_map']
#     vmax_global = max(vmax_global, np.percentile(np.abs(kSZ_map), 99))

# for idx, (hii_dim, ax) in enumerate(zip(HII_DIM_VALUES, axes)):
#     if hii_dim not in kSZ_map_results:
#         ax.text(0.5, 0.5, 'No data', ha='center', va='center',
#                transform=ax.transAxes, fontsize=16)
#         ax.set_title(f'DIM={hii_dim}', fontsize=14)
#         continue
        
#     kSZ_map = kSZ_map_results[hii_dim]['kSZ_map']
    
#     # Plot the kSZ map (all same physical size: 800 Mpc)
#     im = ax.imshow(kSZ_map.T,
#                    cmap='seismic',
#                    origin='lower',
#                    extent=[0, BOX_LEN_FIXED, 0, BOX_LEN_FIXED],
#                    aspect='equal',
#                    vmin=-vmax_global,
#                    vmax=vmax_global)
    
#     # Colorbar
#     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.ax.tick_params(labelsize=10)
    
#     # Labels
#     ax.set_xlabel('x [Mpc]', fontsize=11)
#     ax.set_ylabel('y [Mpc]', fontsize=11)
    
#     # Get rainbow color for title
#     color_label = cmap(norm(hii_dim))
#     cell_size_mpc = BOX_LEN_FIXED / hii_dim
#     cell_size_kpc = cell_size_mpc * 1000
    
#     ax.set_title(f'DIM={hii_dim}³\nCell={cell_size_mpc:.3f} Mpc\n({cell_size_kpc:.1f} kpc)', 
#                 fontsize=11, fontweight='bold',
#                 bbox=dict(boxstyle='round', facecolor=color_label, alpha=0.6, 
#                          edgecolor='black', linewidth=1.5))

# # Overall title
# fig.suptitle(r'kSZ Maps at $z=5$ (line-of-sight integrated, BOX_LEN=' + f'{BOX_LEN_FIXED:.0f} Mpc)', 
#              fontsize=18, fontweight='bold')

# # Save
# plot_name = "kSZ_maps_z5_resolution_all"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

# print(f"✓ Saved: {plot_name}")
# plt.close(fig)

# # =============================================================================
# # PLOT: kSZ Maps - Stacked Vertically
# # =============================================================================

# fig, axes = plt.subplots(len(HII_DIM_VALUES), 1, 
#                          figsize=(8, 6*len(HII_DIM_VALUES)), 
#                          constrained_layout=True)

# if len(HII_DIM_VALUES) == 1:
#     axes = [axes]

# for idx, (hii_dim, ax) in enumerate(zip(HII_DIM_VALUES, axes)):
#     if hii_dim not in kSZ_map_results:
#         ax.text(0.5, 0.5, f'DIM={hii_dim}\nNo data', 
#                ha='center', va='center',
#                transform=ax.transAxes, fontsize=16, color='red')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         continue
        
#     kSZ_map = kSZ_map_results[hii_dim]['kSZ_map']
    
#     # Plot the kSZ map
#     im = ax.imshow(kSZ_map.T,
#                    cmap='seismic',
#                    origin='lower',
#                    extent=[0, BOX_LEN_FIXED, 0, BOX_LEN_FIXED],
#                    aspect='equal',
#                    vmin=-vmax_global,
#                    vmax=vmax_global)
    
#     # Colorbar
#     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.ax.tick_params(labelsize=10)
#     cbar.set_label('kSZ (dimensionless)', fontsize=11)
    
#     # Labels
#     ax.set_xlabel('x [Mpc]', fontsize=12)
#     ax.set_ylabel('y [Mpc]', fontsize=12)
    
#     # Get rainbow color for label
#     color_label = cmap(norm(hii_dim))
#     cell_size_mpc = BOX_LEN_FIXED / hii_dim
#     cell_size_kpc = cell_size_mpc * 1000
    
#     # Add label
#     ax.text(0.02, 0.98, 
#            f'DIM = {hii_dim}³  |  Cell = {cell_size_mpc:.3f} Mpc ({cell_size_kpc:.1f} kpc)', 
#            transform=ax.transAxes, fontsize=12, fontweight='bold',
#            verticalalignment='top',
#            bbox=dict(boxstyle='round', facecolor=color_label, alpha=0.7, 
#                     edgecolor='black', linewidth=2))

# # Overall title
# fig.suptitle(r'kSZ Maps at $z=5$ - Resolution Comparison', 
#              fontsize=20, fontweight='bold', y=0.995)

# # Save
# plot_name = "kSZ_maps_z5_resolution_stack"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

# print(f"✓ Saved: {plot_name}")
# plt.close(fig)

# # =============================================================================
# # PLOT: kSZ Map Histograms - All Resolutions Overlay
# # =============================================================================

# fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

# for hii_dim in sorted(kSZ_map_results.keys()):
#     kSZ_map = kSZ_map_results[hii_dim]['kSZ_map']
#     color = cmap(norm(hii_dim))
    
#     ax.hist(kSZ_map.flatten(), bins=100, 
#            color=color, 
#            alpha=0.5, 
#            edgecolor='black',
#            linewidth=0.5,
#            label=f'{hii_dim}³')

# ax.set_xlabel('kSZ Signal (dimensionless)', fontsize=20)
# ax.set_ylabel('Number of Pixels', fontsize=20)
# ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
# ax.legend(fontsize=14, loc='best', title='Resolution')
# #ax.grid(True, alpha=0.3, linestyle='--')

# # Add colorbar
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, pad=0.02)
# cbar.set_label(r'HII\_DIM', fontsize=16)

# # Save
# plot_name = "kSZ_map_histogram_resolution_all"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

# ax.set_title(f'kSZ Map Pixel Distribution (BOX_LEN={BOX_LEN_FIXED:.0f} Mpc)', 
#             fontsize=20, fontweight='bold')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

# print(f"✓ Saved: {plot_name}")
# plt.close(fig)

# # =============================================================================
# # PLOT: kSZ RMS vs Resolution
# # =============================================================================

# fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

# dim_vals = sorted(kSZ_map_results.keys())
# rms_vals = []
# std_vals = []
# cell_sizes_kpc = []

# for hii_dim in dim_vals:
#     kSZ_map = kSZ_map_results[hii_dim]['kSZ_map']
#     rms_vals.append(np.sqrt(np.mean(kSZ_map**2)))
#     std_vals.append(np.std(kSZ_map))
#     cell_sizes_kpc.append((BOX_LEN_FIXED / hii_dim) * 1000)

# ax.plot(dim_vals, rms_vals, 'o-', linewidth=3, markersize=10,
#        color='darkblue', label='RMS')
# ax.plot(dim_vals, std_vals, 's-', linewidth=3, markersize=10,
#        color='darkred', label='Std Dev')

# ax.set_xlabel(r'HII\_DIM (Resolution)', fontsize=20)
# ax.set_ylabel('kSZ Signal (dimensionless)', fontsize=20)
# ax.legend(loc='best', fontsize=16)
# #ax.grid(True, alpha=0.3, linestyle='--')

# # Add annotation
# ax.text(0.05, 0.95, f'BOX_LEN = {BOX_LEN_FIXED:.0f} Mpc',
#        transform=ax.transAxes, fontsize=14,
#        verticalalignment='top',
#        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# plot_name = "kSZ_rms_vs_resolution"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

# ax.set_title('kSZ Map RMS vs Resolution', 
#             fontsize=22, fontweight='bold')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

# print(f"✓ Saved: {plot_name}")
# plt.close(fig)

# # =============================================================================
# # PLOT: kSZ RMS vs Cell Size
# # =============================================================================

# fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

# ax.plot(cell_sizes_kpc, rms_vals, 'o-', linewidth=3, markersize=10,
#        color='darkblue', label='RMS')
# ax.plot(cell_sizes_kpc, std_vals, 's-', linewidth=3, markersize=10,
#        color='darkred', label='Std Dev')

# ax.set_xlabel(r'Cell Size [kpc]', fontsize=20)
# ax.set_ylabel('kSZ Signal (dimensionless)', fontsize=20)
# ax.legend(loc='best', fontsize=16)
# #ax.grid(True, alpha=0.3, linestyle='--')
# ax.invert_xaxis()  # Higher resolution (smaller cells) on right

# # Add annotation
# ax.text(0.05, 0.95, f'BOX_LEN = {BOX_LEN_FIXED:.0f} Mpc',
#        transform=ax.transAxes, fontsize=14,
#        verticalalignment='top',
#        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# plot_name = "kSZ_rms_vs_cellsize"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

# ax.set_title('kSZ Map RMS vs Cell Size', 
#             fontsize=22, fontweight='bold')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

# print(f"✓ Saved: {plot_name}")
# plt.close(fig)

# print("\n" + "="*70)
# print("kSZ MAP GENERATION COMPLETE!")
# print("="*70)

# %%
# =============================================================================
# CELL 7: Compute kSZ Power Spectrum - P(k), C_ℓ, and D_ℓ for All Resolutions
# THREE CASES: Original (full 3D) / Unrotated skewers / Rotated skewers
# WITH ERROR BUDGET (Sample + Cosmic Variance)
# =============================================================================

print("\n" + "="*70)
print("COMPUTING kSZ POWER SPECTRA WITH ERROR BUDGET — THREE CASES")
print("="*70)

# CMB temperature conversion
T_CMB_0_K    = 2.725
z_obs        = 5.0
T_CMB_z5_uK  = T_CMB_0_K * 1e6   # μK

# Angular diameter / comoving distance
D_A_Mpc          = 1300
chi_comoving_Mpc  = D_A_Mpc * (1 + z_obs)   # 7800 Mpc

print(f"T_CMB(z=0) = {T_CMB_0_K:.3f} K  →  {T_CMB_z5_uK:.2f} μK")
print(f"χ(z=5)     = {chi_comoving_Mpc:.1f} Mpc")

# =============================================================================
# Helper: compute P(k), errors, ell, Cl, Dl from a 2D kSZ map
# =============================================================================

def compute_ps(ksz_map_2d, box_size_Mpc, n_kbins=35):
    """
    Returns dict with k_centers, P1d, errors, ell, Cl, Dl_uK2 and their errors.
    Normalization: P(k) [Mpc²] = (pix_size/N)² |FFT(map - mean)|²
    """
    npix      = ksz_map_2d.shape[0]
    pix_size  = box_size_Mpc / npix
    m         = ksz_map_2d - ksz_map_2d.mean()

    fft_shift = np.fft.fftshift(np.fft.fft2(m))
    ps2d      = (pix_size / npix)**2 * np.abs(fft_shift)**2

    dk    = 2 * np.pi / (npix * pix_size)
    kx    = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    ky    = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), n_kbins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    P1d            = np.full(n_kbins, np.nan)
    P1d_err_sample = np.full(n_kbins, np.nan)
    n_modes        = np.zeros(n_kbins)

    for i in range(n_kbins):
        mask = (kgrid >= k_bins[i]) & (kgrid < k_bins[i+1])
        n_modes[i] = mask.sum()
        if n_modes[i] > 0:
            vals = ps2d[mask]
            P1d[i]            = vals.mean()
            P1d_err_sample[i] = vals.std() / np.sqrt(n_modes[i])

    # Cosmic variance
    k_volume       = (box_size_Mpc / (2 * np.pi))**2
    n_modes_cosmic = 2 * np.pi * k_centers * k_volume * (k_bins[1:] - k_bins[:-1])
    cv_frac        = np.where(n_modes_cosmic > 0,
                              1.0 / np.sqrt(n_modes_cosmic), np.nan)
    P1d_err_cosmic = P1d * cv_frac
    P1d_err_total  = np.sqrt(P1d_err_sample**2 + P1d_err_cosmic**2)

    # ell, Cl, Dl
    ell    = k_centers * chi_comoving_Mpc / 0.67
    Cl     = P1d * 0.67**2 / D_A_Mpc**2
    Dl     = ell * (ell + 1) * Cl / (2 * np.pi)
    Dl_uK2 = Dl * T_CMB_z5_uK**2

    def _prop_err(P_err):
        Cl_e  = P_err * 0.67**2 / D_A_Mpc**2
        Dl_e  = ell * (ell + 1) * Cl_e / (2 * np.pi)
        return Dl_e * T_CMB_z5_uK**2

    return dict(
        k_centers         = k_centers,
        k_bins            = k_bins,
        P1d               = P1d,
        P1d_err_sample    = P1d_err_sample,
        P1d_err_cosmic    = P1d_err_cosmic,
        P1d_err_total     = P1d_err_total,
        n_modes           = n_modes,
        n_modes_cosmic    = n_modes_cosmic,
        ell               = ell,
        Cl                = Cl,
        Dl_uK2            = Dl_uK2,
        Dl_uK2_err_sample = _prop_err(P1d_err_sample),
        Dl_uK2_err_cosmic = _prop_err(P1d_err_cosmic),
        Dl_uK2_err_total  = _prop_err(P1d_err_total),
        pix_size          = pix_size,
        npix              = npix,
        dk                = dk,
    )

# =============================================================================
# Main loop — compute PS for all three cases per resolution
# =============================================================================

power_spectrum_results = {}

cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=HII_DIM_VALUES.min(), vmax=HII_DIM_VALUES.max())

for hii_dim in HII_DIM_VALUES:
    if hii_dim not in kSZ_map_results:
        continue

    cell_size_mpc = BOX_LEN_FIXED / hii_dim
    r = kSZ_map_results[hii_dim]

    print(f"\n{'='*70}")
    print(f"HII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc)")
    print(f"{'='*70}")

    # --- Original (full box, pixel size = cell_size_mpc) ---
    ps_orig  = compute_ps(r['kSZ_map_orig'],  BOX_LEN_FIXED)

    # --- Unrotated skewers ---
    ps_unrot = compute_ps(r['kSZ_map_unrot'], BOX_LEN_FIXED)

    # --- Rotated skewers ---
    ps_rot   = compute_ps(r['kSZ_map_rot'],   BOX_LEN_FIXED)

    for tag, ps in [('Original',  ps_orig),
                    ('Unrotated', ps_unrot),
                    ('Rotated',   ps_rot)]:
        valid = ~np.isnan(ps['P1d']) & (ps['P1d'] > 0)
        print(f"  {tag:<12} — npix={ps['npix']}  "
              f"dk={ps['dk']:.5f} Mpc⁻¹  "
              f"valid bins={valid.sum()}")

    power_spectrum_results[hii_dim] = {
        'orig'   : ps_orig,
        'unrot'  : ps_unrot,
        'rot'    : ps_rot,
        'hii_dim': hii_dim,
    }

print("\n" + "="*70)
print("POWER SPECTRUM CALCULATION COMPLETE — THREE CASES")
print(f"Computed for {len(power_spectrum_results)} HII_DIM values")
print("="*70)

# =============================================================================
# PLOT 1: P(k) — three cases, rainbow by resolution
# =============================================================================

def plot_Pk_three_cases(ax):
    for hii_dim in sorted(power_spectrum_results.keys()):
        r     = power_spectrum_results[hii_dim]
        color = cmap(norm(hii_dim))

        for tag, ps, ls, mk in [
            ('orig',  r['orig'],  '-',  '^'),
            ('unrot', r['unrot'], '--', 'o'),
            ('rot',   r['rot'],   ':',  's'),
        ]:
            valid = ~np.isnan(ps['P1d']) & (ps['P1d'] > 0)
            label = f"{hii_dim}³ ({tag})" if hii_dim == sorted(power_spectrum_results.keys())[0] else None
            ax.errorbar(
                ps['k_centers'][valid], ps['P1d'][valid],
                yerr=ps['P1d_err_total'][valid],
                color=color, ls=ls, marker=mk,
                markersize=4, linewidth=1.8, alpha=0.75,
                capsize=2, capthick=1, label=label
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax.set_ylabel(r'$P(k)$ [Mpc$^{2}$]')
    ax.legend(loc='best', fontsize=9)

save_pdf_png(plot_Pk_three_cases, plot_dir,
             "kSZ_Pk_three_cases_resolution_all",
             title=rf'kSZ $P(k)$ — Original / Unrotated / Rotated (Box={BOX_LEN_FIXED:.0f} Mpc)')
print("✓ Saved: kSZ_Pk_three_cases_resolution_all")

# =============================================================================
# PLOT 2: D_ℓ — three cases, rainbow by resolution  *** MAIN RESULT ***
# =============================================================================

def plot_Dl_three_cases(ax):
    import matplotlib.lines as mlines

    for hii_dim in sorted(power_spectrum_results.keys()):
        r     = power_spectrum_results[hii_dim]
        color = cmap(norm(hii_dim))
        lbl   = f'{hii_dim}³'

        for tag, ps, ls, mk in [
            ('orig',  r['orig'],  '-',  '^'),
            ('unrot', r['unrot'], '--', 'o'),
            ('rot',   r['rot'],   ':',  's'),
        ]:
            valid = (~np.isnan(ps['Dl_uK2']) & (ps['Dl_uK2'] > 0)
                     & (ps['ell'] > 10))
            label = lbl if tag == 'orig' else None
            ax.errorbar(
                ps['ell'][valid], ps['Dl_uK2'][valid],
                yerr=ps['Dl_uK2_err_total'][valid],
                color=color, ls=ls, marker=mk,
                markersize=4, linewidth=1.8, alpha=0.75,
                capsize=2, capthick=1, label=label
            )

    # Line-style legend
    orig_l  = mlines.Line2D([], [], color='gray', ls='-',  marker='^',
                             label='Original (full box)')
    unrot_l = mlines.Line2D([], [], color='gray', ls='--', marker='o',
                             label=r'Unrotated ($\theta=0°$)')
    rot_l   = mlines.Line2D([], [], color='gray', ls=':',  marker='s',
                             label=rf'Rotated ($\theta={angle_deg}°$)')

    # Color legend for resolution
    res_handles = [
        mlines.Line2D([], [], color=cmap(norm(h)), ls='-', linewidth=3,
                      label=f'{h}³')
        for h in sorted(power_spectrum_results.keys())
    ]

    leg1 = ax.legend(handles=[orig_l, unrot_l, rot_l],
                     loc='upper right', fontsize=9, title='Line style')
    ax.add_artist(leg1)
    ax.legend(handles=res_handles, loc='lower left', fontsize=9, title='Resolution')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')

save_pdf_png(plot_Dl_three_cases, plot_dir,
             "kSZ_Dl_three_cases_resolution_all",
             title=rf'kSZ $D_\ell$ — Original / Unrotated / Rotated (Box={BOX_LEN_FIXED:.0f} Mpc)')
print("✓ Saved: kSZ_Dl_three_cases_resolution_all  *** MAIN RESULT ***")

# =============================================================================
# Summary Table
# =============================================================================

print("\n" + "="*70)
print("POWER SPECTRUM SUMMARY — THREE CASES")
print("="*70)
print(f"{'HII_DIM':<10} {'Cell[Mpc]':<10} {'Case':<12} "
      f"{'ell_min':<10} {'ell_max':<10} {'Peak_Dl[μK²]':<16} {'Err[%]':<8}")
print("-" * 80)

for hii_dim in sorted(power_spectrum_results.keys()):
    r    = power_spectrum_results[hii_dim]
    cell = BOX_LEN_FIXED / hii_dim

    for tag, ps in [('Original',  r['orig']),
                    ('Unrotated', r['unrot']),
                    ('Rotated',   r['rot'])]:
        valid = (~np.isnan(ps['Dl_uK2']) & (ps['Dl_uK2'] > 0)
                 & (ps['ell'] > 10))
        if valid.sum() == 0:
            continue
        peak_idx = np.argmax(ps['Dl_uK2'][valid])
        peak_dl  = ps['Dl_uK2'][valid][peak_idx]
        peak_err = ps['Dl_uK2_err_total'][valid][peak_idx]
        print(f"{hii_dim:<10} {cell:<10.3f} {tag:<12} "
              f"{ps['ell'][valid].min():<10.0f} "
              f"{ps['ell'][valid].max():<10.0f} "
              f"{peak_dl:<16.4e} "
              f"{peak_err/peak_dl*100:<8.1f}")

print("\n" + "="*70)
print("ALL kSZ POWER SPECTRUM ANALYSIS COMPLETE!")
print("="*70)

# %%



