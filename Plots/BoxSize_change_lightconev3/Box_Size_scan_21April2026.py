# %%
# =============================================================================
# CELL 1: Imports and Setup
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import py21cmfast as p21c
from py21cmfast import plotting
import os
from datetime import datetime

print(f"py21cmfast version: {p21c.__version__}")

# --- Skewed LOS parameters ---
angle_deg = 10    # Rotation angle in degrees
Nlos      = None  # None → full Ndim² box face per box (recommended for convergence)

# =============================================================================
# CELL 1a: Create Output Directory for Plots
# =============================================================================

plot_dir = "BOX_SIZE_SCAN_15April2026/plots"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
else:
    print(f"Directory already exists: {plot_dir}")

print(f"All plots will be saved to: {os.path.abspath(plot_dir)}")

# =============================================================================
# CELL 1c: Define Parameters
# =============================================================================

CELL_SIZE_MPC  = 800 / 128
BOX_LEN_VALUES = np.arange(200, 801, 200)
HII_DIM_VALUES = (BOX_LEN_VALUES / CELL_SIZE_MPC).astype(int)

z_min = 5.0
z_max = 20.0

print(f"\n=== PARAMETER SCAN SETUP ===")
print(f"Fixed CELL_SIZE = {CELL_SIZE_MPC:.3f} Mpc")
print(f"\nScanning BOX_LEN:")
print(f"  Number of values: {len(BOX_LEN_VALUES)}")
print(f"  Range: {BOX_LEN_VALUES.min():.0f} Mpc → {BOX_LEN_VALUES.max():.0f} Mpc")
print(f"  Step size: {BOX_LEN_VALUES[1] - BOX_LEN_VALUES[0]:.0f} Mpc")
print(f"  Values: {BOX_LEN_VALUES}")
print(f"\nCorresponding HII_DIM values: {HII_DIM_VALUES}")
print(f"\nTotal simulations: {len(BOX_LEN_VALUES)}")

print("\n=== RESOLUTION DETAILS ===")
print(f"{'BOX_LEN [Mpc]':<15} {'HII_DIM':<10} {'Cell Size [Mpc]':<20} {'Cell Size [kpc]':<15}")
print("-" * 70)
for box_len, hii_dim in zip(BOX_LEN_VALUES, HII_DIM_VALUES):
    cell_size_mpc = box_len / hii_dim
    cell_size_kpc = cell_size_mpc * 1000
    print(f"{box_len:<15.0f} {hii_dim:<10} {cell_size_mpc:<20.3f} {cell_size_kpc:<15.1f}")

# --- Skewed LOS geometry (box-independent) ---
angle_rad = np.deg2rad(angle_deg)
sin_a     = float(np.sin(angle_rad))
cos_a     = float(np.cos(angle_rad))

print(f"\n=== SKEWED LOS PARAMETERS ===")
print(f"  Rotation angle : {angle_deg}°")
print(f"  sin(θ)         : {sin_a:.4f}")
print(f"  cos(θ)         : {cos_a:.4f}")
print(f"  Nlos mode      : FULL BOX FACE (Ndim² per box)")
print(f"\n  Skewers and artefact suppression by box size:")
print(f"  {'BOX_LEN [Mpc]':<15} {'HII_DIM':<10} {'Nlos=Ndim²':<12} "
      f"{'k_box [Mpc⁻¹]':<16} {'k_box_rot [Mpc⁻¹]':<20} {'Suppression':<12}")
print(f"  {'-'*85}")
for box_len, hii_dim in zip(BOX_LEN_VALUES, HII_DIM_VALUES):
    k_box     = 2 * np.pi / box_len
    k_box_rot = 2 * np.pi / (box_len / sin_a)
    print(f"  {box_len:<15.0f} {hii_dim:<10} {hii_dim**2:<12} "
          f"{k_box:<16.5f} {k_box_rot:<20.5f} {1/sin_a:.1f}×")

# %%
# =============================================================================
# CELL 1d: Define Plotting Utilities
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib as mpl

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
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
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

def save_pdf_png(plot_func, plot_dir, plot_name, title=None):
    """Save plot as both PDF and PNG"""
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

def redshift_to_time(z, H0=67.4, Om0=0.315):
    """Convert redshift to time since Big Bang in Gyr"""
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    return cosmo.age(z).to(u.Gyr).value

def make_lightcone_plotter(
    lightcone,
    field,
    cmap=None,
    clim=None,
    cbar_label=None,
    labelsize=16,
    axlabelsize=20,
    user_params=None,
    add_time_axis=False,
):
    """Create a lightcone plotting function"""
    def _plot(ax):
        import numpy as np
        fig = ax.figure
        
        # Plot the lightcone
        plotting.lightcone_sliceplot(lightcone, field, ax=ax, fig=fig)
        im = ax.images[0]
        
        # Apply colormap and limits
        if cmap is not None:
            im.set_cmap(cmap)
        if clim is not None:
            im.set_clim(*clim)
        
        # Y-axis label
        if user_params is not None:
            ax.set_ylabel(r'y-axis [Mpc]', fontsize=axlabelsize) #[$h^{-1}$ cMpc]
        
        # Tick formatting
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.xaxis.label.set_size(axlabelsize)
        ax.yaxis.label.set_size(axlabelsize)
        
        # Add time axis on top if requested
        ax2 = None
        if add_time_axis:
            ax2 = ax.twiny()
            xlim = ax.get_xlim()
            z_min, z_max = xlim
            t_min = redshift_to_time(z_max)
            t_max = redshift_to_time(z_min)
            
            ax2.set_xlim(t_min, t_max)
            ax2.set_xlabel('Time Since Big Bang [Gyr]', fontsize=axlabelsize)
            
            # Smart tick spacing
            time_range = t_max - t_min
            if time_range < 2:
                tick_spacing = 0.2
            elif time_range < 5:
                tick_spacing = 0.5
            elif time_range < 10:
                tick_spacing = 1.0
            else:
                tick_spacing = 2.0
            
            first_tick = np.ceil(t_min / tick_spacing) * tick_spacing
            last_tick = np.floor(t_max / tick_spacing) * tick_spacing
            t_ticks = np.arange(first_tick, last_tick + tick_spacing/2, tick_spacing)
            
            if len(t_ticks) > 8:
                t_ticks = t_ticks[::2]
            
            ax2.set_xticks(t_ticks)
            ax2.set_xticklabels([f'{t:.1f}' for t in t_ticks])
            ax2.tick_params(axis='x', which='major', labelsize=labelsize, direction='in')
            ax2.tick_params(axis='x', which='minor', direction='in')
            ax2.minorticks_on()
        
        # Format colorbar
        for cax in fig.axes:
            if cax is ax:
                continue
            if ax2 is not None and cax is ax2:
                continue
            cax.tick_params(labelsize=labelsize)
            cax.xaxis.label.set_size(axlabelsize)
            if cbar_label is not None:
                cax.set_xlabel(cbar_label)
    
    return _plot

print("✓ Simple lightcone plotter loaded")



# %%
# =============================================================================
# CELL 2: Run Lightcone Simulations - BOX_LEN Scan
# =============================================================================

import time

print("\n" + "="*70)
print("RUNNING BOX_LEN SCAN (FIXED RESOLUTION)")
print("="*70)

# Create main cache directory
main_cache_dir = "BOX_SIZE_SCAN_15April2026/cache"
if not os.path.exists(main_cache_dir):
    os.makedirs(main_cache_dir)
    print(f"Created cache directory: {main_cache_dir}")
else:
    print(f"Cache directory exists: {main_cache_dir}")

# Dictionary to store lightcone results
lightcones = {}

# Track timing
scan_start_time = time.time()

for idx, (box_len, hii_dim) in enumerate(zip(BOX_LEN_VALUES, HII_DIM_VALUES)):
    sim_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"SIMULATION {idx+1}/{len(BOX_LEN_VALUES)}")
    print(f"BOX_LEN = {box_len:.0f} Mpc")
    print(f"HII_DIM = {hii_dim}")
    print(f"Cell size = {box_len/hii_dim:.3f} Mpc = {box_len/hii_dim*1000:.1f} kpc")
    print(f"{'='*70}")
    
    # Define user parameters for this box size
    user_params = p21c.UserParams(
        HII_DIM=hii_dim,
        BOX_LEN=box_len,
        USE_INTERPOLATION_TABLES=True,
        N_THREADS=32
    )
    
    print(f"Redshift range: z = {z_min} → {z_max}")
    print(f"Resolution: {user_params.HII_DIM}³ cells")
    
    # Create subdirectory for this box size
    cache_subdir = f"{main_cache_dir}/BOX{box_len:.0f}_DIM{hii_dim}"
    
    # ← NEW: Check if lightcone exists in cache
    lightcone_file = f"{cache_subdir}/LightCone_z{z_min:.2f}_z{z_max:.2f}.h5"
    
    if os.path.exists(lightcone_file):
        print(f"\n✓ Found cached lightcone: {lightcone_file}")
        print(f"  Loading from cache instead of recomputing...")
        try:
            # Load existing lightcone
            from py21cmfast import LightCone
            lightcone = LightCone.read(lightcone_file)
            lightcones[box_len] = lightcone
            
            sim_time = time.time() - sim_start_time
            print(f"  Load time: {sim_time:.2f} seconds")
            print(f"  Shape: {lightcone.brightness_temp.shape}")
            print(f"  Redshift range: [{lightcone.lightcone_redshifts.min():.2f}, "
                  f"{lightcone.lightcone_redshifts.max():.2f}]")
            
            # Skip to next iteration
            continue
            
        except Exception as e:
            print(f"  ✗ Failed to load cached lightcone: {e}")
            print(f"  Will recompute...")
    
    # Run lightcone simulation (only if not cached)
    try:
        lightcone = p21c.run_lightcone(
            redshift=z_min,
            max_redshift=z_max,
            lightcone_quantities=('brightness_temp', 'density', 'xH_box', 'velocity'),
            user_params=user_params,
            random_seed=37,
            direc=cache_subdir,
            write=True  # ← Make sure to save!
        )
        
        # Store the lightcone
        lightcones[box_len] = lightcone
        

        sim_time = time.time() - sim_start_time
        
        print(f"\n✓ Simulation complete!")
        print(f"  Time: {sim_time/60:.2f} minutes")
        print(f"  Cache: {cache_subdir}")
        print(f"  Shape: {lightcone.brightness_temp.shape}")
        print(f"  Redshift range: [{lightcone.lightcone_redshifts.min():.2f}, "
              f"{lightcone.lightcone_redshifts.max():.2f}]")
        
        # Quick reionization stats
        z_nodes = lightcone.node_redshifts[::-1]
        x_e_nodes = 1.0 - lightcone.global_xH[::-1]
        
        try:
            idx_10 = np.argmin(np.abs(x_e_nodes - 0.1))
            idx_50 = np.argmin(np.abs(x_e_nodes - 0.5))
            idx_90 = np.argmin(np.abs(x_e_nodes - 0.9))
            
            z_10 = z_nodes[idx_10]
            z_50 = z_nodes[idx_50]
            z_90 = z_nodes[idx_90]
            delta_z = z_10 - z_90
            
            print(f"  Quick stats:")
            print(f"    z(10% ionized) = {z_10:.2f}")
            print(f"    z(50% ionized) = {z_50:.2f}")
            print(f"    z(90% ionized) = {z_90:.2f}")
            print(f"    Δz (10%→90%) = {delta_z:.2f}")
        except:
            print(f"  Could not compute reionization stats")
        
        # Progress estimate
        
        elapsed = time.time() - scan_start_time
        avg_time_per_sim = elapsed / (idx + 1)
        remaining_sims = len(BOX_LEN_VALUES) - (idx + 1)
        eta_minutes = (remaining_sims * avg_time_per_sim) / 60
        
        print(f"\n  Progress: {idx+1}/{len(BOX_LEN_VALUES)} ({100*(idx+1)/len(BOX_LEN_VALUES):.1f}%)")
        print(f"  Average time per sim: {avg_time_per_sim/60:.2f} min")
        print(f"  ETA: {eta_minutes:.1f} minutes (~{eta_minutes/60:.2f} hours)")
        
    except Exception as e:
        print(f"\n✗ Simulation FAILED!")
        print(f"  Error: {e}")
        lightcones[box_len] = None

total_time = time.time() - scan_start_time

print(f"\n{'='*70}")
print("ALL SIMULATIONS COMPLETE")
print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
print(f"Successful simulations: {sum(1 for lc in lightcones.values() if lc is not None)}/{len(BOX_LEN_VALUES)}")
print("="*70)

# %%
# =============================================================================
# NOT FOR REPORTS
# CELL 2b: Plot Lightcones (All Box Sizes Stacked)
# =============================================================================

print("\n" + "="*70)
print("GENERATING STACKED LIGHTCONE PLOTS")
print("="*70)

cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=BOX_LEN_VALUES.min(), vmax=BOX_LEN_VALUES.max())

fields_to_plot = [
    ('brightness_temp', '21cm Brightness Temperature', 'EoR'),
    ('xH_box', 'Neutral Fraction (xHI)', 'viridis'),
    ('density', 'Overdensity δ', 'magma'),
    ('velocity', 'Line-of-Sight Velocity', 'RdBu_r')
]

for field_name, field_title, field_cmap in fields_to_plot:
    print(f"\nPlotting {field_name}...")

    fig, axes = plt.subplots(len(BOX_LEN_VALUES), 1,
                             figsize=(14, 4*len(BOX_LEN_VALUES)),
                             constrained_layout=True)
    if len(BOX_LEN_VALUES) == 1:
        axes = [axes]

    for idx, (box_len, hii_dim, ax) in enumerate(zip(BOX_LEN_VALUES, HII_DIM_VALUES, axes)):  # ← fixed
        if box_len not in lightcones or lightcones[box_len] is None:
            ax.text(0.5, 0.5, f'BOX_LEN = {box_len:.0f} Mpc\nSimulation Failed',
                   ha='center', va='center', fontsize=16, color='red',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lightcone = lightcones[box_len]
        plotting.lightcone_sliceplot(lightcone, field_name, ax=ax, fig=fig)
        im = ax.images[0]
        im.set_cmap(field_cmap)

        color = cmap(norm(box_len))
        cell_size_mpc = box_len / hii_dim          # ← fixed
        cell_size_kpc = cell_size_mpc * 1000

        ax.text(0.02, 0.98,
               f'BOX = {box_len:.0f} Mpc  |  Cell = {cell_size_mpc:.2f} Mpc ({cell_size_kpc:.1f} kpc)',
               transform=ax.transAxes, fontsize=13, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7,
                        edgecolor='black', linewidth=2))
        ax.text(0.98, 0.98,
               f'{box_len:.0f} Mpc',
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plot_name = f"{field_name}_lightcone_boxsize_stack"
    fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
    fig.suptitle(f'{field_title} - Box Size Scan (Fixed Cell = {CELL_SIZE_MPC:.2f} Mpc)',  # ← fixed
                fontsize=22, fontweight='bold', y=0.995)
    fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_name}")
    plt.close(fig)

print("\n✓ STACKED LIGHTCONE PLOTTING COMPLETE!")

# %%
# =============================================================================
# NOT FOR REPORTS
# CELL 3: 2D SLICES AT z = 8 (ALL BOX SIZES)
# =============================================================================

print("\n" + "="*70)
print("GENERATING 2D SLICES AT z = 8.0")
print("="*70)

target_z = 8.0

# Build a helper dict for quick hii_dim lookup by box_len
hii_dim_lookup = dict(zip(BOX_LEN_VALUES, HII_DIM_VALUES))

fields_info = [
    ('brightness_temp', '21cm Brightness Temperature [mK]', 'EoR'),
    ('xH_box', 'Neutral Fraction (xHI)', 'viridis'),
    ('density', 'Overdensity δ', 'magma'),
    ('velocity', 'Line-of-Sight Velocity [km/s]', 'RdBu_r'),
]

for field_name, field_label, cmap_name in fields_info:
    print(f"\nProcessing {field_name}...")

    fig, axes = plt.subplots(1, len(BOX_LEN_VALUES),
                             figsize=(5*len(BOX_LEN_VALUES), 5.5),
                             constrained_layout=True)
    if len(BOX_LEN_VALUES) == 1:
        axes = [axes]

    vmin_global = np.inf
    vmax_global = -np.inf
    slices_data = []

    for box_len in BOX_LEN_VALUES:
        if box_len not in lightcones or lightcones[box_len] is None:
            slices_data.append((None, None, box_len))
            continue

        lightcone = lightcones[box_len]
        z_values = lightcone.lightcone_redshifts
        closest_idx = np.argmin(np.abs(z_values - target_z))
        actual_z = z_values[closest_idx]

        field_data = getattr(lightcone, field_name)
        slice_2d = field_data[:, :, closest_idx]

        slices_data.append((slice_2d, actual_z, box_len))
        vmin_global = min(vmin_global, slice_2d.min())
        vmax_global = max(vmax_global, slice_2d.max())

    cmap_rainbow = mpl.cm.rainbow
    norm_rainbow = mpl.colors.Normalize(vmin=BOX_LEN_VALUES.min(), vmax=BOX_LEN_VALUES.max())

    for idx, (data_tuple, ax) in enumerate(zip(slices_data, axes)):
        slice_2d, actual_z, box_len = data_tuple

        if slice_2d is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'BOX={box_len:.0f} Mpc', fontsize=14)
            continue

        im = ax.imshow(slice_2d.T, origin='lower', cmap=cmap_name,
                      vmin=vmin_global, vmax=vmax_global,
                      extent=[0, box_len, 0, box_len], aspect='auto')

        ax.set_xlabel('Comoving Distance [Mpc]', fontsize=12)
        ax.set_ylabel('Comoving Distance [Mpc]', fontsize=12)

        color_label = cmap_rainbow(norm_rainbow(box_len))
        hii_dim = hii_dim_lookup[box_len]          # ← fixed
        cell_size_mpc = box_len / hii_dim           # ← fixed
        cell_size_kpc = cell_size_mpc * 1000

        ax.set_title(f'BOX={box_len:.0f} Mpc\nCell={cell_size_mpc:.2f} Mpc ({cell_size_kpc:.0f} kpc)\nz={actual_z:.2f}',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color_label, alpha=0.6,
                             edgecolor='black', linewidth=1.5))

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f'{field_label} at z ≈ {target_z} (Fixed Cell = {CELL_SIZE_MPC:.2f} Mpc)',  # ← fixed
                fontsize=18, fontweight='bold')

    plot_name = f"{field_name}_slice_z{int(target_z)}_boxsize_all"
    fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
    fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_name}")
    print(f"    Range: [{vmin_global:.3f}, {vmax_global:.3f}]")
    plt.close(fig)

print("\n✓ 2D SLICE PLOTTING COMPLETE!")

# =============================================================================
# CELL 3b: Print Statistics for 2D Slices
# =============================================================================

print("\n" + "="*70)
print("2D SLICE STATISTICS AT z = 8")
print("="*70)

for box_len in BOX_LEN_VALUES:
    if box_len not in lightcones or lightcones[box_len] is None:
        continue

    lightcone = lightcones[box_len]
    z_values = lightcone.lightcone_redshifts
    closest_idx = np.argmin(np.abs(z_values - target_z))
    actual_z = z_values[closest_idx]

    hii_dim = hii_dim_lookup[box_len]              # ← fixed
    cell_size_mpc = box_len / hii_dim              # ← fixed
    cell_size_kpc = cell_size_mpc * 1000

    print(f"\nBOX_LEN = {box_len:.0f} Mpc (Cell = {cell_size_mpc:.2f} Mpc = {cell_size_kpc:.0f} kpc) at z = {actual_z:.2f}:")

    for field_name, field_label, _ in fields_info:
        field_data = getattr(lightcone, field_name)
        slice_2d = field_data[:, :, closest_idx]
        print(f"  {field_name:15s}: min={slice_2d.min():10.3e}, "
              f"max={slice_2d.max():10.3e}, mean={slice_2d.mean():10.3e}")

print("\n" + "="*70)

# =============================================================================
# CELL 3c: Summary Statistics for 2D Slices at z=8
# =============================================================================

print("\n" + "="*70)
print(f"SUMMARY STATISTICS FOR 2D SLICES AT z ≈ {target_z}")
print("="*70)

for box_len in BOX_LEN_VALUES:
    if box_len not in lightcones or lightcones[box_len] is None:
        continue

    lightcone = lightcones[box_len]
    z_values = lightcone.lightcone_redshifts
    closest_idx = np.argmin(np.abs(z_values - target_z))
    actual_z = z_values[closest_idx]

    hii_dim = hii_dim_lookup[box_len]              # ← fixed
    cell_size_mpc = box_len / hii_dim              # ← fixed
    cell_size_kpc = cell_size_mpc * 1000

    brightness_slice = lightcone.brightness_temp[:, :, closest_idx]
    xHI_slice        = lightcone.xH_box[:, :, closest_idx]
    density_slice    = lightcone.density[:, :, closest_idx]
    velocity_slice   = lightcone.velocity[:, :, closest_idx]

    print(f"\nBOX_LEN = {box_len:.0f} Mpc (Cell = {cell_size_mpc:.2f} Mpc = {cell_size_kpc:.0f} kpc) at z = {actual_z:.2f}:")
    print(f"  Brightness temp [mK]: min={brightness_slice.min():.2f}, max={brightness_slice.max():.2f}, mean={brightness_slice.mean():.2f}")
    print(f"  Neutral fraction:     min={xHI_slice.min():.4f},  max={xHI_slice.max():.4f},  mean={xHI_slice.mean():.4f}")
    print(f"  Overdensity δ:        min={density_slice.min():.3f}, max={density_slice.max():.3f}, mean={density_slice.mean():.3f}")
    print(f"  Velocity [km/s]:      min={velocity_slice.min():.2f}, max={velocity_slice.max():.2f}, mean={velocity_slice.mean():.2f}")

print("\n" + "="*70)

# =============================================================================
# CELL 3d: Summary Statistics for Full Lightcones
# =============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS FOR FULL LIGHTCONES (ALL BOX SIZES)")
print("="*70)
print(f"Fixed cell size = {CELL_SIZE_MPC:.3f} Mpc")   # ← fixed
print(f"Total simulations: {len(BOX_LEN_VALUES)}")

for idx, box_len in enumerate(BOX_LEN_VALUES):
    if box_len not in lightcones or lightcones[box_len] is None:
        print(f"\n[{idx+1}/{len(BOX_LEN_VALUES)}] BOX_LEN = {box_len:.0f} Mpc: FAILED")
        continue

    lightcone = lightcones[box_len]
    hii_dim = hii_dim_lookup[box_len]              # ← fixed
    cell_size_mpc = box_len / hii_dim              # ← fixed
    cell_size_kpc = cell_size_mpc * 1000

    print(f"\n[{idx+1}/{len(BOX_LEN_VALUES)}] BOX_LEN = {box_len:.0f} Mpc (Cell = {cell_size_mpc:.2f} Mpc = {cell_size_kpc:.0f} kpc):")
    print(f"  Redshift range: [{lightcone.lightcone_redshifts.min():.2f}, {lightcone.lightcone_redshifts.max():.2f}]")
    print(f"  Brightness temp [mK]: min={lightcone.brightness_temp.min():.2f}, max={lightcone.brightness_temp.max():.2f}, mean={lightcone.brightness_temp.mean():.2f}")
    print(f"  Neutral fraction:     min={lightcone.xH_box.min():.4f}, max={lightcone.xH_box.max():.4f}, mean={lightcone.xH_box.mean():.4f}")
    print(f"  Overdensity δ:        min={lightcone.density.min():.3f}, max={lightcone.density.max():.3f}, mean={lightcone.density.mean():.3f}")
    print(f"  Velocity [km/s]:      min={lightcone.velocity.min():.2f}, max={lightcone.velocity.max():.2f}, mean={lightcone.velocity.mean():.2f}")

print("\n" + "="*70)

# =============================================================================
# CELL 3e: Compact Summary Table
# =============================================================================

print("\n" + "="*70)
print("COMPACT SUMMARY: REIONIZATION PROGRESS BY BOX SIZE")
print("="*70)
print(f"{'BOX[Mpc]':<12} {'Cell[kpc]':<12} {'<xHI>':<10} {'<Tb>[mK]':<12} {'z_range':<15}")
print("-" * 70)

for box_len in BOX_LEN_VALUES:
    if box_len not in lightcones or lightcones[box_len] is None:
        continue

    lightcone = lightcones[box_len]
    hii_dim = hii_dim_lookup[box_len]              # ← fixed
    cell_size_kpc = (box_len / hii_dim) * 1000     # ← fixed
    mean_xHI = lightcone.xH_box.mean()
    mean_Tb  = lightcone.brightness_temp.mean()
    z_min_actual = lightcone.lightcone_redshifts.min()
    z_max_actual = lightcone.lightcone_redshifts.max()

    print(f"{box_len:<12.0f} {cell_size_kpc:<12.0f} {mean_xHI:<10.4f} {mean_Tb:<12.2f} "
          f"[{z_min_actual:.2f}, {z_max_actual:.2f}]")

print("="*70)

# %%
# =============================================================================
# CELL 4: Reionization History Analysis - Box Size Scan
# =============================================================================
print("\n" + "="*70)
print("GENERATING REIONIZATION HISTORY COMPARISON")
print("="*70)

# Create rainbow colormap for all box sizes
cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=BOX_LEN_VALUES.min(), vmax=BOX_LEN_VALUES.max())

# =============================================================================
# PLOT 4a: Ionization Fraction vs Redshift (All Box Sizes)
# =============================================================================

def plot_reionization_xe(ax):
    for box_len in BOX_LEN_VALUES:
        if box_len not in lightcones or lightcones[box_len] is None:
            continue
            
        lightcone = lightcones[box_len]
        z_nodes = lightcone.node_redshifts[::-1]
        x_e_nodes = 1.0 - lightcone.global_xH[::-1]
        
        color = cmap(norm(box_len))
        
        ax.plot(
            z_nodes,
            x_e_nodes,
            linewidth=2.5,
            color=color,
            marker='o',
            markersize=3,
            alpha=0.8,
            label=f'{box_len:.0f} Mpc'
        )

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Ionization Fraction $x_e$')
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    ax.legend(loc='best', ncol=1)

plot_name = "reionization_history_xe_boxsize_all"

save_pdf_png(
    plot_reionization_xe,  # ← Pass the function
    plot_dir,
    plot_name,
    title=rf'Reionization History: Ionization Fraction (Fixed Resolution = {CELL_SIZE_MPC:.2f} Mpc)'
)

print(f"✓ Saved: {plot_name}")

# =============================================================================
# CELL 4f: Optical Depth Calculations for All Box Sizes
# =============================================================================

print("\n" + "="*70)
print("OPTICAL DEPTH CALCULATIONS")
print("="*70)

# Physical constants for optical depth calculation
c_km_s = 2.998e5                    # Speed of light [km/s]
h = 0.6766                          # Hubble parameter (from default cosmology)
H0 = 100 * h                        # Hubble constant [km/s/Mpc]
Omega_b = 0.04897468161869667       # Baryon density (from default cosmology)
Omega_m = 0.30964144154550644       # Matter density (from default cosmology)

# Critical density of the universe [protons/cm^3]
rho_crit_p_cm3 = 1.88e-29 * h**2 / (1.67e-24)  # Convert to protons/cm^3
n_H0_cm3 = Omega_b * rho_crit_p_cm3             # Mean hydrogen number density [cm^-3]

# Thomson scattering cross section
sigma_T_cm2 = 6.65e-25              # [cm^2]

# Convert to Mpc units
cm_per_Mpc = 3.086e24
n_e0_Mpc3 = n_H0_cm3 * cm_per_Mpc**3
sigma_T_Mpc2 = sigma_T_cm2 / cm_per_Mpc**2

# Prefactor for dτ calculation
prefactor = n_e0_Mpc3 * sigma_T_Mpc2  # [Mpc^-1]

print(f"\nPhysical constants:")
print(f"  n_H0 = {n_H0_cm3:.6e} cm^-3")
print(f"  σ_T = {sigma_T_cm2:.6e} cm^2")
print(f"  Prefactor = {prefactor:.6e} Mpc^-1")

# Dictionary to store optical depth results
tau_results = {}

# ← CHANGED: Loop through both arrays
for box_len, hii_dim in zip(BOX_LEN_VALUES, HII_DIM_VALUES):
    if box_len not in lightcones or lightcones[box_len] is None:
        continue
        
    lightcone = lightcones[box_len]
    
    # Extract redshift and distance axes
    red_axis = lightcone.lightcone_redshifts
    pos_axis = lightcone.lightcone_distances  # Comoving distance [Mpc]
    
    # Trim to z <= z_max if needed
    ind_z = np.where(red_axis <= z_max)[0]
    red_axis = red_axis[ind_z]
    pos_axis = pos_axis[ind_z]
    
    # Get ionization history
    z_nodes_sorted = lightcone.node_redshifts[::-1]
    xHI_nodes_sorted = lightcone.global_xH[::-1]
    x_e_nodes_sorted = 1.0 - xHI_nodes_sorted
    
    # Interpolate x_e onto lightcone redshift grid
    x_e_interp = np.interp(red_axis, z_nodes_sorted, x_e_nodes_sorted)
    
    # Calculate geometric quantities
    s = pos_axis  # Comoving distance [Mpc]
    ds = np.diff(s)  # Distance element [Mpc]
    
    # Midpoint values for integration
    z_mid = 0.5 * (red_axis[:-1] + red_axis[1:])
    x_e_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])
    
    # Calculate dτ
    dtau = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds
    
    # Cumulative optical depth
    tau = np.cumsum(dtau)
    tau_total = tau[-1]
    
    # Store results (including hii_dim for reference)  # ← CHANGED
    tau_results[box_len] = {
        'red_axis': red_axis,
        'pos_axis': pos_axis,
        's': s,
        'ds': ds,
        'z_mid': z_mid,
        'x_e_interp': x_e_interp,
        'x_e_mid': x_e_mid,
        'dtau': dtau,
        'tau': tau,
        'tau_total': tau_total,
        'hii_dim': hii_dim  # ← CHANGED: Store HII_DIM
    }

# Print compact summary  # ← CHANGED
print(f"\n{'BOX[Mpc]':<12} {'HII_DIM':<10} {'Cell[Mpc]':<12} {'z_range':<15} {'<ds>[Mpc]':<12} {'τ_total':<10}")
print("-" * 85)

for box_len in sorted(tau_results.keys()):
    results = tau_results[box_len]
    red_axis = np.asarray(results['red_axis'])
    ds = np.asarray(results['ds'])
    tau_total = float(np.asarray(results['tau_total']))
    hii_dim = results['hii_dim']  # ← CHANGED
    cell_size_mpc = box_len / hii_dim  # ← CHANGED
    
    print(f"{box_len:<12.0f} {hii_dim:<10} {cell_size_mpc:<12.3f} [{red_axis.min():.2f}, {red_axis.max():.2f}] "
          f"{ds.mean():<12.3f} {tau_total:<10.6f}")

print("\n" + "="*70)

# =============================================================================
# PLOT: Cumulative Optical Depth τ vs z (All Box Sizes - Rainbow)
# =============================================================================

def plot_tau_vs_z(ax):
    for box_len in sorted(tau_results.keys()):
        results = tau_results[box_len]
        
        z_mid_plot = np.asarray(results['z_mid'])
        tau_plot = np.asarray(results['tau'])
        
        color = cmap(norm(box_len))
        
        ax.plot(
            z_mid_plot,
            tau_plot,
            linewidth=2.5,
            color=color,
            marker='o',
            markersize=3,
            alpha=0.8,
            label=f'{box_len:.0f} Mpc'
        )

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$')
    ax.invert_xaxis()
    ax.legend(loc='best', ncol=1)

plot_name = "tau_vs_z_boxsize_all"

save_pdf_png(
    plot_tau_vs_z,
    plot_dir,
    plot_name,
    title=rf'Cumulative Optical Depth vs Redshift (Fixed Resolution = {CELL_SIZE_MPC:.2f} Mpc)'
)

print(f"✓ Saved: {plot_name}")


# =============================================================================
# PLOT: Optical Depth Element dτ vs z (All Box Sizes - Rainbow)
# =============================================================================

def plot_dtau_vs_z(ax):
    for box_len in sorted(tau_results.keys()):
        results = tau_results[box_len]
        z_mid_plot = np.asarray(results['z_mid'])
        dtau_plot = np.asarray(results['dtau'])
        
        color = cmap(norm(box_len))
        
        ax.plot(
            z_mid_plot,
            dtau_plot,
            linewidth=2.5,
            color=color,
            marker='o',
            markersize=3,
            alpha=0.8,
            label=f'{box_len:.0f} Mpc'
        )

    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Optical Depth Element $d\tau$')
    ax.invert_xaxis()
    ax.legend(loc='best', ncol=1)

plot_name = "dtau_vs_z_boxsize_all"

save_pdf_png(
    plot_dtau_vs_z,
    plot_dir,
    plot_name,
    title=r'Optical Depth Element $d\tau$ vs Redshift'
)

print(f"✓ Saved: {plot_name}")


# =============================================================================
# PLOT: Total Optical Depth vs BOX_LEN
# =============================================================================

# Prepare data outside function
box_vals = sorted(tau_results.keys())
tau_total_vals = [float(np.asarray(tau_results[b]['tau_total'])) for b in box_vals]
cell_sizes_mpc = [box_vals[i] / tau_results[box_vals[i]]['hii_dim'] for i in range(len(box_vals))]

def plot_tau_total_vs_boxsize(ax):
    ax.plot(
        box_vals,
        tau_total_vals,
        'o-',
        linewidth=3,
        markersize=10,
        color='darkblue',
        label=r'Total $\tau$'
    )

    ax.set_xlabel(r'BOX\_LEN [Mpc]')
    ax.set_ylabel(r'Total Optical Depth $\tau$')
    ax.legend(loc='best')

    ax.text(
        0.05, 0.95,
        rf'Fixed Resolution = {CELL_SIZE_MPC:.2f} Mpc',
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

plot_name = "tau_total_vs_boxsize"

save_pdf_png(
    plot_tau_total_vs_boxsize,
    plot_dir,
    plot_name,
    title="Total Optical Depth vs Box Size"
)

print(f"✓ Saved: {plot_name}")


# =============================================================================
# PLOT: Total Optical Depth vs Cell Size
# =============================================================================

def plot_tau_total_vs_cellsize(ax):
    ax.plot(
        cell_sizes_mpc,
        tau_total_vals,
        's-',
        linewidth=3,
        markersize=10,
        color='darkred',
        label=r'Total $\tau$'
    )

    ax.set_xlabel(r'Cell Size [Mpc]')
    ax.set_ylabel(r'Total Optical Depth $\tau$')
    ax.legend(loc='best')

    ax.text(
        0.05, 0.95,
        rf'Fixed Resolution = {CELL_SIZE_MPC:.2f} Mpc',
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

plot_name = "tau_total_vs_cellsize"

save_pdf_png(
    plot_tau_total_vs_cellsize,
    plot_dir,
    plot_name,
    title="Total Optical Depth vs Cell Size (Should be constant)"
)

print(f"✓ Saved: {plot_name}")

# %%
# =============================================================================
# CELL 5: Compute kSZ Integrand with Visibility Function — Skewed LOS
# kSZ integrand = (1 + δ) × x_e × v_z / c × e^(-τ(z))
# Stores both unrotated (θ=0°) and rotated (θ=angle_deg°) skewers per box
# Full box face sampling: Nlos_box = Ndim² for every box
# =============================================================================

print("\n" + "="*70)
print("COMPUTING kSZ INTEGRAND WITH VISIBILITY FUNCTION (SKEWED LOS)")
print("="*70)

c_Mpc_s = 299792.458 / 3.08567758e19   # Speed of light in Mpc/s
print(f"Speed of light: c = {c_Mpc_s:.6e} Mpc/s")

kSZ_results = {}

for box_len, hii_dim in zip(BOX_LEN_VALUES, HII_DIM_VALUES):
    if box_len not in lightcones or lightcones[box_len] is None:
        continue
    if box_len not in tau_results:
        continue

    cell_size_mpc = box_len / hii_dim

    print(f"\n{'='*70}")
    print(f"BOX_LEN = {box_len:.0f} Mpc, HII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc)")
    print(f"{'='*70}")

    lightcone = lightcones[box_len]
    res       = tau_results[box_len]

    red_axis  = np.asarray(res['red_axis'])
    z_mid     = np.asarray(res['z_mid'])
    tau       = np.asarray(res['tau'])
    pos_axis  = np.asarray(res['pos_axis'])
    ds        = np.asarray(res['ds'])

    Nbins     = len(red_axis)
    Ndim      = int(hii_dim)
    Lbox      = float(box_len)
    cell_size = Lbox / Ndim
    delta_z   = float(pos_axis[1] - pos_axis[0])

    # Relative comoving distance starting at 0
    s_axis = pos_axis - pos_axis[0]

    # ==========================================================================
    # Extract 3D fields
    # ==========================================================================

    red_axis_full = np.asarray(lightcone.lightcone_redshifts)
    ind_z = np.where(red_axis_full <= z_max)[0]

    _Delta_3d = np.array(lightcone.density[:, :, ind_z],  dtype=np.float32) + 1.0
    _xHI_3d   = np.array(lightcone.xH_box[:, :, ind_z],   dtype=np.float32)
    _vel_3d   = np.array(lightcone.velocity[:, :, ind_z],  dtype=np.float32)

    print(f"3D field shape: {_Delta_3d.shape}  (Ndim x Ndim x Nbins)")

    # ==========================================================================
    # Build LOS grid — always full box face (Ndim²)
    # ==========================================================================

    Nlos_box = Ndim * Ndim   # full face, consistent across all box sizes
    LOS_ind  = np.array([[i, j] for i in range(Ndim) for j in range(Ndim)])

    print(f"LOS grid: {Ndim}×{Ndim} = {Nlos_box} skewers (full box face)")

    # ==========================================================================
    # Skewer extraction functions
    # ==========================================================================

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

    # ==========================================================================
    # Extract unrotated AND rotated skewers
    # ==========================================================================

    density_unrot  = np.zeros((Nlos_box, Nbins), dtype=np.float32)
    xH_box_unrot   = np.zeros((Nlos_box, Nbins), dtype=np.float32)
    velocity_unrot = np.zeros((Nlos_box, Nbins), dtype=np.float32)

    density_rot    = np.zeros((Nlos_box, Nbins), dtype=np.float32)
    xH_box_rot     = np.zeros((Nlos_box, Nbins), dtype=np.float32)
    velocity_rot   = np.zeros((Nlos_box, Nbins), dtype=np.float32)

    for k, (x0, y0) in enumerate(LOS_ind):
        ix, iy = int(x0) % Ndim, int(y0) % Ndim

        # Unrotated — direct z-axis read
        density_unrot[k]  = _Delta_3d[ix, iy, :]
        xH_box_unrot[k]   = _xHI_3d[ix,  iy, :]
        velocity_unrot[k] = _vel_3d[ix,   iy, :]

        # Rotated — interpolated diagonal
        density_rot[k]    = rotated_skewer(_Delta_3d, int(x0), int(y0))
        xH_box_rot[k]     = rotated_skewer(_xHI_3d,   int(x0), int(y0))
        velocity_rot[k]   = rotated_skewer(_vel_3d,    int(x0), int(y0))

        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{Nlos_box} skewers done")

    print(f"  Extraction complete.")
    print(f"  Unrotated — <1+δ>={density_unrot.mean():.4f}  "
          f"<xHI>={xH_box_unrot.mean():.4f}  <v>={velocity_unrot.mean():.4e}")
    print(f"  Rotated   — <1+δ>={density_rot.mean():.4f}  "
          f"<xHI>={xH_box_rot.mean():.4f}  <v>={velocity_rot.mean():.4e}")

    # ==========================================================================
    # Visibility function (1D, shape Nbins) — same for both
    # ==========================================================================

    tau_extended     = np.concatenate([[0], tau])
    tau_at_lightcone = np.interp(red_axis,
                                  np.concatenate([[red_axis[0]], z_mid]),
                                  tau_extended)
    visibility = np.exp(-tau_at_lightcone)   # (Nbins,)

    print(f"τ range      : [{tau_at_lightcone.min():.6f}, {tau_at_lightcone.max():.6f}]")
    print(f"e^(-τ) range : [{visibility.min():.6f}, {visibility.max():.6f}]")

    # ==========================================================================
    # kSZ integrand — shape (Nlos_box, Nbins) for both
    # ==========================================================================

    x_e_unrot  = 1.0 - xH_box_unrot
    x_e_rot    = 1.0 - xH_box_rot
    vis_broad  = visibility[np.newaxis, :]   # (1, Nbins)

    kSZ_integrand_unrot = density_unrot * x_e_unrot * velocity_unrot / c_Mpc_s * vis_broad
    kSZ_integrand_rot   = density_rot   * x_e_rot   * velocity_rot   / c_Mpc_s * vis_broad

    print(f"\nkSZ integrand shape : {kSZ_integrand_unrot.shape}")
    print(f"  Unrotated — RMS={np.sqrt(np.mean(kSZ_integrand_unrot**2)):.4e}  "
          f"mean={kSZ_integrand_unrot.mean():.4e}")
    print(f"  Rotated   — RMS={np.sqrt(np.mean(kSZ_integrand_rot**2)):.4e}  "
          f"mean={kSZ_integrand_rot.mean():.4e}")

    # ==========================================================================
    # Store
    # ==========================================================================

    kSZ_results[box_len] = {
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
        'Nlos_box'            : Nlos_box,
        'Ndim'                : Ndim,
        'hii_dim'             : hii_dim,
        's_axis'              : s_axis,
        'delta_z'             : delta_z,
        'cell_size'           : cell_size,
    }

print("\n" + "="*70)
print("kSZ INTEGRAND CALCULATION COMPLETE (UNROTATED + ROTATED SKEWERS)")
print(f"Computed for {len(kSZ_results)} BOX_LEN values")
print("="*70)

# =============================================================================
# Summary Statistics Table
# =============================================================================

print("\n" + "="*70)
print("kSZ INTEGRAND SUMMARY STATISTICS")
print("="*70)
print(f"{'BOX[Mpc]':<12} {'Ndim':<8} {'Nlos=Ndim²':<12} "
      f"{'RMS_unrot':<14} {'RMS_rot':<14} {'Cell[Mpc]':<12}")
print("-" * 75)

for box_len in sorted(kSZ_results.keys()):
    r    = kSZ_results[box_len]
    rms_u = np.sqrt(np.mean(r['kSZ_integrand_unrot']**2))
    rms_r = np.sqrt(np.mean(r['kSZ_integrand_rot']**2))
    print(f"{box_len:<12.0f} {r['Ndim']:<8} {r['Nlos_box']:<12} "
          f"{rms_u:<14.4e} {rms_r:<14.4e} {box_len/r['hii_dim']:<12.3f}")

# %%
# =============================================================================
# NOT FOR REPORTS
# CEll5b: PLOT: kSZ Integrand - All Box Sizes Stacked
# =============================================================================

print("\n" + "="*70)
print("GENERATING kSZ INTEGRAND PLOTS")
print("="*70)

# Stacked plots for all box sizes
fig, axes = plt.subplots(len(BOX_LEN_VALUES), 1, 
                         figsize=(14, 4*len(BOX_LEN_VALUES)), 
                         constrained_layout=True)

if len(BOX_LEN_VALUES) == 1:
    axes = [axes]

# Rainbow colormap for labels
cmap = mpl.cm.rainbowprint("v_los_km_s stats [km/s]:",
      np.min(v_los_km_s),
      np.mean(v_los_km_s),
      np.max(v_los_km_s),
      np.std(v_los_km_s))
norm = mpl.colors.Normalize(vmin=BOX_LEN_VALUES.min(), vmax=BOX_LEN_VALUES.max())

for idx, (box_len, ax) in enumerate(zip(BOX_LEN_VALUES, axes)):
    if box_len not in kSZ_results:
        ax.text(0.5, 0.5, f'BOX_LEN = {box_len:.0f} Mpc\nNo data', 
               ha='center', va='center',
               transform=ax.transAxes, fontsize=16, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        continue
        
    lightcone = lightcones[box_len]
    
    plotting.lightcone_sliceplot(lightcone, 'kSZ_integrand', ax=ax, fig=fig)
    
    # Change colormap to seismic (diverging colormap for positive/negative)
    im = ax.images[0]
    im.set_cmap('seismic')
    
    # Set symmetric color limits
    kSZ_data = kSZ_results[box_len]['kSZ_integrand']
    vmax = np.percentile(np.abs(kSZ_data), 99)
    im.set_clim(-vmax, vmax)
    
    # Get rainbow color for label
    color = cmap(norm(box_len))
    cell_size_mpc = box_len / HII_DIM_FIXED
    cell_size_kpc = cell_size_mpc * 1000
    
    # Add label with box info
    ax.text(0.02, 0.98, 
           f'BOX = {box_len:.0f} Mpc  |  Cell = {cell_size_mpc:.2f} Mpc ({cell_size_kpc:.0f} kpc)', 
           transform=ax.transAxes, 
           fontsize=13, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, 
                    edgecolor='black', linewidth=2))
    
    # Add box size on right
    ax.text(0.98, 0.98, 
           f'{box_len:.0f} Mpc', 
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save
plot_name = "kSZ_integrand_with_visibility_boxsize_stack"
fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

fig.suptitle(r'kSZ Integrand: $(1+\delta) \times x_e \times v_z/c \times e^{-\tau(z)}$ (HII_DIM=' + f'{HII_DIM_FIXED})', 
             fontsize=20, fontweight='bold', y=0.995)
fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

print(f"✓ Saved: {plot_name}")
plt.close(fig)

# =============================================================================
# PLOT: Visibility Function e^(-τ) vs z (All Box Sizes - Rainbow)
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

for box_len in sorted(kSZ_results.keys()):
    results = kSZ_results[box_len]
    red_axis_plot = results['red_axis']
    visibility_plot = results['visibility']
    
    color = cmap(norm(box_len))
    
    ax.plot(red_axis_plot, visibility_plot, 
           linewidth=2.5, color=color,
           marker='o', markersize=3, alpha=0.8,
           label=f'{box_len:.0f} Mpc')

ax.set_xlabel('Redshift $z$', fontsize=20)
ax.set_ylabel(r'Visibility Function $e^{-\tau(z)}$', fontsize=20)
ax.set_ylim(0, 1.05)
ax.invert_xaxis()
ax.legend(fontsize=12, loc='best', ncol=1)
##ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'BOX\_LEN [Mpc]', fontsize=16)

plot_name = "visibility_function_vs_z_boxsize_all"
fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

ax.set_title(r'Visibility Function $e^{-\tau(z)}$ vs Redshift', 
            fontsize=20, fontweight='bold')
fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

print(f"✓ Saved: {plot_name}")
plt.close(fig)

print("\n" + "="*70)
print("kSZ INTEGRAND PLOTTING COMPLETE!")
print("="*70)

# %%
# =============================================================================
# CELL 6: Compute Line-of-Sight Integrated kSZ Maps for All Box Sizes
# Three cases: Original (full 3D), Unrotated skewers, Rotated skewers
# kSZ(z=5) = ∫ n_e0 σ_T (1/a²) (1+δ) x_e v_z/c e^(-τ) ds
# =============================================================================

print("\n" + "="*70)
print("LINE-OF-SIGHT kSZ MAP INTEGRATION — THREE CASES")
print("="*70)

# Physical constants in CGS
c_cm_s       = 3.0e10       # cm/s
sigma_T_cm2  = 6.6525e-25   # cm²
n_e0_cm3     = 2.06e-7      # cm⁻³
Mpc_to_cm    = 3.0857e24    # cm/Mpc
prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s   # [s⁻¹]

print(f"Prefactor n_e0 × σ_T × c = {prefactor_cgs:.4e} s⁻¹")

kSZ_map_results = {}

for box_len, hii_dim in zip(BOX_LEN_VALUES, HII_DIM_VALUES):
    if box_len not in kSZ_results or box_len not in tau_results:
        continue

    cell_size_mpc = box_len / hii_dim

    print(f"\n{'='*70}")
    print(f"BOX_LEN = {box_len:.0f} Mpc, HII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc)")
    print(f"{'='*70}")

    lightcone = lightcones[box_len]
    res       = kSZ_results[box_len]

    red_axis  = np.asarray(res['red_axis'])
    ind_z     = res['ind_z']
    ds_Mpc    = np.asarray(res['ds'])
    ds_cm     = ds_Mpc * Mpc_to_cm

    # Scale factor midpoints — shared across all three cases
    a             = 1.0 / (1.0 + red_axis)
    a_squared     = a**2
    a_squared_mid = 0.5 * (a_squared[:-1] + a_squared[1:])   # (Nbins-1,)

    # Prefactor × ds/c  — (Nbins-1,)
    weight = (prefactor_cgs / a_squared_mid) * (ds_cm / c_cm_s)

    # ===========================================================================
    # CASE 1: Original — full 3D LOS integral  (Ndim, Ndim, Nbins)
    # ===========================================================================

    density_orig = np.array(lightcone.density[:, :, ind_z],  dtype=np.float64) + 1.0
    x_e_orig     = 1.0 - np.array(lightcone.xH_box[:, :, ind_z],  dtype=np.float64)
    vel_orig      = np.array(lightcone.velocity[:, :, ind_z], dtype=np.float64)
    vis_orig      = res['visibility']   # (Nbins,)

    integrand_orig = (density_orig * x_e_orig * vel_orig /
                      c_cm_s * vis_orig[None, None, :])   # Mpc/s → unitless after /c_cm_s? 
    # Note: velocity from 21cmFAST is in Mpc/s, c_cm_s is cm/s — keep consistent:
    # use c_Mpc_s for v/c ratio, then multiply by prefactor_cgs separately
    c_Mpc_s      = 299792.458 / 3.08567758e19
    integrand_orig = (density_orig * x_e_orig *
                      (vel_orig / c_Mpc_s) *
                      vis_orig[None, None, :])             # dimensionless ratio

    integrand_orig_mid = 0.5 * (integrand_orig[:, :, :-1] +
                                 integrand_orig[:, :, 1:])  # (Ndim, Ndim, Nbins-1)

    kSZ_map_orig = np.sum(
        weight[None, None, :] * integrand_orig_mid, axis=2
    )   # (Ndim, Ndim)

    print(f"Original   — shape {kSZ_map_orig.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map_orig**2)):.4e}")

    # ===========================================================================
    # CASE 2: Unrotated skewers  (Nlos_box, Nbins)
    # ===========================================================================

    ki_unrot     = np.asarray(res['kSZ_integrand_unrot'], dtype=np.float64)
    ki_unrot_mid = 0.5 * (ki_unrot[:, :-1] + ki_unrot[:, 1:])   # (Nlos, Nbins-1)

    kSZ_1d_unrot = np.sum(weight[None, :] * ki_unrot_mid, axis=1)   # (Nlos,)

    # Reshape to 2D map
    Nlos_box  = res['Nlos_box']
    npix_map  = int(np.floor(np.sqrt(Nlos_box)))
    n_use     = npix_map * npix_map
    kSZ_map_unrot = kSZ_1d_unrot[:n_use].reshape(npix_map, npix_map)

    print(f"Unrotated  — 1D shape {kSZ_1d_unrot.shape}  "
          f"map {kSZ_map_unrot.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map_unrot**2)):.4e}")

    # ===========================================================================
    # CASE 3: Rotated skewers  (Nlos_box, Nbins)
    # ===========================================================================

    ki_rot     = np.asarray(res['kSZ_integrand_rot'], dtype=np.float64)
    ki_rot_mid = 0.5 * (ki_rot[:, :-1] + ki_rot[:, 1:])   # (Nlos, Nbins-1)

    kSZ_1d_rot = np.sum(weight[None, :] * ki_rot_mid, axis=1)   # (Nlos,)

    kSZ_map_rot = kSZ_1d_rot[:n_use].reshape(npix_map, npix_map)

    print(f"Rotated    — 1D shape {kSZ_1d_rot.shape}  "
          f"map {kSZ_map_rot.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map_rot**2)):.4e}")

    # ===========================================================================
    # Store all three
    # ===========================================================================

    kSZ_map_results[box_len] = {
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
        'pix_size_Mpc'  : float(box_len) / npix_map,
        'pix_size_orig' : float(box_len) / hii_dim,
        'hii_dim'       : hii_dim,
        'Nlos_box'      : Nlos_box,
    }

    lightcone.kSZ_map      = kSZ_map_orig    # attach original to lightcone object

print("\n" + "="*70)
print("kSZ MAP INTEGRATION COMPLETE — THREE CASES")
print(f"Computed maps for {len(kSZ_map_results)} BOX_LEN values")
print("="*70)

# =============================================================================
# Summary Statistics Table
# =============================================================================

print("\n" + "="*70)
print("kSZ MAP SUMMARY STATISTICS")
print("="*70)
print(f"{'BOX[Mpc]':<10} {'Cell[Mpc]':<10} {'npix_skew':<12} "
      f"{'RMS_orig':<14} {'RMS_unrot':<14} {'RMS_rot':<14}")
print("-" * 80)

for box_len in sorted(kSZ_map_results.keys()):
    r = kSZ_map_results[box_len]
    rms_o = np.sqrt(np.mean(r['kSZ_map_orig']**2))
    rms_u = np.sqrt(np.mean(r['kSZ_map_unrot']**2))
    rms_r = np.sqrt(np.mean(r['kSZ_map_rot']**2))
    print(f"{box_len:<10.0f} {box_len/r['hii_dim']:<10.3f} "
          f"{r['npix_map']:<12} "
          f"{rms_o:<14.4e} {rms_u:<14.4e} {rms_r:<14.4e}")

# %%
# =============================================================================
# Not FOR REPORTs
# Cell 6b:  PLOT: kSZ Maps - All Box Sizes Side-by-Side
# =============================================================================

print("\n" + "="*70)
print("GENERATING kSZ MAP PLOTS")
print("="*70)

fig, axes = plt.subplots(1, len(BOX_LEN_VALUES), 
                         figsize=(5*len(BOX_LEN_VALUES), 5.5), 
                         constrained_layout=True)

if len(BOX_LEN_VALUES) == 1:
    axes = [axes]

# Rainbow colormap for labels
cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=BOX_LEN_VALUES.min(), vmax=BOX_LEN_VALUES.max())

# Find global vmax for consistent color scale
vmax_global = 0
for box_len in kSZ_map_results.keys():
    kSZ_map = kSZ_map_results[box_len]['kSZ_map']
    vmax_global = max(vmax_global, np.percentile(np.abs(kSZ_map), 99))

for idx, (box_len, ax) in enumerate(zip(BOX_LEN_VALUES, axes)):
    if box_len not in kSZ_map_results:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=16)
        ax.set_title(f'BOX={box_len:.0f} Mpc', fontsize=14)
        continue
        
    kSZ_map = kSZ_map_results[box_len]['kSZ_map']
    
    # Plot the kSZ map
    im = ax.imshow(kSZ_map.T,
                   cmap='seismic',
                   origin='lower',
                   extent=[0, box_len, 0, box_len],
                   aspect='equal',
                   vmin=-vmax_global,
                   vmax=vmax_global)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    # Labels
    ax.set_xlabel('x [Mpc]', fontsize=11)
    ax.set_ylabel('y [Mpc]', fontsize=11)
    
    # Get rainbow color for title
    color_label = cmap(norm(box_len))
    cell_size_mpc = box_len / HII_DIM_FIXED
    cell_size_kpc = cell_size_mpc * 1000
    
    ax.set_title(f'BOX={box_len:.0f} Mpc\nCell={cell_size_mpc:.2f} Mpc\n({cell_size_kpc:.0f} kpc)', 
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color_label, alpha=0.6, 
                         edgecolor='black', linewidth=1.5))

# Overall title
fig.suptitle(r'kSZ Maps at $z=5$ (line-of-sight integrated, HII_DIM=' + f'{HII_DIM_FIXED})', 
             fontsize=18, fontweight='bold')

# Save
plot_name = "kSZ_maps_z5_boxsize_all"
fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

print(f"✓ Saved: {plot_name}")
plt.close(fig)

# =============================================================================
# PLOT: kSZ Maps - Stacked Vertically
# =============================================================================

fig, axes = plt.subplots(len(BOX_LEN_VALUES), 1, 
                         figsize=(8, 6*len(BOX_LEN_VALUES)), 
                         constrained_layout=True)

if len(BOX_LEN_VALUES) == 1:
    axes = [axes]

for idx, (box_len, ax) in enumerate(zip(BOX_LEN_VALUES, axes)):
    if box_len not in kSZ_map_results:
        ax.text(0.5, 0.5, f'BOX={box_len:.0f} Mpc\nNo data', 
               ha='center', va='center',
               transform=ax.transAxes, fontsize=16, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        continue
        
    kSZ_map = kSZ_map_results[box_len]['kSZ_map']
    
    # Plot the kSZ map
    im = ax.imshow(kSZ_map.T,
                   cmap='seismic',
                   origin='lower',
                   extent=[0, box_len, 0, box_len],
                   aspect='equal',
                   vmin=-vmax_global,
                   vmax=vmax_global)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('kSZ (dimensionless)', fontsize=11)
    
    # Labels
    ax.set_xlabel('x [Mpc]', fontsize=12)
    ax.set_ylabel('y [Mpc]', fontsize=12)
    
    # Get rainbow color for label
    color_label = cmap(norm(box_len))
    cell_size_mpc = box_len / HII_DIM_FIXED
    cell_size_kpc = cell_size_mpc * 1000
    
    # Add label
    ax.text(0.02, 0.98, 
           f'BOX = {box_len:.0f} Mpc  |  Cell = {cell_size_mpc:.2f} Mpc ({cell_size_kpc:.0f} kpc)', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color_label, alpha=0.7, 
                    edgecolor='black', linewidth=2))

# Overall title
fig.suptitle(r'kSZ Maps at $z=5$ - Box Size Comparison', 
             fontsize=20, fontweight='bold', y=0.995)

# Save
plot_name = "kSZ_maps_z5_boxsize_stack"
fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

print(f"✓ Saved: {plot_name}")
plt.close(fig)

# =============================================================================
# PLOT: kSZ Map Histograms - All Box Sizes Overlay
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

for box_len in sorted(kSZ_map_results.keys()):
    kSZ_map = kSZ_map_results[box_len]['kSZ_map']
    color = cmap(norm(box_len))
    
    ax.hist(kSZ_map.flatten(), bins=100, 
           color=color, 
           alpha=0.5, 
           edgecolor='black',
           linewidth=0.5,
           label=f'{box_len:.0f} Mpc')

ax.set_xlabel('kSZ Signal (dimensionless)', fontsize=20)
ax.set_ylabel('Number of Pixels', fontsize=20)
ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.legend(fontsize=14, loc='best')
##ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'BOX\_LEN [Mpc]', fontsize=16)

# Save
plot_name = "kSZ_map_histogram_boxsize_all"
fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

ax.set_title(f'kSZ Map Pixel Distribution (HII_DIM={HII_DIM_FIXED})', 
            fontsize=20, fontweight='bold')
fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

print(f"✓ Saved: {plot_name}")
plt.close(fig)

# =============================================================================
# PLOT: kSZ RMS vs Box Size
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

box_vals = sorted(kSZ_map_results.keys())
rms_vals = []
std_vals = []
cell_sizes_kpc = []

for box_len in box_vals:
    kSZ_map = kSZ_map_results[box_len]['kSZ_map']
    rms_vals.append(np.sqrt(np.mean(kSZ_map**2)))
    std_vals.append(np.std(kSZ_map))
    cell_sizes_kpc.append((box_len / HII_DIM_FIXED) * 1000)

ax.plot(box_vals, rms_vals, 'o-', linewidth=3, markersize=10,
       color='darkblue', label='RMS')
ax.plot(box_vals, std_vals, 's-', linewidth=3, markersize=10,
       color='darkred', label='Std Dev')

ax.set_xlabel(r'BOX\_LEN [Mpc]', fontsize=20)
ax.set_ylabel('kSZ Signal (dimensionless)', fontsize=20)
ax.legend(loc='best', fontsize=16)
##ax.grid(True, alpha=0.3, linestyle='--')

# Add annotation
ax.text(0.05, 0.95, f'HII_DIM = {HII_DIM_FIXED}',
       transform=ax.transAxes, fontsize=14,
       verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plot_name = "kSZ_rms_vs_boxsize"
fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')

ax.set_title('kSZ Map RMS vs Box Size', 
            fontsize=22, fontweight='bold')
fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')

print(f"✓ Saved: {plot_name}")
plt.close(fig)


print("\n" + "="*70)
print("kSZ MAP GENERATION COMPLETE!")
print("="*70)

# %%
# =============================================================================
# CELL 7: Compute kSZ Power Spectrum - P(k), C_ℓ, and D_ℓ for All Box Sizes
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
    Returns dict with k_centers, P1d, errors, ell, Cl, Dl, Dl_uK2 and their errors.
    Normalization: P(k) [Mpc²] = (pix_size/N)² |FFT(map - mean)|²
    """
    npix      = ksz_map_2d.shape[0]
    pix_size  = box_size_Mpc / npix
    m         = ksz_map_2d - ksz_map_2d.mean()

    fft_shift = np.fft.fftshift(np.fft.fft2(m))
    ps2d      = (pix_size / npix)**2 * np.abs(fft_shift)**2

    dk   = 2 * np.pi / (npix * pix_size)
    kx   = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    ky   = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), n_kbins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    P1d       = np.full(n_kbins, np.nan)
    P1d_err_sample = np.full(n_kbins, np.nan)
    n_modes   = np.zeros(n_kbins)

    for i in range(n_kbins):
        mask = (kgrid >= k_bins[i]) & (kgrid < k_bins[i+1])
        n_modes[i] = mask.sum()
        if n_modes[i] > 0:
            vals = ps2d[mask]
            P1d[i]           = vals.mean()
            P1d_err_sample[i] = vals.std() / np.sqrt(n_modes[i])

    # Cosmic variance
    k_volume       = (box_size_Mpc / (2 * np.pi))**2
    n_modes_cosmic = 2 * np.pi * k_centers * k_volume * (k_bins[1:] - k_bins[:-1])
    cv_frac        = np.where(n_modes_cosmic > 0,
                              1.0 / np.sqrt(n_modes_cosmic), np.nan)
    P1d_err_cosmic = P1d * cv_frac
    P1d_err_total  = np.sqrt(P1d_err_sample**2 + P1d_err_cosmic**2)

    # ell, Cl, Dl
    ell  = k_centers * chi_comoving_Mpc / 0.67
    Cl   = P1d * 0.67**2 / D_A_Mpc**2
    Dl   = ell * (ell + 1) * Cl / (2 * np.pi)
    Dl_uK2 = Dl * T_CMB_z5_uK**2

    def _prop_err(P_err):
        Cl_e  = P_err * 0.67**2 / D_A_Mpc**2
        Dl_e  = ell * (ell + 1) * Cl_e / (2 * np.pi)
        return Dl_e * T_CMB_z5_uK**2

    return dict(
        k_centers        = k_centers,
        k_bins           = k_bins,
        P1d              = P1d,
        P1d_err_sample   = P1d_err_sample,
        P1d_err_cosmic   = P1d_err_cosmic,
        P1d_err_total    = P1d_err_total,
        n_modes          = n_modes,
        n_modes_cosmic   = n_modes_cosmic,
        ell              = ell,
        Cl               = Cl,
        Dl_uK2           = Dl_uK2,
        Dl_uK2_err_sample = _prop_err(P1d_err_sample),
        Dl_uK2_err_cosmic = _prop_err(P1d_err_cosmic),
        Dl_uK2_err_total  = _prop_err(P1d_err_total),
        pix_size         = pix_size,
        npix             = npix,
        dk               = dk,
    )

# =============================================================================
# Main loop — compute PS for all three cases per box
# =============================================================================

power_spectrum_results = {}

for box_len, hii_dim in zip(BOX_LEN_VALUES, HII_DIM_VALUES):
    if box_len not in kSZ_map_results:
        continue

    cell_size_mpc = box_len / hii_dim
    r = kSZ_map_results[box_len]

    print(f"\n{'='*70}")
    print(f"BOX_LEN = {box_len:.0f} Mpc, HII_DIM = {hii_dim} (Cell = {cell_size_mpc:.3f} Mpc)")
    print(f"{'='*70}")

    # --- Original (full box, pixel size = cell_size_mpc) ---
    ps_orig  = compute_ps(r['kSZ_map_orig'],  box_len)

    # --- Unrotated skewers (pixel size = box_len / npix_map) ---
    ps_unrot = compute_ps(r['kSZ_map_unrot'], box_len)

    # --- Rotated skewers ---
    ps_rot   = compute_ps(r['kSZ_map_rot'],   box_len)

    for tag, ps in [('Original', ps_orig),
                    ('Unrotated', ps_unrot),
                    ('Rotated',   ps_rot)]:
        valid = ~np.isnan(ps['P1d']) & (ps['P1d'] > 0)
        print(f"  {tag:<12} — npix={ps['npix']}  "
              f"dk={ps['dk']:.5f} Mpc⁻¹  "
              f"valid bins={valid.sum()}")

    power_spectrum_results[box_len] = {
        'orig'   : ps_orig,
        'unrot'  : ps_unrot,
        'rot'    : ps_rot,
        'hii_dim': hii_dim,
    }

print("\n" + "="*70)
print("POWER SPECTRUM CALCULATION COMPLETE — THREE CASES")
print(f"Computed for {len(power_spectrum_results)} BOX_LEN values")
print("="*70)

# =============================================================================
# PLOT 1: P(k) — three cases, rainbow by box size
# =============================================================================

cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=BOX_LEN_VALUES.min(), vmax=BOX_LEN_VALUES.max())

def plot_Pk_three_cases(ax):
    for box_len in sorted(power_spectrum_results.keys()):
        r    = power_spectrum_results[box_len]
        color = cmap(norm(box_len))

        for tag, ps, ls, mk in [
            ('orig',  r['orig'],  '-',  '^'),
            ('unrot', r['unrot'], '--', 'o'),
            ('rot',   r['rot'],   ':',  's'),
        ]:
            valid = ~np.isnan(ps['P1d']) & (ps['P1d'] > 0)
            label = f"{box_len:.0f} Mpc ({tag})" if box_len == sorted(power_spectrum_results.keys())[0] else None
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
             "kSZ_Pk_three_cases_boxsize_all",
             title=rf'kSZ $P(k)$ — Original / Unrotated / Rotated (Cell = {CELL_SIZE_MPC:.2f} Mpc)')
print("✓ Saved: kSZ_Pk_three_cases_boxsize_all")

# =============================================================================
# PLOT 2: D_ℓ — three cases, rainbow by box size  *** MAIN RESULT ***
# =============================================================================

def plot_Dl_three_cases(ax):
    for box_len in sorted(power_spectrum_results.keys()):
        r     = power_spectrum_results[box_len]
        color = cmap(norm(box_len))
        lbl   = f'{box_len:.0f} Mpc'

        for tag, ps, ls, mk in [
            ('orig',  r['orig'],  '-',  '^'),
            ('unrot', r['unrot'], '--', 'o'),
            ('rot',   r['rot'],   ':',  's'),
        ]:
            valid = (~np.isnan(ps['Dl_uK2']) & (ps['Dl_uK2'] > 0)
                     & (ps['ell'] > 10))
            # Only label once per box (on the original line)
            label = lbl if tag == 'orig' else None
            ax.errorbar(
                ps['ell'][valid], ps['Dl_uK2'][valid],
                yerr=ps['Dl_uK2_err_total'][valid],
                color=color, ls=ls, marker=mk,
                markersize=4, linewidth=1.8, alpha=0.75,
                capsize=2, capthick=1, label=label
            )

    # Line-style legend (case labels, box-size-independent)
    import matplotlib.lines as mlines
    orig_l  = mlines.Line2D([], [], color='gray', ls='-',  marker='^', label='Original (full box)')
    unrot_l = mlines.Line2D([], [], color='gray', ls='--', marker='o', label=r'Unrotated ($\theta=0°$)')
    rot_l   = mlines.Line2D([], [], color='gray', ls=':',  marker='s', label=rf'Rotated ($\theta={angle_deg}°$)')
    ax.legend(handles=[orig_l, unrot_l, rot_l], loc='best', fontsize=9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')

save_pdf_png(plot_Dl_three_cases, plot_dir,
             "kSZ_Dl_three_cases_boxsize_all",
             title=rf'kSZ $D_\ell$ — Original / Unrotated / Rotated (Cell = {CELL_SIZE_MPC:.2f} Mpc)')
print("✓ Saved: kSZ_Dl_three_cases_boxsize_all  *** MAIN RESULT ***")

# =============================================================================
# Summary Table
# =============================================================================

print("\n" + "="*70)
print("POWER SPECTRUM SUMMARY — THREE CASES")
print("="*70)
print(f"{'BOX[Mpc]':<10} {'Cell[Mpc]':<10} {'Case':<12} "
      f"{'ell_min':<10} {'ell_max':<10} {'Peak_Dl[μK²]':<16} {'Err[%]':<8}")
print("-" * 80)

for box_len in sorted(power_spectrum_results.keys()):
    r = power_spectrum_results[box_len]
    cell = box_len / r['hii_dim']

    for tag, ps in [('Original', r['orig']),
                    ('Unrotated', r['unrot']),
                    ('Rotated',   r['rot'])]:
        valid = (~np.isnan(ps['Dl_uK2']) & (ps['Dl_uK2'] > 0)
                 & (ps['ell'] > 10))
        if valid.sum() == 0:
            continue
        peak_idx = np.argmax(ps['Dl_uK2'][valid])
        peak_dl  = ps['Dl_uK2'][valid][peak_idx]
        peak_err = ps['Dl_uK2_err_total'][valid][peak_idx]
        print(f"{box_len:<10.0f} {cell:<10.3f} {tag:<12} "
              f"{ps['ell'][valid].min():<10.0f} "
              f"{ps['ell'][valid].max():<10.0f} "
              f"{peak_dl:<16.4e} "
              f"{peak_err/peak_dl*100:<8.1f}")

print("\n" + "="*70)
print("ALL kSZ POWER SPECTRUM ANALYSIS COMPLETE!")
print("="*70)


