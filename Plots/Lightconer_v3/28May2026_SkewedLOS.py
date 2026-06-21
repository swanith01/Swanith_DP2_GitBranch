# %%
# =============================================================================
# LIGHTCONE GENERATION - MINIMAL & CLEAN VERSION
# py21cmfast v3.4
# =============================================================================

# =============================================================================
# CELL 1: Imports
# =============================================================================
import numpy as np
import matplotlib as mpl
#matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import py21cmfast as p21c
from py21cmfast import plotting

import os
from datetime import datetime

print(f"py21cmfast version: {p21c.__version__}")

# =============================================================================
# CELL 1a0: Create Output Directory for Plots
# =============================================================================
plot_dir = "14May2026/plots"

# Create directory if it doesn't exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
else:
    print(f"Directory already exists: {plot_dir}")

print(f"All plots will be saved to: {os.path.abspath(plot_dir)}")


import matplotlib as mpl
import matplotlib.pyplot as plt



# =============================================================================
# CELL 1a: Define Minimal User Parameters
# Only specify what differs from defaults or what's required
# =============================================================================

# Basic simulation parameters
user_params = p21c.UserParams(
    HII_DIM=128,                      # Resolution of ionization/spin-temp grids
    BOX_LEN=800.0,                    # Box size in comoving Mpc
    USE_INTERPOLATION_TABLES=True,    # Speed up calculations
    N_THREADS=32                       # Number of CPU threads
)

# Redshift range for the lightcone
z_min = 5.0                           # Final/lowest redshift
z_max = 20.0                          # Highest redshift to simulate

print("User parameters defined:")
print(user_params)


# =============================================================================
# CELL 1b : Print Default Parameters (Optional - for reference)
# See what py21cmfast uses by default if you don't specify
# =============================================================================

print("\n=== DEFAULT COSMOLOGY ===")
print(p21c.CosmoParams())

print("\n=== DEFAULT ASTROPHYSICS ===")
print(p21c.AstroParams())

print("\n=== DEFAULT FLAGS ===")
print(p21c.FlagOptions())





# %%
# =============================================================================
# SIMPLE LIGHTCONE PLOTTING - NO TICK CUSTOMIZATION
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
    'legend.fontsize': 18,
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

# =============================================================================
# USAGE
# =============================================================================
"""
lightcone_plots = {
    "xH_box": {
        "title": r"Neutral Fraction Lightcone",
        "plot_name": "xHI_lightcone",
        "cbar_label": r"$x_{\mathrm{HI}}$",
        "labelsize": 20,
        "axlabelsize": 27,
        "add_time_axis": True,
    },
}

for field, cfg in lightcone_plots.items():
    plot_func = make_lightcone_plotter(
        lightcone=lightcone,
        field=field,
        cmap=cfg.get("cmap"),
        cbar_label=cfg.get("cbar_label"),
        labelsize=cfg.get("labelsize", 16),
        axlabelsize=cfg.get("axlabelsize", 22),
        user_params=user_params,
        add_time_axis=cfg.get("add_time_axis", False),
    )
    
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
        plot_func(ax)
        plt.show()
    
    save_pdf_png(
        plot_func=plot_func,
        plot_dir=plot_dir,
        plot_name=cfg["plot_name"],
        title=cfg.get("title"),
    )
"""

# %%
# =============================================================================
# CELL 2: Run Lightcone Simulation
# Using defaults for cosmo/astro/flags - only override if needed
# =============================================================================

print("\n=== RUNNING LIGHTCONE ===")
print(f"Redshift range: z = {z_min} → {z_max}")
print(f"Box size: {user_params.BOX_LEN} Mpc")
print(f"Resolution: {user_params.HII_DIM}³ cells")

lightcone = p21c.run_lightcone(
    redshift=z_min,                           # Final redshift
    max_redshift=z_max,                       # Starting redshift
    lightcone_quantities=('brightness_temp', 'density', 'xH_box', 'velocity'),
    #global_quantities=('xH_box'),
    user_params=user_params,
    random_seed=37,                           # For reproducibility
    direc='_cache'                            # Cache directory
)

print("\nLightcone simulation complete!")
print(f"Lightcone shape: {lightcone.brightness_temp.shape}")
print(f"Number of redshift slices: {len(lightcone.lightcone_redshifts)}")


# =============================================================================
# CELL 2a: Inspect Lightcone Structure
# What fields are available and their shapes
# =============================================================================

print("\n=== LIGHTCONE CONTENTS ===")
print(f"Redshift axis: z ∈ [{lightcone.lightcone_redshifts.min():.2f}, "
      f"{lightcone.lightcone_redshifts.max():.2f}]")
print(f"Comoving distance axis: r ∈ [{lightcone.lightcone_distances.min():.1f}, "
      f"{lightcone.lightcone_distances.max():.1f}] Mpc")

print("\nAvailable fields:")
for field in ['brightness_temp', 'density', 'xH_box', 'velocity']:
    if hasattr(lightcone, field):
        data = getattr(lightcone, field)
        print(f"  {field:20s} -> shape {data.shape}, dtype {data.dtype}")


# =============================================================================
# CELL 2b: Extract Redshift and Distance Axes
# Trim to the requested redshift range if needed
# =============================================================================

# Full axes from lightcone
red_axis = lightcone.lightcone_redshifts      # Redshift per slice
pos_axis = lightcone.lightcone_distances      # Comoving distance [Mpc]

# Trim to z <= z_max (in case simulation went slightly beyond)
ind_z = np.where(red_axis <= z_max)[0]
red_axis = red_axis[ind_z]
pos_axis = pos_axis[ind_z]

print(f"\nTrimmed axes:")
print(f"  Redshift: {len(red_axis)} slices from z={red_axis.min():.2f} to z={red_axis.max():.2f}")
print(f"  Distance: {pos_axis.min():.1f} to {pos_axis.max():.1f} Mpc")


# =============================================================================
# CELL 2c: LIGHTCONE PLOTS
# =============================================================================

# Updated lightcone_plots configuration
lightcone_plots = {
    "brightness_temp": {
        "title": "21cm Brightness Temperature",
        "plot_name": "brightness_temp_lightcone",
    },
    "density": {
        "title": r"Matter Density Field ($\delta$)",
        "plot_name": "density_lightcone",
    },
    "velocity": {
        "title": "Line-of-Sight Velocity",
        "plot_name": "velocity_z_lightcone",
        "cmap": "RdBu_r",
        "cbar_label": "Velocity [Mpc/s]",
    },
    "xH_box": {
        "title": r"Neutral Fraction Lightcone",
        "plot_name": "xHI_lightcone",
        "cbar_label": r"Neutral Fraction ($x_{\mathrm{HI}}$)",
        "labelsize": 20,
        "axlabelsize": 27,
        "add_time_axis": False,
    },
}

for field, cfg in lightcone_plots.items():
    plot_func = make_lightcone_plotter(
        lightcone=lightcone,
        field=field,
        cmap=cfg.get("cmap"),
        cbar_label=cfg.get("cbar_label"),
        labelsize=cfg.get("labelsize", 16),
        axlabelsize=cfg.get("axlabelsize", 22),
        user_params=user_params,
        add_time_axis=cfg.get("add_time_axis", False),
    )
    
    # Preview PDF style
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
        plot_func(ax)
        plt.show()
    
    # Save PDF + PNG
    save_pdf_png(
        plot_func=plot_func,
        plot_dir=plot_dir,
        plot_name=cfg["plot_name"],
        title=cfg.get("title") if cfg.get("plot_name") != "xHI_lightcone" else None,  # No title for xHI
    )
    print(f"Saved: {cfg['plot_name']}")
# =============================================================================
# Summary
# =============================================================================
print(f"\n=== ALL PLOTS SAVED ===")
print(f"Location: {os.path.abspath(plot_dir)}")
print(f"Files created:")
for fname in sorted(os.listdir(plot_dir)):
    print(f"  - {fname}")

# %%
# =============================================================================
# CELL 2.5: LOS ROTATION + DIAGNOSTIC SKEWER PLOTS (Fixed)
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

angle_deg = 10
Nlos      = 3000 #Number of skewers of pixel size widh

print(f"\n{'='*60}")
print(f"LOS ROTATION — θ = {angle_deg}° — {Nlos} skewers")
print(f"{'='*60}")

# --- Parameters ------------------------------------------------------------
Ndim      = int(user_params.HII_DIM)
Lbox      = float(user_params.BOX_LEN)
cell_size = Lbox / Ndim
angle_rad = np.deg2rad(angle_deg)
sin_a     = float(np.sin(angle_rad))
cos_a     = float(np.cos(angle_rad))

pos_axis_arr = np.array(pos_axis, dtype=np.float64)
red_axis_arr = np.array(red_axis, dtype=np.float64)
Nbins        = len(pos_axis_arr)
delta_z      = float(pos_axis_arr[1] - pos_axis_arr[0])

# Relative comoving distance — MUST start at 0 for rotation geometry
s_axis = pos_axis_arr - pos_axis_arr[0]

print(f"  Lbox      : {Lbox:.1f} cMpc  |  Ndim = {Ndim}")
print(f"  cell_size : {cell_size:.3f} cMpc")
print(f"  Nbins     : {Nbins}  |  delta_z = {delta_z:.3f} cMpc")
print(f"  s_axis    : {s_axis[0]:.3f} – {s_axis[-1]:.3f} cMpc  (starts at 0)")
print(f"  Artefact period unrotated : {Lbox:.1f} cMpc")
print(f"  Artefact period rotated   : {Lbox/sin_a:.1f} cMpc  ({1/sin_a:.1f}x suppression)")
print(f"  Path-length change        : {(1/cos_a - 1)*100:.2f}%")


# --- Save original lightcone before any patching ---------------------------
# Use _lightcone_original (with underscore) for consistency with downstream cells
try:
    _src = _lightcone_original
    print("\nRe-run detected — using saved _lightcone_original as source.")
except NameError:
    _lightcone_original = lightcone
    _src = _lightcone_original
    print("\nFirst run — saving lightcone as _lightcone_original.")

# Sanity check
_test_shape = np.array(_src.density).shape
print(f"Source field shape: {_test_shape}  (must be Ndim x Ndim x Nbins)")
assert len(_test_shape) == 3 and _test_shape[0] > 1, (
    f"Shape {_test_shape} looks like a wrapper — restart kernel and re-run from Cell 1.")


# --- Rotated skewer function -----------------------------------------------

def _periodic(n, ngrid):
    return int(round(float(n))) % int(ngrid)


def rotated_skewer(field, x_start, y_idx):
    """
    Extract one rotated LOS skewer with linear z-interpolation.
    Uses s_axis (relative comoving distance starting at 0) for geometry.
    """
    Nbins_box = field.shape[2]
    skewer    = np.empty(Nbins, dtype=np.float32)
    for i in range(Nbins):
        s      = s_axis[i]                              # relative distance, starts at 0
        z_cont = s * cos_a / delta_z                    # continuous z index
        x_cont = float(x_start) + s * sin_a / cell_size  # continuous x offset

        z0 = int(np.floor(z_cont))
        z1 = z0 + 1
        fz = z_cont - z0

        z0 = min(max(z0, 0), Nbins_box - 1)
        z1 = min(max(z1, 0), Nbins_box - 1)
        x  = _periodic(x_cont, Ndim)

        skewer[i] = field[x, y_idx, z0] * (1 - fz) + \
                    field[x, y_idx, z1] * fz
    return skewer


# --- Extract 3D fields from original lightcone ----------------------------

_Delta_3d = np.array(_src.density[:, :, ind_z],  dtype=np.float32) + 1.0
_xHI_3d   = np.array(_src.xH_box[:, :, ind_z],   dtype=np.float32)
_vel_3d   = np.array(_src.velocity[:, :, ind_z],  dtype=np.float32)/67.4

print(f"\nField shape: {_Delta_3d.shape}  (Ndim x Ndim x Nbins)")



# --- LOS grid --------------------------------------------------------------

# Cap Nlos to available unique positions on box face
Nlos_max = Ndim * Ndim   # = 64*64 = 4096
if Nlos > Nlos_max:
    print(f"WARNING: Nlos={Nlos} > Ndim²={Nlos_max} — capping to avoid duplicates")
    Nlos = Nlos_max

# Use full grid if Nlos == Ndim*Ndim, otherwise use ceil(sqrt) spacing
if Nlos == Nlos_max:
    # Every cell exactly once
    LOS_ind = np.array([
        [i, j]
        for i in range(Ndim)
        for j in range(Ndim)
    ])   # (4096, 2)
    print(f"LOS grid: {Ndim}×{Ndim} = {Nlos} skewers (full box face)")
else:
    Nlos_perrow = int(np.ceil(np.sqrt(Nlos)))
    ind_step    = int(np.ceil(Ndim / Nlos_perrow))
    LOS_ind = np.array([
        [int((i + 0.5) * ind_step) % Ndim,
         int((j + 0.5) * ind_step) % Ndim]
        for i in range(Nlos_perrow)
        for j in range(Nlos_perrow)
    ])[:Nlos]
    print(f"LOS grid: {Nlos_perrow}×{Nlos_perrow} = {Nlos} skewers")

# Duplicate check
n_unique = len(set(map(tuple, LOS_ind.tolist())))
print(f"Unique positions : {n_unique} / {Nlos}  "
      f"({'OK' if n_unique == Nlos else 'DUPLICATES PRESENT'})")

# --- Extract skewers -------------------------------------------------------

print("Extracting unrotated and rotated skewers...")

density_unrot  = np.zeros((Nlos, Nbins), dtype=np.float32)
xH_box_unrot   = np.zeros((Nlos, Nbins), dtype=np.float32)
velocity_unrot = np.zeros((Nlos, Nbins), dtype=np.float32)

density_rot    = np.zeros((Nlos, Nbins), dtype=np.float32)
xH_box_rot     = np.zeros((Nlos, Nbins), dtype=np.float32)
velocity_rot   = np.zeros((Nlos, Nbins), dtype=np.float32)

for k, (x0, y0) in enumerate(LOS_ind):
    ix, iy = int(x0) % Ndim, int(y0) % Ndim

    # Unrotated — direct z-axis extraction
    density_unrot[k]  = _Delta_3d[ix, iy, :]
    xH_box_unrot[k]   = _xHI_3d[ix,  iy, :]
    velocity_unrot[k] = _vel_3d[ix,   iy, :]

    # Rotated — diagonal interpolated extraction
    density_rot[k]    = rotated_skewer(_Delta_3d, int(x0), int(y0))
    xH_box_rot[k]     = rotated_skewer(_xHI_3d,   int(x0), int(y0))
    velocity_rot[k]   = rotated_skewer(_vel_3d,    int(x0), int(y0))

    if (k + 1) % 200 == 0:
        print(f"  {k+1}/{Nlos} skewers done")

print("  Done.\n")


# --- Store both ensembles --------------------------------------------------

lightcone_unrot = {
    'density'  : density_unrot,
    'xH_box'   : xH_box_unrot,
    'velocity' : velocity_unrot,
    'red_axis' : red_axis_arr,
    'pos_axis' : pos_axis_arr,
    'angle_deg': 0,
    'label'    : 'Unrotated (θ=0°)',
}

lightcone_rot = {
    'density'  : density_rot,
    'xH_box'   : xH_box_rot,
    'velocity' : velocity_rot,
    'red_axis' : red_axis_arr,
    'pos_axis' : pos_axis_arr,
    'angle_deg': angle_deg,
    'label'    : f'Rotated (θ={angle_deg}°)',
}

print(f"Stored: lightcone_unrot  — {Nlos} skewers x {Nbins} bins, θ=0°")
print(f"Stored: lightcone_rot    — {Nlos} skewers x {Nbins} bins, θ={angle_deg}°")


# --- Patch main lightcone --------------------------------------------------

class _RotatedLightcone:
    def __init__(self, original, density_r, xH_r, vel_r):
        self._orig     = original
        self._density  = density_r[np.newaxis, :, :]   # (1, Nlos, Nbins)
        self._xH_box   = xH_r[np.newaxis, :, :]
        self._velocity = vel_r[np.newaxis, :, :]
        # Explicitly copy axis attributes
        self.lightcone_redshifts  = original.lightcone_redshifts
        self.lightcone_distances  = original.lightcone_distances
        self.lightcone_dimensions = original.lightcone_dimensions

    def __getattr__(self, name):
        return getattr(self._orig, name)

    @property
    def density(self):   return self._density
    @property
    def xH_box(self):    return self._xH_box
    @property
    def velocity(self):  return self._velocity


lightcone = _RotatedLightcone(
    _lightcone_original, density_rot, xH_box_rot, velocity_rot,
)

print(f"\nlightcone patched — downstream cells use rotated fields.")
print(f"  .density  : {lightcone.density.shape}")
print(f"  .xH_box   : {lightcone.xH_box.shape}")
print(f"  .velocity : {lightcone.velocity.shape}")


# --- Validation ------------------------------------------------------------

print(f"\n{'='*60}")
print(f"VALIDATION")
print(f"{'='*60}")
print(f"  <x_HI> raw 3D    : {float(np.mean(_xHI_3d)):.4f}")
print(f"  <x_HI> unrotated : {float(np.mean(xH_box_unrot)):.4f}  (should match raw)")
print(f"  <x_HI> rotated   : {float(np.mean(xH_box_rot)):.4f}   (should match raw)")
print(f"  <1+d>  unrotated : {float(np.mean(density_unrot)):.4f}  (should be ~1)")
print(f"  <1+d>  rotated   : {float(np.mean(density_rot)):.4f}   (should be ~1)")
print(f"  <v_z>  unrotated : {float(np.mean(velocity_unrot)):.4e}  (should be ~0)")
print(f"  <v_z>  rotated   : {float(np.mean(velocity_rot)):.4e}   (should be ~0)")
print(f"{'='*60}")


# --- Diagnostic plots ------------------------------------------------------

_ext  = [float(red_axis_arr[0]), float(red_axis_arr[-1]), 0, Nlos]
_vabs = max(float(np.percentile(np.abs(velocity_unrot), 99)),
            float(np.percentile(np.abs(velocity_rot),   99)))

def _make_plot(unrot, rot, cbar_label, cmap, lognorm=False, vmin=None, vmax=None):
    def _plot(ax):
        fig = ax.figure
        fig.clf()
        ax_l = fig.add_subplot(1, 2, 1)
        ax_r = fig.add_subplot(1, 2, 2)

        # Fix: pass norm OR vmin/vmax, never both
        if lognorm:
            kw = dict(norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            kw = dict(vmin=vmin, vmax=vmax)

        base = dict(origin='lower', interpolation='none',
                    aspect='auto', cmap=cmap)

        im_l = ax_l.imshow(unrot, extent=_ext, **base, **kw)
        ax_l.set_title(r'Unrotated ($\theta=0°$)')
        ax_l.set_xlabel('Redshift z')
        ax_l.set_ylabel('LOS index')
        fig.colorbar(im_l, ax=ax_l, label=cbar_label)

        im_r = ax_r.imshow(rot, extent=_ext, **base, **kw)
        ax_r.set_title(rf'Rotated ($\theta={angle_deg}°$)')
        ax_r.set_xlabel('Redshift z')
        fig.colorbar(im_r, ax=ax_r, label=cbar_label)

    return _plot

_panels = [
    ("cell2p5_density",  r"$1+\delta$",
     _make_plot(density_unrot,  density_rot,
                r"$1+\delta$",       "viridis",
                lognorm=True, vmin=5e-2, vmax=6e1)),

    ("cell2p5_xHI",      r"$x_{\rm HI}$",
     _make_plot(xH_box_unrot,   xH_box_rot,
                r"$x_{\rm HI}$",     "magma",
                vmin=0., vmax=1.)),

    ("cell2p5_velocity", r"$v_{\rm LOS}$ [Mpc/s]",
     _make_plot(velocity_unrot, velocity_rot,
                r"$v_{\rm LOS}$ [Mpc/s]", "RdBu_r",
                vmin=-_vabs, vmax=_vabs)),
]

for plot_name, label, plot_func in _panels:
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
        plot_func(ax)
        plt.show()
    save_pdf_png(plot_func=plot_func, plot_dir=plot_dir,
                 plot_name=plot_name,
                 title=f"Unrotated vs Rotated — {label}")
    print(f"Saved: {plot_name}")

print(f"\nAll plots saved to: {plot_dir}")
print("Downstream cells will use rotated fields automatically.")

# %%
# =============================================================================
# CELL 2.6: DIAGNOSTIC — skewer difference from original lightcone
# Unrotated skewers should show ~zero difference from original.
# Rotated skewers should show measurable difference.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

print(f"{'='*60}")
print(f"SKEWER DIFFERENCE DIAGNOSTICS")
print(f"{'='*60}")

# =============================================================================
# Build reference: original lightcone sampled at same (x,y) positions
# Shape (Nlos, Nbins) — direct read, no interpolation
# =============================================================================

density_orig  = np.zeros((Nlos, Nbins), dtype=np.float32)
xHI_orig      = np.zeros((Nlos, Nbins), dtype=np.float32)
vel_orig      = np.zeros((Nlos, Nbins), dtype=np.float32)

for k, (x0, y0) in enumerate(LOS_ind):
    ix, iy = int(x0) % Ndim, int(y0) % Ndim
    density_orig[k] = _Delta_3d[ix, iy, :]
    xHI_orig[k]     = _xHI_3d[ix,  iy, :]
    vel_orig[k]     = _vel_3d[ix,   iy, :]

print(f"Reference (original) extracted: {density_orig.shape}")


# =============================================================================
# Compute differences
# =============================================================================

# Unrotated - Original  (should be ~zero everywhere)
d_density_unrot = density_unrot  - density_orig
d_xHI_unrot     = xH_box_unrot   - xHI_orig
d_vel_unrot     = velocity_unrot  - vel_orig

# Rotated - Original  (should show structure)
d_density_rot   = density_rot    - density_orig
d_xHI_rot       = xH_box_rot     - xHI_orig
d_vel_rot       = velocity_rot    - vel_orig

print(f"\n{'='*60}")
print(f"DIFFERENCE STATISTICS")
print(f"{'='*60}")
for label, du, dr in [
    ("density (1+δ)", d_density_unrot, d_density_rot),
    ("x_HI",         d_xHI_unrot,     d_xHI_rot),
    ("v_LOS",        d_vel_unrot,      d_vel_rot),
]:
    print(f"\n  {label}:")
    print(f"    Unrot - Orig : mean={du.mean():.3e}  "
          f"std={du.std():.3e}  max|Δ|={np.abs(du).max():.3e}")
    print(f"    Rot   - Orig : mean={dr.mean():.3e}  "
          f"std={dr.std():.3e}  max|Δ|={np.abs(dr).max():.3e}")
print(f"{'='*60}")


# =============================================================================
# Plots — 3 rows x 3 columns
# Col 1: Unrotated - Original  (expect ~zero)
# Col 2: Rotated   - Original  (expect structure)
# Col 3: 1D mean profile comparison along redshift
# =============================================================================

_ext  = [float(red_axis_arr[0]), float(red_axis_arr[-1]), 0, Nlos]

_rows = [
    ("1+δ",
     d_density_unrot, d_density_rot,
     density_orig, density_unrot, density_rot,
     "viridis", r"$\Delta(1+\delta)$", r"$1+\delta$"),

    (r"$x_{\rm HI}$",
     d_xHI_unrot, d_xHI_rot,
     xHI_orig, xH_box_unrot, xH_box_rot,
     "magma", r"$\Delta x_{\rm HI}$", r"$x_{\rm HI}$"),

    (r"$v_{\rm LOS}$",
     d_vel_unrot, d_vel_rot,
     vel_orig, velocity_unrot, velocity_rot,
     "RdBu_r", r"$\Delta v_{\rm LOS}$ [Mpc/s]", r"$v_{\rm LOS}$ [Mpc/s]"),
]


def plot_differences(ax):
    fig = ax.figure
    fig.clf()

    axes = fig.subplots(3, 3)

    for row, (label, du, dr, orig, unrot, rot, cmap, dlabel, flabel) in \
            enumerate(_rows):

        # Symmetric colour limit for difference panels
        vd = float(np.percentile(
            np.abs(np.concatenate([du.ravel(), dr.ravel()])), 99))

        base = dict(origin='lower', interpolation='none',
                    aspect='auto', extent=_ext)

        # Col 0: Unrotated - Original
        im0 = axes[row, 0].imshow(du, cmap='coolwarm',
                                   vmin=-vd, vmax=vd, **base)
        axes[row, 0].set_ylabel('LOS index')
        if row == 0:
            axes[row, 0].set_title('Unrotated − Original\n(expect ~zero)')
        fig.colorbar(im0, ax=axes[row, 0], label=dlabel)

        # Col 1: Rotated - Original
        im1 = axes[row, 1].imshow(dr, cmap='coolwarm',
                                   vmin=-vd, vmax=vd, **base)
        if row == 0:
            axes[row, 1].set_title(
                f'Rotated (θ={angle_deg}°) − Original\n(expect structure)')
        fig.colorbar(im1, ax=axes[row, 1], label=dlabel)

        # Col 2: Mean profile along redshift
        ax2 = axes[row, 2]
        ax2.plot(red_axis_arr, np.mean(orig,  axis=0),
                 color='black',      lw=2,   label='Original')
        ax2.plot(red_axis_arr, np.mean(unrot, axis=0),
                 color='steelblue',  lw=1.5, ls='--', label='Unrotated')
        ax2.plot(red_axis_arr, np.mean(rot,   axis=0),
                 color='darkorange', lw=1.5, ls=':',
                 label=f'Rotated (θ={angle_deg}°)')
        ax2.set_ylabel(flabel)
        ax2.invert_xaxis()
        if row == 0:
            ax2.set_title('Mean profile vs redshift')
            ax2.legend(loc='best')

        # x-axis labels on bottom row only
        for col in range(3):
            if row == 2:
                axes[row, col].set_xlabel('Redshift z')
            else:
                axes[row, col].set_xlabel('')

        # Row label
        axes[row, 0].set_ylabel(f'{label}\nLOS index')


with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(20, 14), constrained_layout=True)
    plot_differences(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_differences,
    plot_dir=plot_dir,
    plot_name="cell2p6_skewer_difference_diagnostic",
    title="Skewer difference from original lightcone",
)

print(f"\nSaved: cell2p6_skewer_difference_diagnostic")
print(f"\nWhat to look for:")
print(f"  Col 1 (Unrot - Orig) : should be uniformly ~zero (black/white)")
print(f"  Col 2 (Rot - Orig)   : should show coloured structure")
print(f"  Col 3 (mean profiles): Orig and Unrot should overlap exactly,")
print(f"                         Rotated may differ slightly at low z")

# %%
# =============================================================================
# CELL 3: Extract 3D Fields (if needed for further analysis)
# Trimmed to the redshift range of interest
# =============================================================================

# Extract fields with proper indexing
brightness_temp = lightcone.brightness_temp[:, :, ind_z]  # [mK]
density_field   = 1+ lightcone.density[:, :, ind_z]          # 1 + δ
neutral_frac    = lightcone.xH_box[:, :, ind_z]           # xHI
velocity_los    = 1e17*lightcone.velocity[:, :, ind_z]/67.4        # [comoving km/s]

print("\n=== EXTRACTED 3D FIELDS ===")
print(f"Brightness temperature: {brightness_temp.shape}")
print(f"Density field:          {density_field.shape}")
print(f"Neutral fraction:       {neutral_frac.shape}")


# =============================================================================
# CELL 3a: Summary Statistics (Optional)
# Quick overview of the simulation results
# =============================================================================

print("\n=== SUMMARY STATISTICS ===")
print(f"Brightness temperature: min={brightness_temp.min():.1f} mK, "
      f"max={brightness_temp.max():.1f} mK, mean={brightness_temp.mean():.1f} mK")
print(f"Neutral fraction:       min={neutral_frac.min():.3f}, "
      f"max={neutral_frac.max():.3f}, mean={neutral_frac.mean():.3f}")
print(f"Density (1+δ):          min={density_field.min():.3f}, "
      f"max={density_field.max():.3f}, mean={density_field.mean():.3f}")
print(f"Velocity_z (1e-17 Mpc/s):          min={velocity_los.min():.3f}, "
      f"max={velocity_los.max():.3f}, mean={velocity_los.mean():.3f}")

# =============================================================================
# NOTES ON CUSTOMIZATION:
# 
# If you want to change cosmology/astrophysics from defaults, add:
#
#   cosmo_params = p21c.CosmoParams(
#       SIGMA_8=0.81,      # Matter power spectrum normalization
#       hlittle=0.68,      # Hubble parameter h
#       OMm=0.31,          # Total matter density
#       OMb=0.049,         # Baryon density
#       POWER_INDEX=0.965  # Primordial spectral index n_s
#   )
#
#   astro_params = p21c.AstroParams(
#       HII_EFF_FACTOR=30.0,  # Ionization efficiency
#       R_BUBBLE_MAX=30.0,    # Maximum bubble size [Mpc]
#       L_X=1e40              # X-ray luminosity per SFR
#   )
#
#   flag_options = p21c.FlagOptions(
#       USE_TS_FLUCT=True     # Include spin temperature fluctuations
#   )
#
# Then pass them to run_lightcone():
#   lightcone = p21c.run_lightcone(
#       ...,
#       cosmo_params=cosmo_params,
#       astro_params=astro_params,
#       flag_options=flag_options
#   )
# =============================================================================
# =============================================================================
# CELL 3.5: Summary statistics for rotated skewer arrays
# Mirrors Cell 3 but operates on the (Nlos, Nbins) skewer dicts
# produced in Cell 2.5 — confirms rotation preserved field statistics.
# =============================================================================

import numpy as np

print("=== ROTATED SKEWER STATISTICS ===\n")

for lc, tag in [(lightcone_unrot, "Unrotated (θ=0°)"), 
                (lightcone_rot,   f"Rotated   (θ={angle_deg}°)")]:

    d   = lc['density']           # (Nlos, Nbins)  — already 1+δ
    xHI = lc['xH_box']            # (Nlos, Nbins)
    vel = 1e17 * lc['velocity']   # scaled same as Cell 3

    print(f"--- {tag} ---")
    print(f"  Shape                : {d.shape}  (Nlos × Nbins)")
    print(f"  Density (1+δ)        : min={d.min():.3f},  "
          f"max={d.max():.3f},  mean={d.mean():.3f}")
    print(f"  Neutral fraction     : min={xHI.min():.3f},  "
          f"max={xHI.max():.3f},  mean={xHI.mean():.3f}")
    print(f"  Velocity (×1e17)     : min={vel.min():.3f},  "
          f"max={vel.max():.3f},  mean={vel.mean():.4e}")
    print(f"  Ionised fraction x_e : mean={1 - xHI.mean():.3f}")
    print()

# Cross-check: rotated mean should match unrotated mean to ~1%
d_diff   = abs(lightcone_rot['density'].mean()  - lightcone_unrot['density'].mean())
xhi_diff = abs(lightcone_rot['xH_box'].mean()   - lightcone_unrot['xH_box'].mean())
vel_diff = abs(lightcone_rot['velocity'].mean()  - lightcone_unrot['velocity'].mean())

print("=== ROTATION CONSISTENCY CHECK ===")
print(f"  Δ<1+δ>   : {d_diff:.2e}  (should be < 0.01)")
print(f"  Δ<x_HI>  : {xhi_diff:.2e}  (should be < 0.01)")
print(f"  Δ<v_z>   : {vel_diff:.2e}  (should be ~0, both near zero)")

all_ok = d_diff < 0.01 and xhi_diff < 0.01
print(f"\n  {'✓ Statistics consistent — rotation looks correct.' if all_ok else '✗ Warning: means differ by more than 1% — check rotation.'}")

# %%
# =============================================================================
# COMPLETE CELL 4: Calculate and Plot s, ds, x_e, dτ, τ vs z
# Using interpolated x_e onto lightcone redshifts
# =============================================================================

# =============================================================================
# Step 1: Interpolate x_e onto lightcone redshifts
# =============================================================================

# Reverse node arrays to match lightcone order (low z → high z)
z_nodes_sorted = lightcone.node_redshifts[::-1]      # 5.0 → 20.3
xHI_nodes_sorted = lightcone.global_xH[::-1]
x_e_nodes_sorted = 1.0 - xHI_nodes_sorted

# Interpolate x_e onto lightcone redshift grid
x_e_interp = np.interp(red_axis, z_nodes_sorted, x_e_nodes_sorted)

print(f"\n=== INTERPOLATED IONIZATION FRACTION ===")
print(f"x_e range: [{x_e_interp.min():.4f}, {x_e_interp.max():.4f}]")

# =============================================================================
# Step 2: Extract/Calculate geometric quantities
# =============================================================================

# Comoving distance (already available)
s = pos_axis  # Comoving distance [Mpc]

# Distance element
ds = np.diff(s)  # [Mpc], length 963

# Midpoint values for integration
z_mid = 0.5 * (red_axis[:-1] + red_axis[1:])
x_e_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])

print(f"\n=== GEOMETRIC QUANTITIES ===")
print(f"s range: [{s.min():.1f}, {s.max():.1f}] Mpc")
print(f"ds: mean = {ds.mean():.3f} Mpc, std = {ds.std():.3f} Mpc")
print(f"Number of slices: {len(red_axis)}")

# =============================================================================
# Step 3: Calculate dτ and τ
# =============================================================================
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
# Calculate dτ
dtau = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds

# Cumulative optical depth
tau = np.cumsum(dtau)
tau_total = tau[-1]

print(f"\n=== OPTICAL DEPTH ===")
print(f"dτ range: [{dtau.min():.6e}, {dtau.max():.6e}]")
print(f"Total τ: {tau_total:.6f}")
print(f"Integrated from z = {red_axis.min():.2f} → {red_axis.max():.2f}")

# =============================================================================
# PLOT 1: Comoving Distance s vs z
# =============================================================================

def plot_s_vs_z(ax):
    ax.plot(red_axis, s, '-', linewidth=2, color='darkgreen')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Comoving Distance $s$ [Mpc]')
    ax.invert_xaxis()

plot_name = "s_vs_z"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_s_vs_z(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_s_vs_z,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title="Comoving Distance vs Redshift"
)

print(f"Saved: {plot_name}")


# =============================================================================
# PLOT 2: Distance Element ds vs z
# =============================================================================

def plot_ds_vs_z(ax):
    ax.plot(z_mid, ds, 'o-', linewidth=2, markersize=4, color='darkblue')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Distance Element $ds$ [Mpc]')
    ax.invert_xaxis()

plot_name = "ds_vs_z"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_ds_vs_z(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_ds_vs_z,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title="Distance Element vs Redshift"
)

print(f"Saved: {plot_name}")


# =============================================================================
# PLOT 3: Ionization Fraction x_e vs z (Extrapolated & Saturated Overlays)
# =============================================================================
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Find z where x_e = 0.5 (midpoint) for simulation
interp_z_from_xe = interp1d(x_e_interp, red_axis, bounds_error=False, fill_value='extrapolate')
z_re_sim = float(interp_z_from_xe(0.5))

def process_georgiev(df, is_rapid=False):
    """Extrapolates CSV data to xe=1 and saturates."""
    z_raw = df.iloc[:,0].values
    xe_raw = df.iloc[:,1].values
    
    # Sort by redshift descending (standard lightcone order)
    idx = np.argsort(z_raw)[::-1]
    z, xe = z_raw[idx], xe_raw[idx]

    if is_rapid:
        # Force a step function at z=7
        z_new = np.linspace(red_axis.min(), red_axis.max(), 500)
        xe_new = np.where(z_new <= 6.5, 1.0, 0.0)
        return z_new, xe_new

    # For Mid and Slow: Find where to extrapolate to xe=1
    # Use last two points to find slope (dz/dxe)
    if xe[-1] < 1.0:
        dz_dxe = (z[-1] - z[-2]) / (xe[-1] - xe[-2])
        z_at_one = z[-1] + dz_dxe * (1.0 - xe[-1])
        
        # Append completion point
        z = np.append(z, z_at_one)
        xe = np.append(xe, 1.0)

    # Create saturation range (to z=0) and neutral range (to z_max)
    z_final = np.linspace(red_axis.min(), red_axis.max(), 1000)
    xe_interp_func = interp1d(z, xe, bounds_error=False, fill_value=(1.0, 0.0))
    xe_final = np.clip(xe_interp_func(z_final), 0, 1)
    
    return z_final, xe_final

def plot_xe_vs_z(ax):
    # 1. Plot Simulation Data
    ax.plot(red_axis, x_e_interp, '-', linewidth=4, color='darkblue', label='Interpolated History', zorder=5)

    # 2. Overlay Georgiev+24 (Processed)
    georgiev_configs = {
        'Georgiev_zend_slow.csv':  ('gray',      '--', 'Georgiev+24 (Slow)', False),
        'Georgiev_zend_mid.csv':   ('black',     '-.', 'Georgiev+24 (Mid)', False),
        'Georgiev_rapid_zend.csv': ('darkred',   ':',  'Georgiev+24 (Rapid)', True)
    }

    for filename, (color, ls, label, rapid_flag) in georgiev_configs.items():
        try:
            df = pd.read_csv(filename)
            z_proc, xe_proc = process_georgiev(df, is_rapid=rapid_flag)
            ax.plot(z_proc, xe_proc, color=color, linestyle=ls, label=label, linewidth=2, alpha=0.9)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    # Vertical line for simulation midpoint
    ax.axvline(z_re_sim, color='black', linestyle='--', linewidth=1, label=f'Sim $z_{{re}}={z_re_sim:.1f}$')
    
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'Ionization Fraction $x_e$')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(5, 20) # Focusing on the reionization window
    ax.legend(loc='lower left', frameon=True)
    #ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

# -------- Execution --------
plot_name = "xe_vs_z_extrapolated"
with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(11, 8), constrained_layout=True)
    plot_xe_vs_z(ax)
    plt.show()
    
# =============================================================================
# PLOT 4: Optical Depth Element dτ vs z
# =============================================================================

def plot_dtau_vs_z(ax):
    ax.plot(z_mid, dtau, 'o-', linewidth=2, markersize=4, color='darkred')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Optical Depth Element $d\tau$')
    ax.invert_xaxis()

plot_name = "dtau_vs_z"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_dtau_vs_z(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_dtau_vs_z,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"Optical Depth Element $d\tau$ vs Redshift"
)

print(f"Saved: {plot_name}")


# =============================================================================
# PLOT 5: Cumulative Optical Depth τ vs z
# =============================================================================

def plot_tau_vs_z(ax):
    ax.plot(z_mid, tau, 'o-', linewidth=2.5, markersize=4, color='darkred')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$')
    ax.invert_xaxis()

plot_name = "tau_vs_z"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_tau_vs_z(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_tau_vs_z,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title="Cumulative Optical Depth vs Redshift"
)

print(f"Saved: {plot_name}")


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"SUMMARY OF CALCULATIONS")
print(f"{'='*60}")
print(f"Redshift range:           {red_axis.min():.2f} → {red_axis.max():.2f}")
print(f"Comoving distance:        {s.min():.1f} → {s.max():.1f} Mpc")
print(f"Mean distance element:    {ds.mean():.3f} Mpc")
print(f"Ionization fraction:      {x_e_interp.min():.4f} → {x_e_interp.max():.4f}")
print(f"Total optical depth:      τ = {tau_total:.6f}")
print(f"Number of data points:    {len(red_axis)}")
print(f"{'='*60}")

# %%
# =============================================================================
# SKEWERS
# CELL 4.5: Compare global x_e, dτ, τ, visibility for unrotated vs rotated
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Strip astropy units from all Cell 4 quantities before use
z_mid      = np.array(z_mid,      dtype=np.float64)
ds         = np.array(ds,         dtype=np.float64)
tau_global_cell4 = np.array(tau,  dtype=np.float64)   # save Cell 4 tau under new name
dtau_cell4       = np.array(dtau, dtype=np.float64)
prefactor_f      = float(prefactor)
red_axis_np      = np.array(red_axis, dtype=np.float64)

# =============================================================================
# Skewer-averaged x_e(z) for each ensemble
# =============================================================================

x_e_unrot_skewer = 1.0 - np.mean(lightcone_unrot['xH_box'], axis=0)  # (Nbins,)
x_e_rot_skewer   = 1.0 - np.mean(lightcone_rot['xH_box'],   axis=0)  # (Nbins,)
x_e_global       = np.array(x_e_interp, dtype=np.float64)             # (Nbins,) from Cell 4

print("=== SKEWER-AVERAGED IONIZATION FRACTION ===")
print(f"  x_e global    : mean={x_e_global.mean():.4f}")
print(f"  x_e unrotated : mean={x_e_unrot_skewer.mean():.4f}  "
      f"(Δ={x_e_unrot_skewer.mean()-x_e_global.mean():+.4e})")
print(f"  x_e rotated   : mean={x_e_rot_skewer.mean():.4f}  "
      f"(Δ={x_e_rot_skewer.mean()-x_e_global.mean():+.4e})")


# =============================================================================
# dτ and τ for each
# =============================================================================

def compute_tau(x_e_arr, z_mid, ds, prefactor):
    x_e_mid = 0.5 * (x_e_arr[:-1] + x_e_arr[1:])
    dtau    = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds
    tau     = np.cumsum(dtau)
    return dtau, tau

dtau_global, tau_global = compute_tau(x_e_global,       z_mid, ds, prefactor_f)
dtau_unrot,  tau_unrot  = compute_tau(x_e_unrot_skewer,  z_mid, ds, prefactor_f)
dtau_rot,    tau_rot    = compute_tau(x_e_rot_skewer,    z_mid, ds, prefactor_f)

# All plain numpy now — np.exp will work fine
visibility_global = np.exp(-tau_global)
visibility_unrot  = np.exp(-tau_unrot)
visibility_rot    = np.exp(-tau_rot)

# Update `visibility` to rotated version for Cell 5
visibility = visibility_rot

print(f"\n=== CUMULATIVE OPTICAL DEPTH ===")
print(f"  τ global    : {tau_global[-1]:.6f}")
print(f"  τ unrotated : {tau_unrot[-1]:.6f}  "
      f"({(tau_unrot[-1]/tau_global[-1]-1)*100:+.4f}%)")
print(f"  τ rotated   : {tau_rot[-1]:.6f}  "
      f"({(tau_rot[-1]/tau_global[-1]-1)*100:+.4f}%)")
print(f"\n`visibility` updated → rotated version active for Cell 5.")
print(f"`visibility_unrot` and `visibility_rot` stored for Cell 5.5.")


# =============================================================================
# Plots
# =============================================================================

def plot_xe_comparison(ax):
    ax.plot(red_axis_np, x_e_global,
            color='black',      lw=2,   label='Global (volume avg)')
    ax.plot(red_axis_np, x_e_unrot_skewer,
            color='steelblue',  lw=1.8, ls='--',
            label=r'Unrotated ($\theta=0°$)')
    ax.plot(red_axis_np, x_e_rot_skewer,
            color='darkorange', lw=1.8, ls=':',
            label=rf'Rotated ($\theta={angle_deg}°$)')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$x_e(z)$')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    ax.invert_xaxis()

def plot_tau_comparison(ax):
    ax.plot(z_mid, tau_global,
            color='black',      lw=2,
            label=f'Global  τ={tau_global[-1]:.4f}')
    ax.plot(z_mid, tau_unrot,
            color='steelblue',  lw=1.8, ls='--',
            label=f'Unrotated τ={tau_unrot[-1]:.4f}')
    ax.plot(z_mid, tau_rot,
            color='darkorange', lw=1.8, ls=':',
            label=f'Rotated τ={tau_rot[-1]:.4f}')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Cumulative $\tau(<z)$')
    ax.legend(loc='best')
    ax.invert_xaxis()

def plot_visibility_comparison(ax):
    ax.plot(z_mid, visibility_global,
            color='black',      lw=2,   label='Global')
    ax.plot(z_mid, visibility_unrot,
            color='steelblue',  lw=1.8, ls='--', label='Unrotated')
    ax.plot(z_mid, visibility_rot,
            color='darkorange', lw=1.8, ls=':',
            label=rf'Rotated ($\theta={angle_deg}°$)')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Visibility $e^{-\tau}$')
    ax.legend(loc='best')
    ax.invert_xaxis()

def plot_dtau_comparison(ax):
    ax.plot(z_mid, dtau_global,
            color='black',      lw=2,   label='Global')
    ax.plot(z_mid, dtau_unrot,
            color='steelblue',  lw=1.8, ls='--', label='Unrotated')
    ax.plot(z_mid, dtau_rot,
            color='darkorange', lw=1.8, ls=':',
            label=rf'Rotated ($\theta={angle_deg}°$)')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$d\tau$')
    ax.legend(loc='best')
    ax.invert_xaxis()

_plots = [
    ("cell4p5_xe_comparison",
     plot_xe_comparison,
     r"Ionization fraction $x_e$: global vs skewer-averaged"),
    ("cell4p5_tau_comparison",
     plot_tau_comparison,
     r"Optical depth $\tau$: global vs skewer-averaged"),
    ("cell4p5_visibility_comparison",
     plot_visibility_comparison,
     r"Visibility $e^{-\tau}$: global vs skewer-averaged"),
    ("cell4p5_dtau_comparison",
     plot_dtau_comparison,
     r"$d\tau$: global vs skewer-averaged"),
]

for plot_name, plot_func, title in _plots:
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        plt.show()
    save_pdf_png(plot_func=plot_func, plot_dir=plot_dir,
                 plot_name=plot_name, title=title)
    print(f"Saved: {plot_name}")

# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  τ global    : {tau_global[-1]:.6f}")
print(f"  τ unrotated : {tau_unrot[-1]:.6f}  "
      f"({(tau_unrot[-1]/tau_global[-1]-1)*100:+.4f}%)")
print(f"  τ rotated   : {tau_rot[-1]:.6f}  "
      f"({(tau_rot[-1]/tau_global[-1]-1)*100:+.4f}%)")
print(f"\nExpect differences < 1% — finite {Nlos}-skewer sampling vs full volume.")
print(f"{'='*60}")

# %%
# =============================================================================
# CELL 5: Compute kSZ Integrand with Visibility Function
# kSZ integrand = (1 + δ) × x_e × v_z / c × e^(-τ(z))
# =============================================================================
# Extract fields (already trimmed to ind_z)
density_1plus = 1 + lightcone.density[:, :, ind_z]  # 1 + δ
x_HII_field = 1 - lightcone.xH_box[:, :, ind_z]     # LOCAL ionized fraction
v_los_Mpc_s = lightcone.velocity[:, :, ind_z]/67.4       # Velocity in Mpc/s

# Speed of light in Mpc/s
c_Mpc_s = 299792.458 / 3.08567758e19  # Mpc/s
print(f"Speed of light: c = {c_Mpc_s:.6e} Mpc/s")

# =============================================================================
# Normalize x_e field: x_e = (1-x_HI) / <(1+δ)(1-x_HI)>
# =============================================================================
# Compute density-weighted mean ionized fraction at each redshift slice
x_e_mean_per_slice = np.mean(density_1plus * x_HII_field, axis=(0, 1))  # Shape: (964,)

# Normalized x_e for kSZ integral
x_e_3D = x_HII_field #/ x_e_mean_per_slice[None, None, :]  # Shape: (128, 128, 964)

print(f"\n=== NORMALIZED x_e FIELD ===")
print(f"<(1+δ)·(1-x_HI)> range: [{x_e_mean_per_slice.min():.4f}, {x_e_mean_per_slice.max():.4f}]")
print(f"x_e_3D range: [{x_e_3D.min():.4f}, {x_e_3D.max():.4f}]")
verification_per_slice = np.mean(density_1plus * x_e_3D, axis=(0, 1))
print(f"Verification <(1+δ)·x_e> per slice: min={verification_per_slice.min():.6f}, max={verification_per_slice.max():.6f}")
print(f"All should be ≈1.0")
# =============================================================================
# Interpolate τ(z) onto lightcone redshifts
# =============================================================================

# Ensure tau is dimensionless (strip any units)
tau_dimensionless = np.asarray(tau, dtype=float)

# tau is defined at z_mid (midpoints), need to interpolate to red_axis
# Extend tau to match red_axis length by prepending 0 (at lowest z)
tau_extended = np.concatenate([[0], tau_dimensionless])

# Interpolate tau onto red_axis (trimmed)
tau_at_lightcone = np.interp(red_axis, 
                              np.concatenate([[red_axis[0]], z_mid]), 
                              tau_extended)

print(f"\n=== OPTICAL DEPTH AT LIGHTCONE REDSHIFTS ===")
print(f"τ range: [{tau_at_lightcone.min():.6f}, {tau_at_lightcone.max():.6f}]")
print(f"τ is dimensionless: {type(tau_at_lightcone)}")

# Visibility function e^(-τ)
visibility = np.exp(-tau_at_lightcone)

print(f"\n=== VISIBILITY FUNCTION ===")
print(f"e^(-τ) range: [{visibility.min():.6f}, {visibility.max():.6f}]")

# Broadcast visibility to 3D for multiplication with lightcone fields
# visibility shape: (964,) -> need to reshape to (1, 1, 964) for broadcasting
visibility_3D = visibility[None, None, :]  # Shape (1, 1, 964)

# =============================================================================
# Compute kSZ integrand WITHOUT visibility (for comparison)
# =============================================================================

kSZ_integrand_no_vis = density_1plus * x_e_3D * v_los_Mpc_s / c_Mpc_s

print(f"\n=== kSZ INTEGRAND (no visibility) STATISTICS ===")
print(f"  Mean: {kSZ_integrand_no_vis.mean():.4e}")
print(f"  Std:  {kSZ_integrand_no_vis.std():.4e}")
print(f"  Min:  {kSZ_integrand_no_vis.min():.4e}")
print(f"  Max:  {kSZ_integrand_no_vis.max():.4e}")
print(f"  RMS:  {np.sqrt(np.mean(kSZ_integrand_no_vis**2)):.4e}")

# =============================================================================
# Compute kSZ integrand WITH visibility function
# =============================================================================

kSZ_integrand_with_vis = density_1plus * x_e_3D * v_los_Mpc_s / c_Mpc_s * visibility_3D

print(f"\n=== kSZ INTEGRAND (with visibility) STATISTICS ===")
print(f"  Mean: {kSZ_integrand_with_vis.mean():.4e}")
print(f"  Std:  {kSZ_integrand_with_vis.std():.4e}")
print(f"  Min:  {kSZ_integrand_with_vis.min():.4e}")
print(f"  Max:  {kSZ_integrand_with_vis.max():.4e}")
print(f"  RMS:  {np.sqrt(np.mean(kSZ_integrand_with_vis**2)):.4e}")

# Add to lightcone object for easy access
# CORRECT — attaches to original, __getattr__ on wrapper will find it
lightcone.kSZ_integrand_no_vis = kSZ_integrand_no_vis
lightcone.kSZ_integrand        = kSZ_integrand_with_vis
lightcone.visibility           = visibility_3D
# =============================================================================
# kSZ LIGHTCONE PLOTS
# =============================================================================

vmax_no_vis = np.percentile(np.abs(kSZ_integrand_no_vis), 99)
vmax_vis = np.percentile(np.abs(kSZ_integrand_with_vis), 99)

ksz_plots = {
    "kSZ_integrand_no_vis": {
        "title": r"kSZ Integrand (no visibility): $(1+\delta)\,x_e\,v_z/c$",
        "plot_name": "kSZ_integrand_no_visibility",
        "clim": (-vmax_no_vis, vmax_no_vis),
    },
    "kSZ_integrand": {
        "title": r"kSZ Integrand",
        "plot_name": "kSZ_integrand_with_visibility",
        "clim": (-vmax_vis, vmax_vis),
        "cbar_label": r'$(1+\delta)\,x_e\,\frac{v_z}{c}\,e^{-\tau(z)}$',
    },
}

for field, cfg in ksz_plots.items():
    plot_func = make_lightcone_plotter(
        lightcone=lightcone,   # <-- was `lightcone`
        field=field,
        cmap="seismic_r",
        clim=cfg["clim"],
        cbar_label=cfg.get("cbar_label"),
        user_params=user_params,
    )

    # Preview PDF style
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
        plot_func(ax)
        plt.show()

    # Save PDF + PNG
    save_pdf_png(
        plot_func=plot_func,
        plot_dir=plot_dir,
        plot_name=cfg["plot_name"],
        title=cfg["title"],
    )

    print(f"Saved: {cfg['plot_name']}")


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"kSZ INTEGRAND SUMMARY")
print(f"{'='*60}")
print(f"Effect of visibility function:")
print(f"  RMS without e^(-τ): {np.sqrt(np.mean(kSZ_integrand_no_vis**2)):.4e}")
print(f"  RMS with e^(-τ):    {np.sqrt(np.mean(kSZ_integrand_with_vis**2)):.4e}")
print(f"  Suppression factor: {np.sqrt(np.mean(kSZ_integrand_with_vis**2)) / np.sqrt(np.mean(kSZ_integrand_no_vis**2)):.4f}")
print(f"{'='*60}")

# %%
# =============================================================================
# CELL 5.5: Compare kSZ integrand — unrotated vs rotated lightcones
# Mirrors Cell 5 exactly but for both ensembles side by side.
# Power spectrum comparison comes later.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

c_Mpc_s = 299792.458 / 3.08567758e19   # Mpc/s

# =============================================================================
# Visibility interpolated onto red_axis for each ensemble
# (same logic as Cell 5)
# =============================================================================

def make_visibility_1d(tau_arr, red_axis_np, z_mid):
    tau_dim      = np.array(tau_arr, dtype=np.float64)
    tau_extended = np.concatenate([[0], tau_dim])
    tau_at_lc    = np.interp(red_axis_np,
                              np.concatenate([[red_axis_np[0]], z_mid]),
                              tau_extended)
    return np.exp(-tau_at_lc)   # (Nbins,)

vis_unrot_1d = make_visibility_1d(tau_unrot,  red_axis_np, z_mid)
vis_rot_1d   = make_visibility_1d(tau_rot,    red_axis_np, z_mid)


# =============================================================================
# Compute integrands — shape (Nlos, Nbins)
# =============================================================================

def ksz_no_vis(lc_dict):
    d   = np.array(lc_dict['density'],  dtype=np.float64)   # 1+δ
    x_e = 1.0 - np.array(lc_dict['xH_box'],  dtype=np.float64)
    v   = np.array(lc_dict['velocity'], dtype=np.float64)
    return d * x_e * v / c_Mpc_s

def ksz_with_vis(lc_dict, vis_1d):
    return ksz_no_vis(lc_dict) * vis_1d[np.newaxis, :]   # broadcast (1, Nbins)


ksz_unrot_nv  = ksz_no_vis(lightcone_unrot)
ksz_rot_nv    = ksz_no_vis(lightcone_rot)
ksz_unrot_vis = ksz_with_vis(lightcone_unrot, vis_unrot_1d)
ksz_rot_vis   = ksz_with_vis(lightcone_rot,   vis_rot_1d)

print(f"=== kSZ INTEGRAND STATISTICS ===")
for label, arr in [
    ("Unrotated  no vis ", ksz_unrot_nv),
    ("Rotated    no vis ", ksz_rot_nv),
    ("Unrotated with vis", ksz_unrot_vis),
    ("Rotated   with vis", ksz_rot_vis),
]:
    print(f"  {label} :  RMS={np.sqrt(np.mean(arr**2)):.4e}  "
          f"mean={arr.mean():.4e}  "
          f"min={arr.min():.4e}  max={arr.max():.4e}")


# =============================================================================
# Plots — direct imshow, same style as Cell 2.5 comparison
# =============================================================================

_ext = [float(red_axis_np[0]), float(red_axis_np[-1]), 0, Nlos]

def _symclim(arr, pct=99):
    v = float(np.percentile(np.abs(arr), pct))
    return -v, v

_clim_nv  = _symclim(np.concatenate([ksz_unrot_nv.ravel(),  ksz_rot_nv.ravel()]))
_clim_vis = _symclim(np.concatenate([ksz_unrot_vis.ravel(), ksz_rot_vis.ravel()]))

_panels = [
    # (left title,           left data,     right title,          right data,    clim)
    ("Unrotated — no vis",   ksz_unrot_nv,
     f"Rotated (θ={angle_deg}°) — no vis",  ksz_rot_nv,   _clim_nv),

    ("Unrotated — with vis", ksz_unrot_vis,
     f"Rotated (θ={angle_deg}°) — with vis", ksz_rot_vis, _clim_vis),
]

base = dict(origin="lower", interpolation="none", aspect="auto", cmap="seismic_r")

for plot_name, (tl, dl, tr, dr, clim) in zip(
    ["cell5p5_ksz_no_visibility", "cell5p5_ksz_with_visibility"],
    _panels,
):
    def _plot(ax, _tl=tl, _dl=dl, _tr=tr, _dr=dr, _clim=clim,
              _pn=plot_name):
        # Two-panel figure inside the plot_func
        fig = ax.figure
        # Clear the single ax and replace with two side-by-side
        fig.clf()
        ax_l = fig.add_subplot(1, 2, 1)
        ax_r = fig.add_subplot(1, 2, 2)

        im_l = ax_l.imshow(_dl, extent=_ext, vmin=_clim[0], vmax=_clim[1], **base)
        ax_l.set_title(_tl)
        ax_l.set_xlabel("Redshift z")
        ax_l.set_ylabel("LOS index")
        fig.colorbar(im_l, ax=ax_l, pad=0.02,
                     label=r'$(1+\delta)\,x_e\,v_z/c$')

        im_r = ax_r.imshow(_dr, extent=_ext, vmin=_clim[0], vmax=_clim[1], **base)
        ax_r.set_title(_tr)
        ax_r.set_xlabel("Redshift z")
        ax_r.set_ylabel("LOS index")
        fig.colorbar(im_r, ax=ax_r, pad=0.02,
                     label=r'$(1+\delta)\,x_e\,v_z/c$')

    title = ("kSZ integrand — no visibility" if "no" in plot_name
             else r"kSZ integrand — with visibility $e^{-\tau}$")

    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(20, 7), constrained_layout=True)
        _plot(ax)
        plt.show()

    save_pdf_png(plot_func=_plot, plot_dir=plot_dir,
                 plot_name=plot_name, title=title)
    print(f"Saved: {plot_name}")



# %%
# =============================================================================
# CELL 6: Compute Line-of-Sight Integrated kSZ Map (Dimensionless)
# kSZ(z=5) = ∫ n_e0 σ_T (1/a²) (1+δ) x_e v_z/c e^(-τ) ds
# Integration from z=20 to z=5 along line of sight
# =============================================================================

print(f"\n=== LINE-OF-SIGHT kSZ INTEGRATION ===")

# Physical constants in CGS
print(f"\n=== PHYSICAL CONSTANTS (CGS) ===")
c_cm_s = 3.0e10  # cm/s
sigma_T_cm2 = 6.6525e-25  # cm²
n_e0_cm3 = 2.06e-7  # cm⁻³
Mpc_to_cm = 3.0857e24  # cm/Mpc

print(f"c = {c_cm_s:.2e} cm/s")
print(f"σ_T = {sigma_T_cm2:.4e} cm²")
print(f"n_e0 = {n_e0_cm3:.4e} cm⁻³")
print(f"1 Mpc = {Mpc_to_cm:.4e} cm")

# Calculate dimensionless prefactor: n_e0 σ_T c
prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s  # [1/s]
print(f"\nPrefactor n_e0 × σ_T × c = {prefactor_cgs:.4e} s⁻¹")

# Strip units from ds - convert to plain numpy array in Mpc
ds_Mpc = np.asarray(ds.value if hasattr(ds, 'value') else ds, dtype=float)
print(f"\n=== DISTANCE ELEMENTS ===")
print(f"ds (stripped units): {ds_Mpc.min():.6f} - {ds_Mpc.max():.6f} Mpc")

# Convert ds from Mpc to cm
ds_cm = ds_Mpc * Mpc_to_cm  # cm
print(f"ds in cm: {ds_cm.min():.4e} - {ds_cm.max():.4e} cm")

# =============================================================================
# Prepare integrand: (n_e0 σ_T c) × (1/a²) × kSZ_integrand × (ds/c)
# =============================================================================

# Scale factor a = 1/(1+z)
a = 1.0 / (1.0 + red_axis)  # Shape (964,)
a_squared = a**2

# Midpoint kSZ integrand (average between adjacent slices)
kSZ_integrand_mid = 0.5 * (kSZ_integrand_with_vis[:, :, :-1] + 
                            kSZ_integrand_with_vis[:, :, 1:])  # Shape (128, 128, 963)

# Midpoint a²
a_squared_mid = 0.5 * (a_squared[:-1] + a_squared[1:])  # Shape (963,)
a_squared_mid_3D = a_squared_mid[None, None, :]  # Shape (1, 1, 963)

# Full dimensionless integrand for kSZ:
# (n_e0 σ_T c) × (1/a²) × kSZ_integrand × (ds/c)
# [1/s] × [1] × [1] × [s] = [dimensionless]
kSZ_integrand_full = (prefactor_cgs / a_squared_mid_3D) * kSZ_integrand_mid * (ds_cm / c_cm_s)[None, None, :]

print(f"\nIntegrand shape: {kSZ_integrand_full.shape}")
print(f"Integrating over {kSZ_integrand_full.shape[2]} redshift slices")
print(f"From z = {red_axis.max():.2f} to z = {red_axis.min():.2f}")

# Check that integrand is dimensionless
print(f"Integrand type: {type(kSZ_integrand_full)}")
print(f"Sample integrand values: {kSZ_integrand_full[0, 0, :5]}")

# =============================================================================
# Integrate along line of sight (z-axis, axis=2)
# =============================================================================

kSZ_map = np.sum(kSZ_integrand_full, axis=2)  # Shape (128, 128), dimensionless

print(f"\n=== kSZ MAP STATISTICS (DIMENSIONLESS) ===")
print(f"Shape: {kSZ_map.shape}")
print(f"Type: {type(kSZ_map)}")
print(f"Mean: {kSZ_map.mean():.4e}")
print(f"Std:  {kSZ_map.std():.4e}")
print(f"Min:  {kSZ_map.min():.4e}")
print(f"Max:  {kSZ_map.max():.4e}")
print(f"RMS:  {np.sqrt(np.mean(kSZ_map**2)):.4e}")

# Add to lightcone object
lightcone.kSZ_map = kSZ_map

# =============================================================================
# PLOT: kSZ Map at z=5 (final observer slice)
# =============================================================================

def plot_kSZ_map(ax):
    im = ax.imshow(
        kSZ_map.T,
        cmap='seismic_r',
        origin='lower',
        extent=[0, user_params.BOX_LEN, 0, user_params.BOX_LEN],
        aspect='equal'
    )

    # Symmetric colour limits
    vmax_map = np.percentile(np.abs(kSZ_map), 99)
    im.set_clim(-vmax_map, vmax_map)

    # Colorbar
    # Colorbar at the bottom
    fig = ax.figure
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation='horizontal',
        fraction=0.05,
        pad=0.08
    )

    cbar.set_label(r'$\frac{\Delta T}{T}|_{\mathrm{kSZ}}(z=5)$')
    cbar.ax.tick_params(labelsize=25)

    # Labels
    ax.set_xlabel(r'x [Mpc]')
    ax.set_ylabel(r'y [Mpc]')


plot_name = "kSZ_map_z5"

# Preview PDF style
with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
    plot_kSZ_map(ax)
    plt.show()

# Save PDF + PNG
save_pdf_png(
    plot_func=plot_kSZ_map,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"kSZ Map at $z=5$ (LOS integrated)"
)

print(f"Saved: {plot_name}")

# =============================================================================
# PLOT: Histogram of kSZ Map Values
# =============================================================================

def plot_kSZ_hist(ax):
    ax.hist(
        kSZ_map.flatten(),
        bins=100,
        color='darkblue',
        alpha=0.7,
        edgecolor='black'
    )

    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('kSZ Signal (dimensionless)')
    ax.set_ylabel('Number of Pixels')

    stats_text = (
        f"Mean: {kSZ_map.mean():.2e}\n"
        f"Std: {kSZ_map.std():.2e}\n"
        f"RMS: {np.sqrt(np.mean(kSZ_map**2)):.2e}"
    )

    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )


plot_name = "kSZ_map_histogram"

# Preview PDF style
with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_kSZ_hist(ax)
    plt.show()

# Save PDF + PNG
save_pdf_png(
    plot_func=plot_kSZ_hist,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title="kSZ Map Pixel Distribution"
)

print(f"Saved: {plot_name}")


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"LINE-OF-SIGHT INTEGRATION COMPLETE")
print(f"{'='*60}")
print(f"Result is DIMENSIONLESS (proper kSZ signal)")
print(f"Integrated from z = {red_axis.max():.2f} → {red_axis.min():.2f}")
print(f"Map dimensions: {kSZ_map.shape[0]} × {kSZ_map.shape[1]} pixels")
print(f"Physical size: {user_params.BOX_LEN:.1f} × {user_params.BOX_LEN:.1f} Mpc²")
print(f"Pixel size: {user_params.BOX_LEN/kSZ_map.shape[0]:.2f} Mpc")
print(f"RMS kSZ signal: {np.sqrt(np.mean(kSZ_map**2)):.4e}")
print(f"{'='*60}")

# %%
# =============================================================================
# CELL 6.5: LOS-integrated kSZ map — unrotated vs rotated
# Plots at actual (x,y) skewer positions, like Image 2 but sparsely sampled.
# Only integrates up to valid z-depth for rotated skewers.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Physical constants
c_cm_s      = 3.0e10
sigma_T_cm2 = 6.6525e-25
n_e0_cm3    = 2.06e-7
Mpc_to_cm   = 3.0857e24
prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s

ds_Mpc = np.array(ds, dtype=np.float64)
ds_cm  = ds_Mpc * Mpc_to_cm

a             = 1.0 / (1.0 + red_axis_np)
a_squared     = a**2
a_squared_mid = 0.5 * (a_squared[:-1] + a_squared[1:])

# Maximum valid bin depth for rotated skewers
# cos(θ)*i < Nbins  →  i_max = floor(Nbins / cos_a)
# But we also need z_cont < Nbins, so valid output bins: s*cos_a/delta_z < Nbins
# → s < Nbins * delta_z / cos_a, i.e. bin index i < Nbins / cos_a
# For unrotated: all Nbins valid
# For rotated: only bins where z_cont < Nbins-1
Nbins_rot_valid = min(Nbins - 1,
                      int(np.floor((Nbins - 1) * delta_z / (delta_z / cos_a))))
# Simpler: z_cont[i] = s_axis[i]*cos_a/delta_z < Nbins-1
# s_axis[i] = i * delta_z (approx), so i * cos_a < Nbins-1
# → i_max = floor((Nbins-1) / cos_a) ... but we want i where z_cont < Nbins
Nbins_rot_valid = int(np.floor((Nbins - 1) / 1.0))  # all bins, clipping handles it

print(f"Nbins          : {Nbins}")
print(f"cos(θ)         : {cos_a:.4f}")
print(f"Max z reached  : {s_axis[-1] * cos_a / delta_z:.1f} bins  "
      f"(out of {Nbins})")
print(f"Last ~{Nbins - int(s_axis[-1]*cos_a/delta_z)} bins are clipped to box edge")


# =============================================================================
# Integration — with valid bin masking for rotated case
# =============================================================================

def integrate_ksz(lc_dict, vis_1d, n_valid=None):
    """
    LOS-integrated kSZ. n_valid limits integration depth (for rotated skewers).
    Returns (Nlos,).
    """
    n = n_valid if n_valid is not None else Nbins

    d   = np.array(lc_dict['density'],  dtype=np.float64)[:, :n]
    x_e = 1.0 - np.array(lc_dict['xH_box'],  dtype=np.float64)[:, :n]
    v   = np.array(lc_dict['velocity'], dtype=np.float64)[:, :n]
    vis = vis_1d[np.newaxis, :n]

    integrand_full = d * x_e * (v / c_Mpc_s) * vis
    integrand_mid  = 0.5 * (integrand_full[:, :-1] + integrand_full[:, 1:])

    full = (prefactor_cgs / a_squared_mid[np.newaxis, :n-1]) \
           * integrand_mid \
           * (ds_cm[:n-1] / c_cm_s)[np.newaxis, :]

    return np.sum(full, axis=1)   # (Nlos,)


def integrate_ksz_no_vis(lc_dict, n_valid=None):
    n = n_valid if n_valid is not None else Nbins

    d   = np.array(lc_dict['density'],  dtype=np.float64)[:, :n]
    x_e = 1.0 - np.array(lc_dict['xH_box'],  dtype=np.float64)[:, :n]
    v   = np.array(lc_dict['velocity'], dtype=np.float64)[:, :n]

    integrand_full = d * x_e * (v / c_Mpc_s)
    integrand_mid  = 0.5 * (integrand_full[:, :-1] + integrand_full[:, 1:])

    full = (prefactor_cgs / a_squared_mid[np.newaxis, :n-1]) \
           * integrand_mid \
           * (ds_cm[:n-1] / c_cm_s)[np.newaxis, :]

    return np.sum(full, axis=1)


# Valid depth for rotated skewers — last bin where z_cont is still inside box
n_rot_valid = int(np.floor((Nbins - 1) * delta_z * cos_a / delta_z))
# = int((Nbins-1) * cos_a)
n_rot_valid = int((Nbins - 1) * cos_a)
print(f"Rotated valid bins : {n_rot_valid} / {Nbins}")

kSZ_map_unrot_nv  = integrate_ksz_no_vis(lightcone_unrot)
kSZ_map_rot_nv    = integrate_ksz_no_vis(lightcone_rot, n_valid=n_rot_valid)
kSZ_map_unrot_vis = integrate_ksz(lightcone_unrot, vis_unrot_1d)
kSZ_map_rot_vis   = integrate_ksz(lightcone_rot,   vis_rot_1d, n_valid=n_rot_valid)

print(f"\n=== INTEGRATED kSZ MAP STATISTICS ===")
for label, arr in [
    ("Unrotated  no vis ", kSZ_map_unrot_nv),
    ("Rotated    no vis ", kSZ_map_rot_nv),
    ("Unrotated with vis", kSZ_map_unrot_vis),
    ("Rotated   with vis", kSZ_map_rot_vis),
]:
    print(f"  {label} :  RMS={np.sqrt(np.mean(arr**2)):.4e}  "
          f"mean={arr.mean():.4e}")


# =============================================================================
# Build sparse 2D maps at actual (x, y) skewer positions
# Shape: (Ndim, Ndim) with NaN where no skewer exists
# =============================================================================

def build_sparse_map(ksz_1d, los_ind, Ndim, ind_step):
    """
    Place each skewer's integrated value at its (x,y) grid position.
    Returns (Ndim, Ndim) array with NaN at unsampled positions.
    """
    sparse = np.full((Ndim, Ndim), np.nan)
    for k, (x0, y0) in enumerate(los_ind):
        ix = int(x0) % Ndim
        iy = int(y0) % Ndim
        sparse[ix, iy] = ksz_1d[k]
    return sparse


map2d_unrot_nv  = build_sparse_map(kSZ_map_unrot_nv,  LOS_ind, Ndim, ind_step)
map2d_rot_nv    = build_sparse_map(kSZ_map_rot_nv,    LOS_ind, Ndim, ind_step)
map2d_unrot_vis = build_sparse_map(kSZ_map_unrot_vis, LOS_ind, Ndim, ind_step)
map2d_rot_vis   = build_sparse_map(kSZ_map_rot_vis,   LOS_ind, Ndim, ind_step)

print(f"\nSparse map shape : {map2d_unrot_vis.shape}")
print(f"Filled pixels    : {np.sum(~np.isnan(map2d_unrot_vis))} / {Ndim*Ndim}")

_ext_map = [0, Lbox, 0, Lbox]

_clim_nv  = float(np.nanpercentile(
    np.abs(np.concatenate([map2d_unrot_nv[~np.isnan(map2d_unrot_nv)],
                           map2d_rot_nv[~np.isnan(map2d_rot_nv)]])), 99))
_clim_vis = float(np.nanpercentile(
    np.abs(np.concatenate([map2d_unrot_vis[~np.isnan(map2d_unrot_vis)],
                           map2d_rot_vis[~np.isnan(map2d_rot_vis)]])), 99))


# =============================================================================
# Plots
# =============================================================================

def _make_map_plot(data_l, data_r, title_l, title_r, clim):
    def _plot(ax):
        fig = ax.figure
        fig.clf()
        ax_l = fig.add_subplot(1, 2, 1)
        ax_r = fig.add_subplot(1, 2, 2)
        kw = dict(origin='lower', aspect='equal', cmap='seismic_r',
                  vmin=-clim, vmax=clim, extent=_ext_map,
                  interpolation='none')
        im_l = ax_l.imshow(data_l.T, **kw)
        ax_l.set_title(title_l)
        ax_l.set_xlabel('x [Mpc]')
        ax_l.set_ylabel('y [Mpc]')
        fig.colorbar(im_l, ax=ax_l, label=r'$\Delta T/T|_\mathrm{kSZ}$')
        im_r = ax_r.imshow(data_r.T, **kw)
        ax_r.set_title(title_r)
        ax_r.set_xlabel('x [Mpc]')
        ax_r.set_ylabel('y [Mpc]')
        fig.colorbar(im_r, ax=ax_r, label=r'$\Delta T/T|_\mathrm{kSZ}$')
    return _plot


def plot_hist_comparison(ax, data_u, data_r, clim, title_u, title_r):
    bins = np.linspace(-clim, clim, 80)
    ax.hist(data_u[~np.isnan(data_u)], bins=bins,
            color='steelblue', alpha=0.6, label=title_u)
    ax.hist(data_r[~np.isnan(data_r)], bins=bins,
            color='darkorange', alpha=0.6, label=title_r)
    ax.axvline(0, color='black', ls='--', lw=1.5)
    ax.set_xlabel(r'$\Delta T/T|_\mathrm{kSZ}$')
    ax.set_ylabel('Number of skewers')
    ax.legend(loc='best')


_map_plots = [
    ("cell6p5_ksz_map_no_vis",
     _make_map_plot(map2d_unrot_nv, map2d_rot_nv,
                    r'Unrotated — no vis',
                    rf'Rotated ($\theta={angle_deg}°$) — no vis',
                    _clim_nv),
     r"kSZ map (no visibility): unrotated vs rotated"),

    ("cell6p5_ksz_map_with_vis",
     _make_map_plot(map2d_unrot_vis, map2d_rot_vis,
                    r'Unrotated — with vis',
                    rf'Rotated ($\theta={angle_deg}°$) — with vis',
                    _clim_vis),
     r"kSZ map (with visibility): unrotated vs rotated"),
]

for plot_name, plot_func, title in _map_plots:
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)
        plot_func(ax)
        plt.show()
    save_pdf_png(plot_func=plot_func, plot_dir=plot_dir,
                 plot_name=plot_name, title=title)
    print(f"Saved: {plot_name}")

# Histograms
def plot_hist_nv(ax):
    plot_hist_comparison(ax,
        map2d_unrot_nv.ravel(), map2d_rot_nv.ravel(), _clim_nv,
        r'Unrotated ($\theta=0°$)', rf'Rotated ($\theta={angle_deg}°$)')

def plot_hist_vis(ax):
    plot_hist_comparison(ax,
        map2d_unrot_vis.ravel(), map2d_rot_vis.ravel(), _clim_vis,
        r'Unrotated ($\theta=0°$)', rf'Rotated ($\theta={angle_deg}°$)')

for plot_name, plot_func, title in [
    ("cell6p5_ksz_hist_no_vis",  plot_hist_nv,
     r"kSZ distribution (no visibility)"),
    ("cell6p5_ksz_hist_with_vis", plot_hist_vis,
     r"kSZ distribution (with visibility)"),
]:
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        plt.show()
    save_pdf_png(plot_func=plot_func, plot_dir=plot_dir,
                 plot_name=plot_name, title=title)
    print(f"Saved: {plot_name}")

# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"CELL 6.5 SUMMARY")
print(f"{'='*60}")
print(f"  Nlos skewers     : {Nlos}")
print(f"  Rotated valid bins: {n_rot_valid} / {Nbins}  "
      f"({n_rot_valid/Nbins*100:.1f}%)")
for label, arr in [
    ("Unrotated  no vis ", kSZ_map_unrot_nv),
    ("Rotated    no vis ", kSZ_map_rot_nv),
    ("Unrotated with vis", kSZ_map_unrot_vis),
    ("Rotated   with vis", kSZ_map_rot_vis),
]:
    print(f"  {label} :  RMS={np.sqrt(np.mean(arr**2)):.4e}")
print(f"{'='*60}")
print(f"\nReady for power spectrum comparison when you are.")

# %%
# =============================================================================
# CELL 6.6: Difference maps — skewer kSZ vs full original lightcone kSZ
# Unrotated skewers should show ~zero difference from original.
# Rotated skewers should show measurable difference.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# Align original kSZ_map with skewer maps
# kSZ_map from Cell 6: (128, 128) — full spatial map
# map2d_unrot_vis / map2d_rot_vis: (128, 128) sparse — NaN at unsampled positions
# Difference is only meaningful at sampled (x,y) positions
# =============================================================================

# kSZ_map may have shape issues if Cell 6 ran on patched lightcone
# Always use _lightcone_original.kSZ_map if available
_kSZ_map_orig = np.array(kSZ_map, dtype=np.float64)   # (128, 128)

print(f"Original kSZ map shape  : {_kSZ_map_orig.shape}")
print(f"Skewer map shape        : {map2d_unrot_vis.shape}")
print(f"Sampled pixels          : {np.sum(~np.isnan(map2d_unrot_vis))}")

# =============================================================================
# Compute differences — NaN at unsampled positions
# =============================================================================

diff_unrot = map2d_unrot_vis - _kSZ_map_orig   # (128, 128), NaN where unsampled
diff_rot   = map2d_rot_vis   - _kSZ_map_orig   # (128, 128), NaN where unsampled

# Extract only sampled pixels for statistics
diff_unrot_vals = diff_unrot[~np.isnan(diff_unrot)]
diff_rot_vals   = diff_rot[~np.isnan(diff_rot)]

print(f"\n{'='*60}")
print(f"DIFFERENCE STATISTICS (at sampled positions only)")
print(f"{'='*60}")
print(f"\n  Unrotated - Original:")
print(f"    mean   : {diff_unrot_vals.mean():.4e}")
print(f"    std    : {diff_unrot_vals.std():.4e}")
print(f"    RMS    : {np.sqrt(np.mean(diff_unrot_vals**2)):.4e}")
print(f"    max|Δ| : {np.abs(diff_unrot_vals).max():.4e}")

print(f"\n  Rotated - Original:")
print(f"    mean   : {diff_rot_vals.mean():.4e}")
print(f"    std    : {diff_rot_vals.std():.4e}")
print(f"    RMS    : {np.sqrt(np.mean(diff_rot_vals**2)):.4e}")
print(f"    max|Δ| : {np.abs(diff_rot_vals).max():.4e}")

# Relative difference at sampled positions
orig_vals = _kSZ_map_orig[~np.isnan(map2d_unrot_vis)]
print(f"\n  Original RMS at sampled positions : {np.sqrt(np.mean(orig_vals**2)):.4e}")
print(f"  Rel. diff unrotated : "
      f"{np.sqrt(np.mean(diff_unrot_vals**2))/np.sqrt(np.mean(orig_vals**2))*100:.2f}%")
print(f"  Rel. diff rotated   : "
      f"{np.sqrt(np.mean(diff_rot_vals**2))/np.sqrt(np.mean(orig_vals**2))*100:.2f}%")
print(f"{'='*60}")


# =============================================================================
# Shared colour limits
# =============================================================================

_clim_diff = float(np.nanpercentile(
    np.abs(np.concatenate([diff_unrot_vals, diff_rot_vals])), 99))
_clim_orig = float(np.nanpercentile(np.abs(_kSZ_map_orig), 99))

_ext_map = [0, Lbox, 0, Lbox]
_kw_base = dict(origin='lower', aspect='equal',
                interpolation='none', extent=_ext_map)


# =============================================================================
# Plot 1: Side-by-side difference maps
# =============================================================================

def plot_diff_maps(ax):
    fig = ax.figure
    fig.clf()
    ax_l = fig.add_subplot(1, 2, 1)
    ax_r = fig.add_subplot(1, 2, 2)

    im_l = ax_l.imshow(diff_unrot.T, cmap='coolwarm',
                        vmin=-_clim_diff, vmax=_clim_diff, **_kw_base)
    ax_l.set_title(r'Unrotated $-$ Original  (expect ~zero)')
    ax_l.set_xlabel('x [Mpc]')
    ax_l.set_ylabel('y [Mpc]')
    fig.colorbar(im_l, ax=ax_l, label=r'$\Delta(\Delta T/T)$')

    im_r = ax_r.imshow(diff_rot.T, cmap='coolwarm',
                        vmin=-_clim_diff, vmax=_clim_diff, **_kw_base)
    ax_r.set_title(rf'Rotated ($\theta={angle_deg}°$) $-$ Original  (expect structure)')
    ax_r.set_xlabel('x [Mpc]')
    ax_r.set_ylabel('y [Mpc]')
    fig.colorbar(im_r, ax=ax_r, label=r'$\Delta(\Delta T/T)$')


# =============================================================================
# Plot 2: 3-panel — Original | Unrotated diff | Rotated diff
# =============================================================================

def plot_three_panel(ax):
    fig = ax.figure
    fig.clf()
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    im0 = ax0.imshow(_kSZ_map_orig.T, cmap='seismic_r',
                      vmin=-_clim_orig, vmax=_clim_orig, **_kw_base)
    ax0.set_title('Original (full box)')
    ax0.set_xlabel('x [Mpc]')
    ax0.set_ylabel('y [Mpc]')
    fig.colorbar(im0, ax=ax0, label=r'$\Delta T/T$')

    im1 = ax1.imshow(diff_unrot.T, cmap='coolwarm',
                      vmin=-_clim_diff, vmax=_clim_diff, **_kw_base)
    ax1.set_title(r'Unrotated $-$ Original')
    ax1.set_xlabel('x [Mpc]')
    ax1.set_ylabel('y [Mpc]')
    fig.colorbar(im1, ax=ax1, label=r'$\Delta(\Delta T/T)$')

    im2 = ax2.imshow(diff_rot.T, cmap='coolwarm',
                      vmin=-_clim_diff, vmax=_clim_diff, **_kw_base)
    ax2.set_title(rf'Rotated ($\theta={angle_deg}°$) $-$ Original')
    ax2.set_xlabel('x [Mpc]')
    ax2.set_ylabel('y [Mpc]')
    fig.colorbar(im2, ax=ax2, label=r'$\Delta(\Delta T/T)$')


# =============================================================================
# Plot 3: Histogram of differences
# =============================================================================

def plot_diff_hist(ax):
    bins = np.linspace(-_clim_diff, _clim_diff, 80)
    ax.hist(diff_unrot_vals, bins=bins, color='steelblue',
            alpha=0.6, label=r'Unrotated $-$ Original')
    ax.hist(diff_rot_vals,   bins=bins, color='darkorange',
            alpha=0.6, label=rf'Rotated ($\theta={angle_deg}°$) $-$ Original')
    ax.axvline(0, color='black', ls='--', lw=1.5)
    ax.set_xlabel(r'$\Delta(\Delta T/T)|_\mathrm{kSZ}$')
    ax.set_ylabel('Number of skewers')
    ax.legend(loc='best')


# =============================================================================
# Save all plots
# =============================================================================

_plots = [
    ("cell6p6_diff_maps",
     plot_diff_maps,
     r"kSZ difference: skewer vs original lightcone"),
    ("cell6p6_three_panel",
     plot_three_panel,
     r"kSZ: original, unrotated diff, rotated diff"),
    ("cell6p6_diff_hist",
     plot_diff_hist,
     r"kSZ difference distribution: unrotated vs rotated"),
]

_sizes = [(16, 7), (20, 6), (10, 7)]

for (plot_name, plot_func, title), figsize in zip(_plots, _sizes):
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plot_func(ax)
        plt.show()
    save_pdf_png(plot_func=plot_func, plot_dir=plot_dir,
                 plot_name=plot_name, title=title)
    print(f"Saved: {plot_name}")

print(f"\nWhat to look for:")
print(f"  Unrotated diff : should be ~zero — confirms skewer extraction is correct")
print(f"  Rotated diff   : should show structure — confirms rotation samples")
print(f"                   different physical locations from the original LOS")

# %%
# =============================================================================
# CELL 8: Compute kSZ Power Spectrum - P(k), C_ℓ, and D_ℓ
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

print(f"\n{'='*60}")
print(f"COMPUTING kSZ POWER SPECTRUM")
print(f"{'='*60}")

# =============================================================================
# 1. Settings and Map Info
# =============================================================================

# Map info
npix_side = kSZ_map.shape[0]  # Should be 128
box_size_Mpc = user_params.BOX_LEN  # Physical size in Mpc
pix_size_Mpc = box_size_Mpc / npix_side  # Mpc per pixel

print(f"\n=== MAP PROPERTIES ===")
print(f"Map size: {npix_side} × {npix_side} pixels")
print(f"Physical size: {box_size_Mpc:.1f} × {box_size_Mpc:.1f} Mpc²")
print(f"Pixel size: {pix_size_Mpc:.3f} Mpc/pixel")
print(f"Map RMS: {np.std(kSZ_map):.4e}")

# Remove mean (important for power spectrum!)
kSZ_map_centered = kSZ_map - np.mean(kSZ_map)
print(f"Mean subtracted: {np.mean(kSZ_map_centered):.4e}")

# =============================================================================
# 2. Compute 2D Power Spectrum P(k) in [Mpc²]
# =============================================================================

# FFT and shift to center
fft_map = np.fft.fft2(kSZ_map_centered)
fft_map_shifted = np.fft.fftshift(fft_map)

# Pixel area in physical units
pix_area = pix_size_Mpc**2  # Mpc²

# 2D power spectrum: P(k) in [Mpc²]
#ps2d = np.abs(fft_map_shifted)**2 * pix_area / (npix_side**4)
# 2D power spectrum: P(k) in [Mpc²]
ps2d = (pix_size_Mpc / npix_side)**2 * np.abs(fft_map_shifted)**2

print(f"\n=== 2D POWER SPECTRUM ===")
print(f"P(k) range: [{ps2d.min():.4e}, {ps2d.max():.4e}] Mpc²")

# =============================================================================
# 3. k-space grid [Mpc⁻¹]
# =============================================================================

# Fundamental frequency
dk = 2 * np.pi / (npix_side * pix_size_Mpc)  # Mpc⁻¹

# k-space coordinates
kx = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
ky = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

print(f"\n=== k-SPACE GRID ===")
print(f"dk (fundamental): {dk:.6f} Mpc⁻¹")
print(f"k range: [{kgrid.min():.6f}, {kgrid.max():.6f}] Mpc⁻¹")

# =============================================================================
# 4. Azimuthally Averaged P(k) with Cosmic Variance
# =============================================================================

# k_volume for a 2D plane (Area of k-space)
# Assuming a square box of side L, the fundamental mode is dk = 2*pi / L
k_area_fundamental = dk**2 

# Define k bins
k_bins = np.logspace(np.log10(dk), np.log10(kgrid.max()*0.9), 35)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

P1d = np.zeros(len(k_centers))
P1d_err_sample = np.zeros(len(k_centers))
P1d_err_cosmic = np.zeros(len(k_centers))
P1d_err  = np.zeros(len(k_centers))

for i in range(len(k_centers)):
    mask = (kgrid >= k_bins[i]) & (kgrid < k_bins[i+1])
    n_modes_measured = np.sum(mask)
    
    if n_modes_measured > 0:
        values = ps2d[mask]
        P1d[i] = np.mean(values)
        
        # 1. Sample Variance (Standard Error of the Mean from the data)
        P1d_err_sample[i] = np.std(values) / np.sqrt(n_modes_measured)
        
        # 2. Cosmic Variance Theory: Error = P(k) / sqrt(N_modes)
        # N_modes = Area of ring / Area of fundamental cell
        ring_area = np.pi * (k_bins[i+1]**2 - k_bins[i]**2)
        n_modes_theoretical = ring_area / k_area_fundamental
        
        # Use the theoretical n_modes for cosmic variance
        P1d_err_cosmic[i] = P1d[i] / np.sqrt(n_modes_theoretical)
        
        # 3. Total Error (Quadrature Sum)
        P1d_err[i] = np.sqrt(P1d_err_sample[i]**2 + P1d_err_cosmic[i]**2)
    else:
        P1d[i] = np.nan
        P1d_err[i] = np.nan

print(f"\n=== AZIMUTHALLY AVERAGED P(k) ===")
print(f"Valid k bins: {np.sum(~np.isnan(P1d))} out of {len(k_centers)}")

# =============================================================================
# 5. Convert to ℓ space (flat-sky approximation for kSZ)
# =============================================================================

# For kSZ, we need the COMOVING distance to the reionization era
# At z=5, comoving distance (not angular diameter distance!)
z_reion = 5.0

# Comoving distance to z=5 in Planck cosmology: χ ≈ 7800 Mpc
chi_comoving_Mpc = 7800  # This is the light travel distance (comoving)

# Alternative: If you have angular diameter distance D_A
D_A_physical = 1300  # Physical angular diameter distance at z=5
chi_comoving_Mpc = D_A_physical * (1 + z_reion)  # = 1300 × 6 = 7800 Mpc

print(f"\n=== CONVERSION TO ℓ SPACE ===")
print(f"Reionization redshift: z = {z_reion}")
print(f"Comoving distance: χ = {chi_comoving_Mpc:.1f} Mpc")

# Flat-sky approximation: ℓ = k × χ
ell_from_k = k_centers * chi_comoving_Mpc/0.67

# Power spectrum conversion: C_ℓ = P(k) / χ²
Cl_kSZ_DA = P1d*0.67**2*36 / chi_comoving_Mpc**2 
#Cl_kSZ_comov = P1d / chi_comoving_Mpc**2
# Convert to D_ℓ = ℓ(ℓ+1) C_ℓ / (2π) in μK²
Dl_kSZ_muK2_DA = ell_from_k * (ell_from_k + 1) * Cl_kSZ_DA / (2 * np.pi)
#Dl_kSZ_muK2_comov = ell_from_k * (ell_from_k + 1) * Cl_kSZ_comov / (2 * np.pi)

print(f"ℓ range: {ell_from_k.min():.0f} to {ell_from_k.max():.0f}")
#print(f"Peak D_ℓ: {Dl_kSZ_muK2.max():.2e} μK²")


# =============================================================================
# PLOT 1: P(k) - 3D Matter Power Spectrum
# =============================================================================

def plot_Pk(ax):
    valid = ~np.isnan(P1d) & (P1d > 0)

    ax.loglog(
        k_centers[valid], P1d[valid],
        'o-', color='orange',
        linewidth=2.5, markersize=6,
        label='P(k) from kSZ map'
    )

    ax.fill_between(
        k_centers[valid],
        (P1d - P1d_err)[valid],
        (P1d + P1d_err)[valid],
        alpha=0.3, color='orange'
    )

    ax.set_xlabel(r'$k$ [h $\cdot$ cMpc$^{-1}$]')
    ax.set_ylabel(r'$P(k)$ [Mpc$^{2}$]')
    ax.legend(loc='best')


plot_name = "kSZ_Pk"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_Pk(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_Pk,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title="kSZ Power Spectrum $P(k)$"
)

print(f"Saved: {plot_name}")

# =============================================================================
# PLOT 2: C_ℓ - Angular Power Spectrum
# =============================================================================

def plot_Cl(ax):
    valid = ~np.isnan(Cl_kSZ_DA) & (Cl_kSZ_DA > 0) & (ell_from_k > 10)

    ax.loglog(
        ell_from_k[valid], Cl_kSZ_DA[valid],
        'o--', color='blue',
        alpha=0.5, linewidth=1.5, markersize=4,
        label=r'$C_\ell$'
    )

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$ [$\mu$K$^2$]')
    ax.legend(loc='best')


plot_name = "kSZ_Cl_corrected"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_Cl(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_Cl,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"kSZ Angular Power Spectrum $C_\ell$"
)

print(f"Saved: {plot_name}")


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"POWER SPECTRUM SUMMARY")
print(f"{'='*60}")
#print(f"P(k) computed in range: k = {k_centers[valid].min():.4f} - {k_centers[valid].max():.4f} Mpc⁻¹")
#print(f"C_ℓ computed in range: ℓ = {ell_from_k[valid].min():.1f} - {ell_from_k[valid].max():.1f}")
print(f"\nNOTE: This uses flat-sky approximation")
#print(f"Conversion: ℓ ≈ k × D_A where D_A = {D_A_Mpc:.1f} Mpc at z={z_obs}")
print(f"{'='*60}")

# %%
# =============================================================================
# CELL 8.5: kSZ Power Spectrum — unrotated vs rotated comparison
# Mirrors Cell 8 exactly for both ensembles, with visibility only.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# Build 2D kSZ maps from skewer arrays
# Nlos=1000 is not a perfect square — find the largest square that fits
# =============================================================================

npix_map  = int(np.floor(np.sqrt(Nlos)))   # 31 for Nlos=1000
n_use     = npix_map * npix_map            # 961 skewers used

print(f"Nlos         : {Nlos}")
print(f"npix_map     : {npix_map}  ({npix_map}×{npix_map} = {n_use} skewers used)")
print(f"Skewers dropped : {Nlos - n_use}  (edge skewers, negligible)")

map_unrot = kSZ_map_unrot_vis[:n_use].reshape(npix_map, npix_map)
map_rot   = kSZ_map_rot_vis[:n_use].reshape(npix_map,   npix_map)

print(f"2D map shape : {map_unrot.shape}")

# Pixel size — skewers span the full box face
pix_size_Mpc = float(user_params.BOX_LEN) / npix_map

print(f"Pixel size   : {pix_size_Mpc:.3f} Mpc/pixel")
print(f"Physical size: {float(user_params.BOX_LEN):.1f} × "
      f"{float(user_params.BOX_LEN):.1f} Mpc²")


# =============================================================================
# Power spectrum — identical to Cell 8
# =============================================================================

def compute_Pk_2d(ksz_map_2d, pix_size_Mpc, n_kbins=35):
    npix  = ksz_map_2d.shape[0]
    m     = ksz_map_2d - ksz_map_2d.mean()

    fft_m = np.fft.fftshift(np.fft.fft2(m))
    ps2d  = (pix_size_Mpc / npix)**2 * np.abs(fft_m)**2

    dk    = 2 * np.pi / (npix * pix_size_Mpc)
    kx    = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    ky    = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    k_bins    = np.logspace(np.log10(dk),
                            np.log10(kgrid.max() * 0.9), n_kbins)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    P1d       = np.zeros(len(k_centers))
    P1d_err   = np.zeros(len(k_centers))

    for i in range(len(k_centers)):
        mask = (kgrid >= k_bins[i]) & (kgrid < k_bins[i+1])
        if mask.sum() > 0:
            vals       = ps2d[mask]
            P1d[i]     = vals.mean()
            P1d_err[i] = vals.std() / np.sqrt(mask.sum())
        else:
            P1d[i] = np.nan

    return k_centers, P1d, P1d_err, dk


k_unrot, P_unrot, Pe_unrot, dk_u = compute_Pk_2d(map_unrot, pix_size_Mpc)
k_rot,   P_rot,   Pe_rot,   dk_r = compute_Pk_2d(map_rot,   pix_size_Mpc)

print(f"\n=== P(k) RANGES ===")
print(f"  Unrotated : {np.nanmin(P_unrot[P_unrot>0]):.4e} – "
      f"{np.nanmax(P_unrot):.4e} Mpc²")
print(f"  Rotated   : {np.nanmin(P_rot[P_rot>0]):.4e} – "
      f"{np.nanmax(P_rot):.4e} Mpc²")


# =============================================================================
# ℓ-space conversion — same as Cell 8
# =============================================================================

chi_comoving_Mpc = 7800.0

ell_unrot = k_unrot * chi_comoving_Mpc / 0.67
ell_rot   = k_rot   * chi_comoving_Mpc / 0.67

Cl_unrot  = P_unrot  / chi_comoving_Mpc**2 #0.67**2
Cl_rot    = P_rot    / chi_comoving_Mpc**2 #0.67**2 

Dl_unrot  = ell_unrot * (ell_unrot + 1) * Cl_unrot / (2 * np.pi)
Dl_rot    = ell_rot   * (ell_rot   + 1) * Cl_rot   / (2 * np.pi)

# μK² conversion — same T_CMB as Cell 8
T_CMB_uK  = 2.725e6
Dl_unrot_uK2 = Dl_unrot * T_CMB_uK**2
Dl_rot_uK2   = Dl_rot   * T_CMB_uK**2

# Box-scale artefact wavenumbers
k_box     = 2 * np.pi / float(user_params.BOX_LEN)
k_box_rot = 2 * np.pi / (float(user_params.BOX_LEN) / sin_a)
ell_box     = k_box     * chi_comoving_Mpc / 0.67
ell_box_rot = k_box_rot * chi_comoving_Mpc / 0.67

print(f"\n  Box artefact  k (unrot) : {k_box:.5f} Mpc⁻¹  →  ℓ ~ {ell_box:.0f}")
print(f"  Box artefact  k (rot)   : {k_box_rot:.5f} Mpc⁻¹  →  ℓ ~ {ell_box_rot:.0f}")


# =============================================================================
# Plots
# =============================================================================

def plot_Pk_comparison(ax):
    vu = ~np.isnan(P_unrot) & (P_unrot > 0)
    vr = ~np.isnan(P_rot)   & (P_rot   > 0)
    ax.loglog(k_unrot[vu], P_unrot[vu],
              'o-', color='steelblue', lw=2, ms=5,
              label=r'Unrotated ($\theta=0°$)')
    ax.fill_between(k_unrot[vu],
                    (P_unrot - Pe_unrot)[vu],
                    (P_unrot + Pe_unrot)[vu],
                    alpha=0.25, color='steelblue')
    ax.loglog(k_rot[vr], P_rot[vr],
              's--', color='darkorange', lw=2, ms=5,
              label=rf'Rotated ($\theta={angle_deg}°$)')
    ax.fill_between(k_rot[vr],
                    (P_rot - Pe_rot)[vr],
                    (P_rot + Pe_rot)[vr],
                    alpha=0.25, color='darkorange')
    ax.axvline(k_box,     color='steelblue',  lw=1.2, ls=':', alpha=0.8,
               label=f'Artefact k (unrot) = {k_box:.4f} Mpc⁻¹')
    ax.axvline(k_box_rot, color='darkorange', lw=1.2, ls=':', alpha=0.8,
               label=f'Artefact k (rot) = {k_box_rot:.5f} Mpc⁻¹')
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax.set_ylabel(r'$P(k)$ [Mpc$^{2}$]')
    ax.legend(loc='best')

def plot_Cl_comparison(ax):
    vu = ~np.isnan(Cl_unrot) & (Cl_unrot > 0) & (ell_unrot > 10)
    vr = ~np.isnan(Cl_rot)   & (Cl_rot   > 0) & (ell_rot   > 10)
    ax.loglog(ell_unrot[vu], Cl_unrot[vu],
              'o-', color='steelblue', lw=2, ms=5,
              label=r'Unrotated ($\theta=0^{\circ}$)')
    ax.loglog(ell_rot[vr], Cl_rot[vr],
              's--', color='darkorange', lw=2, ms=5,
              label=rf'Rotated ($\theta={angle_deg}^{{\circ}}$)')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$')
    ax.legend(loc='best')

def plot_Dl_comparison(ax):
    vu = ~np.isnan(Dl_unrot_uK2) & (Dl_unrot_uK2 > 0) & (ell_unrot > 10)
    vr = ~np.isnan(Dl_rot_uK2)   & (Dl_rot_uK2   > 0) & (ell_rot   > 10)
    vo = ~np.isnan(Dl_kSZ_muK2_DA) & (Dl_kSZ_muK2_DA > 0) & (ell_from_k > 10)

    # Original full lightcone (Cell 8) — needs T_CMB² scaling same as Cell 8
    Dl_orig_uK2 = Dl_kSZ_muK2_DA * T_CMB_uK**2

    ax.loglog(ell_from_k[vo], Dl_orig_uK2[vo],
              '^-', color='black', lw=2, ms=5,
              label=r'Original (full box, $128\times128$)')
    ax.loglog(ell_unrot[vu], Dl_unrot_uK2[vu],
              'o-', color='steelblue', lw=2, ms=5,
              label=r'Unrotated skewers ($\theta=0^{\circ}$)')
    ax.loglog(ell_rot[vr], Dl_rot_uK2[vr],
              's--', color='darkorange', lw=2, ms=5,
              label=rf'Rotated skewers ($\theta={angle_deg}^{{\circ}}$)')
   
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell\ [\mu\mathrm{K}^2]$')
    ax.legend(loc='best')
    
_plots = [
    ("cell8p5_Pk_comparison",
     plot_Pk_comparison,
     r"kSZ $P(k)$: unrotated vs rotated (with visibility)"),
    ("cell8p5_Cl_comparison",
     plot_Cl_comparison,
     r"kSZ $C_\ell$: unrotated vs rotated (with visibility)"),
    ("cell8p5_Dl_comparison",
     plot_Dl_comparison,
     r"kSZ $D_\ell\ [\mu\mathrm{K}^2]$: unrotated vs rotated (with visibility)"),
]

for plot_name, plot_func, title in _plots:
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        plt.show()
    save_pdf_png(plot_func=plot_func, plot_dir=plot_dir,
                 plot_name=plot_name, title=title)
    print(f"Saved: {plot_name}")

# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"CELL 8.5 SUMMARY")
print(f"{'='*60}")
print(f"  Map shape      : {map_unrot.shape}  ({npix_map}×{npix_map})")
print(f"  Pixel size     : {pix_size_Mpc:.3f} Mpc")
print(f"  Skewers used   : {n_use} / {Nlos}")
print(f"  k_box unrot    : {k_box:.5f} Mpc⁻¹  (ℓ ~ {ell_box:.0f})")
print(f"  k_box rot      : {k_box_rot:.5f} Mpc⁻¹  (ℓ ~ {ell_box_rot:.0f})")
print(f"  Suppression    : {k_box_rot/k_box:.1f}× in k,  {ell_box_rot/ell_box:.1f}× in ℓ")
print(f"{'='*60}")

# %%
# =============================================================================
# CELL 8b: Normalization Validation via Coarse-Graining
# =============================================================================
# Paste this after Cell 6. Requires: kSZ_map, user_params, plot_dir,
#   PDF_STYLE, PNG_STYLE, save_pdf_png (already defined in your notebook).
# =============================================================================

import numpy as np

# ── core functions ────────────────────────────────────────────────────────────

def compute_ps2d(mp, pix_size_Mpc):
    """
    2D power spectrum. Correct normalization:
        P(k) [Mpc²] = (pix_size / N)² * |FFT(map - mean)|²
    This is invariant under coarse-graining (box_size fixed, N and pix_size
    change together), so CG and original curves should overlap.
    """
    N = mp.shape[0]
    mp_centered = mp - np.mean(mp)
    fft_shifted = np.fft.fftshift(np.fft.fft2(mp_centered))
    ps2d = (pix_size_Mpc / N)**2 * np.abs(fft_shifted)**2
    return ps2d


def compute_kgrid(N, pix_size_Mpc):
    dk = 2 * np.pi / (N * pix_size_Mpc)
    kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    ky = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    return kgrid, dk


def azimuthal_average(ps2d, kgrid, dk_fund, n_bins=35):
    k_bins = np.logspace(np.log10(dk_fund), np.log10(kgrid.max() * 0.9), n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    P1d     = np.full(n_bins, np.nan)
    P1d_err = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (kgrid >= k_bins[i]) & (kgrid < k_bins[i + 1])
        if mask.sum() > 0:
            vals = ps2d[mask]
            P1d[i]     = vals.mean()
            P1d_err[i] = vals.std() / np.sqrt(mask.sum())
    return k_centers, P1d, P1d_err


def coarse_grain(mp, factor):
    N = mp.shape[0]
    assert N % factor == 0, f"Map size {N} not divisible by factor {factor}"
    Nn = N // factor
    return mp.reshape(Nn, factor, Nn, factor).mean(axis=(1, 3))


def compute_Cl_Dl(k_centers, P1d, D_A_physical=1300, h=0.67, z_reion=5.0):
    chi = D_A_physical * (1 + z_reion)
    ell = k_centers * chi / h
    Cl  = P1d * h**2 / D_A_physical**2
    Dl  = ell * (ell + 1) * Cl / (2 * np.pi)
    return ell, Cl, Dl


def build_results(mp, pix_size_Mpc, label, n_bins=40,
                  D_A_physical=1300, h=0.67):
    N = mp.shape[0]
    ps2d          = compute_ps2d(mp, pix_size_Mpc)
    kgrid, dk     = compute_kgrid(N, pix_size_Mpc)
    k, P1d, P1d_err = azimuthal_average(ps2d, kgrid, dk, n_bins=n_bins)
    ell, Cl, Dl   = compute_Cl_Dl(k, P1d, D_A_physical, h)
    return dict(k=k, P1d=P1d, P1d_err=P1d_err,
                ell=ell, Cl=Cl, Dl=Dl,
                k_nyq=np.pi / pix_size_Mpc,
                pix_size=pix_size_Mpc, N=N, label=label)


# ── build all results ─────────────────────────────────────────────────────────

N            = kSZ_map.shape[0]
pix_size_Mpc = user_params.BOX_LEN / N
D_A_physical = 1300   # Mpc at z=5
h            = 0.67
coarse_factors = (2, 4, 8)
n_bins       = 40

results = {}
results['original'] = build_results(kSZ_map, pix_size_Mpc,
                                     f'Original ({N}×{N})',
                                     n_bins, D_A_physical, h)

for factor in coarse_factors:
    mp_cg  = coarse_grain(kSZ_map, factor)
    pix_cg = pix_size_Mpc * factor
    Nn     = N // factor
    results[f'cg_{factor}'] = build_results(mp_cg, pix_cg,
                                             f'CG×{factor} ({Nn}×{Nn})',
                                             n_bins, D_A_physical, h)

# Parseval check
fft2 = np.fft.fft2(kSZ_map - np.mean(kSZ_map))
parseval_ps  = np.mean(np.abs(fft2)**2) / N**2
map_var      = np.var(kSZ_map)
print(f"Parseval check — map variance: {map_var:.4e},  FFT: {parseval_ps:.4e},  ratio: {parseval_ps/map_var:.4f}")

# ── plot helpers ──────────────────────────────────────────────────────────────

COLORS = ['black', 'royalblue', 'tomato', 'forestgreen', 'purple']

def _add_nyquist(ax, res, col, x_is_ell=False, D_A_physical=1300, h=0.67, z_reion=5.0):
    if x_is_ell:
        chi     = D_A_physical * (1 + z_reion)
        x_nyq   = res['k_nyq'] * chi / h
    else:
        x_nyq   = res['k_nyq']
    ax.axvline(x_nyq, color=col, linestyle=':', alpha=0.6, linewidth=1.5)


# ── PLOT 1: P(k) ─────────────────────────────────────────────────────────────

def plot_Pk_validation(ax):
    for i, (key, res) in enumerate(results.items()):
        valid = ~np.isnan(res['P1d']) & (res['P1d'] > 0)
        col   = COLORS[i]
        ls    = '-' if i == 0 else '--'
        ax.loglog(res['k'][valid], res['P1d'][valid],
                  linestyle=ls, marker='o', color=col,
                  label=res['label'])
        ax.fill_between(res['k'][valid],
                        (res['P1d'] - res['P1d_err'])[valid],
                        (res['P1d'] + res['P1d_err'])[valid],
                        alpha=0.15, color=col)
        if key != 'original':
            _add_nyquist(ax, res, col, x_is_ell=False)
    ax.set_xlabel(r'$k\;[\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k)\;[\mathrm{Mpc}^{2}]$')
    ax.legend()

save_pdf_png(plot_Pk_validation, plot_dir,
             'kSZ_coarsegrain_Pk',
             title=r'Normalization check: $P(k)$ — curves must overlap')

with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    plot_Pk_validation(ax)
    plt.show()


# ── PLOT 2: D_ℓ ──────────────────────────────────────────────────────────────

def plot_Dl_validation(ax):
    for i, (key, res) in enumerate(results.items()):
        valid = (~np.isnan(res['Dl']) & (res['Dl'] > 0) & (res['ell'] > 10))
        col   = COLORS[i]
        ls    = '-' if i == 0 else '--'
        ax.loglog(res['ell'][valid], res['Dl'][valid],
                  linestyle=ls, marker='o', color=col,
                  label=res['label'])
        if key != 'original':
            _add_nyquist(ax, res, col, x_is_ell=True,
                         D_A_physical=D_A_physical, h=h)
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell\;[\mu\mathrm{K}^2]$')
    ax.legend()

save_pdf_png(plot_Dl_validation, plot_dir,
             'kSZ_coarsegrain_Dl',
             title=r'Normalization check: $D_\ell$ — curves must overlap')

with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    plot_Dl_validation(ax)
    plt.show()


# ── PLOT 3: ratio P_CG / P_orig ──────────────────────────────────────────────

orig = results['original']
valid_or = ~np.isnan(orig['P1d']) & (orig['P1d'] > 0)

def plot_ratio(ax):
    for i, (key, res) in enumerate(results.items()):
        if key == 'original':
            continue
        col      = COLORS[i]
        valid_cg = ~np.isnan(res['P1d']) & (res['P1d'] > 0)
        P_interp = np.interp(res['k'][valid_cg],
                             orig['k'][valid_or], orig['P1d'][valid_or],
                             left=np.nan, right=np.nan)
        ratio = res['P1d'][valid_cg] / P_interp
        ax.semilogx(res['k'][valid_cg], ratio,
                    'o-', color=col, label=res['label'])
        _add_nyquist(ax, res, col, x_is_ell=False)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
               label='Perfect agreement')
    ax.axhspan(0.95, 1.05, alpha=0.1, color='gray', label=r'$\pm5\%$')
    ax.set_xlabel(r'$k\;[\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P_{\mathrm{CG}}(k)\;/\;P_{\mathrm{orig}}(k)$')
    ax.set_ylim(0, 2)
    ax.legend()

save_pdf_png(plot_ratio, plot_dir,
             'kSZ_coarsegrain_ratio',
             title=r'Ratio $P_{\rm CG}/P_{\rm orig}$ — should be $\approx 1$ below $k_{\rm Nyq}$')

with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    plot_ratio(ax)
    plt.show()

# %%
#== ===========================================================================#
# NOT FOR REPORTS
# EXTRA PLOT: 2D Power Spectrum P(kx, ky)
#== ===========================================================================#

def plot_kSZ_2D_power(ax):
    # Log-scale power for visualisation
    ps2d_log = np.log10(ps2d + 1e-30)

    # k-space extent
    k_max = kgrid.max()

    im = ax.imshow(
        ps2d_log.T,
        origin='lower',
        cmap='viridis',
        extent=[-k_max, k_max, -k_max, k_max],
        aspect='equal'
    )

    # Colorbar
    fig = ax.figure
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_{10} P_{\rm kSZ}(k)$ [Mpc$^2$]')
    cbar.ax.tick_params(labelsize=20)

    # Axes
    ax.set_xlabel(r'$k_x$ [h $\cdot$ cMpc$^{-1}$]')
    ax.set_ylabel(r'$k_y$ [h $\cdot$ cMpc$^{-1}$]')

    # Optional reference scale (e.g. k = 0.1 Mpc⁻¹)
    circle = plt.Circle(
        (0, 0), 0.1,
        color='white',
        fill=False,
        linestyle='--',
        linewidth=1.5
    )
    ax.add_patch(circle)

plot_name = "kSZ_2D_power_map"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    plot_kSZ_2D_power(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_kSZ_2D_power,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"kSZ 2D Power Spectrum in $k$-space"
)

print(f"Saved: {plot_name}")


# %%
# =============================================================================
# 8c. Convert D_ℓ to μK² using CMB temperature
# =============================================================================

# CMB temperature today
T_CMB_0_K = 2.725  # K

# CMB temperature at z=5
z_obs = 5.0
T_CMB_z5_K = T_CMB_0_K #* (1 + z_obs)  # K

# Convert to μK
T_CMB_z5_uK = T_CMB_z5_K * 1e6  # μK

print(f"\n=== CMB TEMPERATURE ===")
print(f"T_CMB(z=0) = {T_CMB_0_K:.3f} K")
print(f"T_CMB(z=5) = {T_CMB_z5_K:.3f} K = {T_CMB_z5_uK:.2f} μK")

# Convert D_ℓ from dimensionless to μK²
Dl_uK2_DA = Dl_kSZ_muK2_DA * T_CMB_z5_uK**2
#Dl_uK2_comov = Dl_kSZ_muK2_comov * T_CMB_z5_uK**2

# Error on original — mirrors Cell 9 propagation
Cl_kSZ_DA_err    = P1d_err * 0.67**2 *36 / chi_comoving_Mpc**2
Dl_kSZ_DA_err    = ell_from_k * (ell_from_k + 1) * Cl_kSZ_DA_err / (2 * np.pi)
Dl_uK2_DA_err    = Dl_kSZ_DA_err * T_CMB_uK**2


print(f"\n=== D_ℓ CONVERSION ===")
print(f"Conversion factor: T_CMB²(z=5) = {T_CMB_z5_uK**2:.4e} μK²")
#print(f"D_ℓ range: [{np.nanmin(Dl_uK2[Dl_uK2>0]):.4e}, {np.nanmax(Dl_uK2):.4e}] μK²")

# =============================================================================
# PLOT 3: D_ℓ - Angular Power Spectrum in μK² (with CSV Overlays + Styled SPT)
# =============================================================================

import pandas as pd
from matplotlib.ticker import LogLocator, LogFormatterMathtext
def plot_Dl_muK2(ax):
    # 1. Plot current simulation data with error bars
    # Using the 'valid' mask to ensure we only plot physical, non-NaN values
    valid = ~np.isnan(Dl_uK2_DA) & (Dl_uK2_DA > 0) & (ell_from_k > 10)
    
    ax.errorbar(
        ell_from_k[valid], Dl_uK2_DA[valid],
        yerr=Dl_uK2_DA_err[valid],
        fmt='s-', color='darkblue', alpha=0.8,
        linewidth=2, markersize=5,
        capsize=3, capthick=1,
        label=r'Current Simulation $D_\ell$'
    )
    
    # Set to log-log scale explicitly since errorbar defaults to linear
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 2. Overlay Georgiev et al. CSV data
    georgiev_files = {
        'Georgiev_kSZ_zend_slow.csv':  ('gray',      '--', 'Georgiev+24 (Slow)'),
        'Georgiev_kSZ_zend_mid.csv':   ('black',     '-',  'Georgiev+24 (Mid)'),
        'Georgiev_kSZ_zend_rapid.csv': ('darkred',   ':',  'Georgiev+24 (Rapid)')
    }

    for filename, (color, ls, label) in georgiev_files.items():
        try:
            # Assuming CSVs have columns 'ell' and 'Dl' (or similar)
            df = pd.read_csv(filename)
            # Standard columns for such datasets are usually 'ell' and 'Dl'
            ax.plot(df.iloc[:,0], df.iloc[:,1], color=color, linestyle=ls, label=label, alpha=0.7)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    # -------------------------------------------------
    # 95% CL upper limit: SPT (Reichardt et al. 2021)
    # -------------------------------------------------
    ell_ul = 3000
    Dl_ul_95 = 2.5  # μK^2

    # Styled upper limit: Horizontal cap with downward arrow
    ax.errorbar(
        ell_ul, Dl_ul_95,
        yerr=0.6 * Dl_ul_95, # length of the arrow stem
        uplims=True,         # This creates the downward arrow
        fmt='none',          # No marker at the center
        color='red',
        linewidth=2,
        capsize=8,           # The horizontal bar at top
        label='SPT (Reichardt+ 2021, 95% CL)'
    )

    # Formatting
    ax.set_xlim(1e2, 10**(4.1))
    ax.set_ylim(1e-2, 1e2) # Slightly wider to see overlays

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell / 2\pi$ [$\mu$K$^2$]')
    ax.legend(loc='upper left', frameon=True)
    #ax.grid(True, which='both', alpha=0.2)

plot_name = "kSZ_Dl_muK2"

# -------- Preview PDF style --------
with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_Dl_muK2(ax)
    plt.show()

# -------- Save PDF + PNG --------
save_pdf_png(
    plot_func=plot_Dl_muK2,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"kSZ Angular Power Spectrum $D_\ell$"
)

print(f"Saved: {plot_name}")

# =============================================================================
# PLOT 4: D_ℓ in μK² - WITH OBSERVATIONS
# =============================================================================

def plot_Dl_muK2_with_obs(ax):
    valid = ~np.isnan(Dl_uK2_DA) & (Dl_uK2_DA > 0) & (ell_from_k > 10)

    ax.loglog(
        ell_from_k[valid], Dl_uK2_DA[valid],
        's-', color='orange', alpha=0.6,
        linewidth=1.5, markersize=4,
        label='This work (using $D_A$, incorrect)',
        zorder=5
    )

    # Alvarez et al. (2016)
    try:
        data = np.loadtxt("Alvarez2016_binned_smoothed.txt")
        ax.loglog(
            data[:, 0], data[:, 1],
            lw=2.5, ls='--', color='black',
            label='Alvarez et al. (2016)',
            alpha=0.7, zorder=5
        )
    except FileNotFoundError:
        pass

    # Reichardt et al. (2021)
    ax.errorbar(
        3000, 1.1,
        yerr=[[0.7], [1.0]],
        fmt='s', markersize=10,
        capsize=5, capthick=2,
        color='red',
        label='Reichardt et al. (2021)',
        zorder=15
    )

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax.legend(loc='best', frameon=True, framealpha=0.95)

    info_text = (
        f"Box: {user_params.BOX_LEN:.0f} Mpc, "
        f"z = {red_axis.min():.1f}-{red_axis.max():.1f}\n"
        f"$D_A$ = {D_A_physical:.0f} Mpc"
    )
    ax.text(
        0.05, 0.05, info_text,
        transform=ax.transAxes,
        fontsize=20,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )


plot_name = "kSZ_Dl_muK2_with_observations"

with mpl.rc_context(PDF_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_Dl_muK2_with_obs(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_Dl_muK2_with_obs,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"kSZ Angular Power $D_\ell$: Comparison with Observations"
)

print(f"Saved: {plot_name}")


# %%
import pandas as pd

# Save the CORRECT version (comoving distance)
df_comov = pd.DataFrame({
    'ell': ell_from_k[valid_comov],
    'Dl_muK2': Dl_uK2_comov[valid_comov]
})
df_comov.to_csv('ksz_power_spectrum_comoving.csv', index=False)
print(f"Saved {len(df_comov)} data points to ksz_power_spectrum_comoving.csv")

# Save the INCORRECT version (physical D_A) for comparison
df_DA = pd.DataFrame({
    'ell': ell_from_k[valid_DA],
    'Dl_muK2': Dl_uK2_DA[valid_DA]
})
df_DA.to_csv('ksz_power_spectrum_DA.csv', index=False)
print(f"Saved {len(df_DA)} data points to ksz_power_spectrum_DA.csv")

# Or save both in one file with different columns
df_both = pd.DataFrame({
    'ell': ell_from_k[valid_comov],
    'Dl_muK2_comoving': Dl_uK2_comov[valid_comov],
    'Dl_muK2_DA': Dl_uK2_DA[valid_comov]  # Using same ell values
})
df_both.to_csv('ksz_power_spectrum_comparison.csv', index=False)
print(f"Saved comparison with {len(df_both)} data points to ksz_power_spectrum_comparison.csv")

# %%
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np

def plot_Dl_muK2(ax):
    valid = ~np.isnan(Dl_uK2_DA) & (Dl_uK2_DA > 0) & (ell_from_k > 10)
    ax.loglog(
        ell_from_k[valid], Dl_uK2_DA[valid],
        's-', color='darkblue', alpha=0.5,
        linewidth=1.5, markersize=4,
        label=r'This work'
    )
    
    # -------------------------------------------------
    # Load and plot Park et al. 2013 data
    # -------------------------------------------------
    park_data = np.loadtxt('Park2013_pkSZ_L3.csv', delimiter=',')
    park_ell = park_data[:, 0]
    park_Dl = park_data[:, 1]
    ax.loglog(
        park_ell, park_Dl,
        'o-', color='green', alpha=0.7,
        linewidth=2, markersize=5,
        label='Park et al. (2013)'
    )
    
    # -------------------------------------------------
    # 95% CL upper limit: SPT (Reichardt et al. 2021)
    # -------------------------------------------------
    # -------------------------------------------------
# 95% CL upper limit: SPT (Reichardt et al. 2021)
# -------------------------------------------------
    # -------------------------------------------------
# 95% CL upper limit: SPT (Reichardt et al. 2021)
# -------------------------------------------------
    ell_ul = 3000
    Dl_ul_95 = 2.5  # μK^2
    ax.errorbar(
        ell_ul, Dl_ul_95,
        yerr=[[Dl_ul_95 * 0.1], [0.0]],  # Reduced from 0.8 to 0.3
        uplims=True,
        fmt='v',
        color='red',
        markersize=4,
        capsize=2,
        capthick=1.0,
        elinewidth=1.0,
        label='SPT (Reichardt+ 2021, 95% CL)'
    )
    ax.set_xlim(1e2, 1e5)
    ax.set_ylim(1e-2, 1e1)
    
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell / 2\pi$ [$\mu$K$^2$]')
    ax.legend(loc='lower center')

plot_name = "kSZ_Dl_muK2"

# -------- Preview PDF style --------
with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_Dl_muK2(ax)
    plt.show()

# -------- Save PDF + PNG --------
save_pdf_png(
    plot_func=plot_Dl_muK2,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"kSZ Angular Power Spectrum $D_\ell$"
)
print(f"Saved: {plot_name}")

# %%
# =============================================================================
# Cell 9: ERROR PROPAGATION: P(k) → C_ℓ → D_ℓ
# =============================================================================

print(f"\n{'='*60}")
print(f"ERROR PROPAGATION")
print(f"{'='*60}")

# You already have P1d_err from your azimuthal averaging
# This represents the sample variance in P(k)

# -----------------------------------------------------------------------------
# 1. Propagate to C_ℓ
# -----------------------------------------------------------------------------
# Transformation: C_ℓ = P(k) / D_A²
# Therefore: σ(C_ℓ) = σ(P(k)) / D_A²

Cl_kSZ_DA_err = P1d_err  / D_A_physical**2

print(f"\n=== C_ℓ ERRORS ===")
valid_Cl = ~np.isnan(Cl_kSZ_DA) & (Cl_kSZ_DA > 0)
if np.sum(valid_Cl) > 0:
    rel_err_Cl = Cl_kSZ_DA_err[valid_Cl] / Cl_kSZ_DA[valid_Cl]
    print(f"Relative error in C_ℓ: {np.nanmean(rel_err_Cl)*100:.1f}% (mean)")
    print(f"Range: {np.nanmin(rel_err_Cl)*100:.1f}% - {np.nanmax(rel_err_Cl)*100:.1f}%")

# -----------------------------------------------------------------------------
# 2. Propagate to D_ℓ (before CMB temperature scaling)
# -----------------------------------------------------------------------------
# Transformation: D_ℓ = ℓ(ℓ+1) C_ℓ / (2π)
# Therefore: σ(D_ℓ) = ℓ(ℓ+1) σ(C_ℓ) / (2π)

Dl_kSZ_muK2_DA_err = ell_from_k * (ell_from_k + 1) * Cl_kSZ_DA_err / (2 * np.pi)

print(f"\n=== D_ℓ ERRORS (dimensionless) ===")
valid_Dl = ~np.isnan(Dl_kSZ_muK2_DA) & (Dl_kSZ_muK2_DA > 0)
if np.sum(valid_Dl) > 0:
    rel_err_Dl = Dl_kSZ_muK2_DA_err[valid_Dl] / Dl_kSZ_muK2_DA[valid_Dl]
    print(f"Relative error in D_ℓ: {np.nanmean(rel_err_Dl)*100:.1f}% (mean)")
    print(f"Range: {np.nanmin(rel_err_Dl)*100:.1f}% - {np.nanmax(rel_err_Dl)*100:.1f}%")

# -----------------------------------------------------------------------------
# 3. Propagate to D_ℓ in μK²
# -----------------------------------------------------------------------------
# Transformation: D_ℓ[μK²] = D_ℓ × T_CMB²(z=5)
# Therefore: σ(D_ℓ[μK²]) = σ(D_ℓ) × T_CMB²(z=5)

# CMB temperature today
T_CMB_0_K = 2.725  # K

# CMB temperature at z=5
z_obs = 5.0
T_CMB_z5_K = T_CMB_0_K #* (1 + z_obs)  # K

# Convert to μK
T_CMB_z5_uK = T_CMB_z5_K * 1e6  # μK


Dl_uK2_DA_err = Dl_kSZ_muK2_DA_err * T_CMB_z5_uK**2

print(f"\n=== D_ℓ ERRORS (μK²) ===")
if np.sum(valid_Dl) > 0:
    print(f"Absolute error range: {np.nanmin(Dl_uK2_DA_err[valid_Dl]):.4e} - {np.nanmax(Dl_uK2_DA_err[valid_Dl]):.4e} μK²")
    # Relative error is the same as dimensionless D_ℓ
    print(f"Relative error: same as dimensionless D_ℓ ({np.nanmean(rel_err_Dl)*100:.1f}% mean)")

print(f"{'='*60}")

# %%
# =============================================================================
# CELL 9.5: Error propagation — unrotated vs rotated
# Mirrors Cell 9 exactly for both ensembles using Cell 8.5 quantities.
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# Propagate errors for both ensembles
# Same transformations as Cell 9:
#   σ(C_ℓ) = σ(P(k)) * 0.67² / chi²
#   σ(D_ℓ) = ℓ(ℓ+1) σ(C_ℓ) / (2π)
#   σ(D_ℓ [μK²]) = σ(D_ℓ) * T_CMB²
# =============================================================================

Cl_unrot_err = Pe_unrot  / chi_comoving_Mpc**2
Cl_rot_err   = Pe_rot   / chi_comoving_Mpc**2

Dl_unrot_err     = ell_unrot * (ell_unrot + 1) * Cl_unrot_err / (2 * np.pi)
Dl_rot_err       = ell_rot   * (ell_rot   + 1) * Cl_rot_err   / (2 * np.pi)

Dl_unrot_uK2_err = Dl_unrot_err * T_CMB_uK**2
Dl_rot_uK2_err   = Dl_rot_err   * T_CMB_uK**2

# =============================================================================
# Print statistics — mirroring Cell 9 printout
# =============================================================================

print(f"\n{'='*60}")
print(f"ERROR PROPAGATION — UNROTATED vs ROTATED")
print(f"{'='*60}")

for tag, Cl, Cl_err, Dl_uK2, Dl_uK2_err, ell in [
    ("Unrotated", Cl_unrot, Cl_unrot_err, Dl_unrot_uK2, Dl_unrot_uK2_err, ell_unrot),
    ("Rotated  ", Cl_rot,   Cl_rot_err,   Dl_rot_uK2,   Dl_rot_uK2_err,   ell_rot),
]:
    valid = ~np.isnan(Cl) & (Cl > 0) & (ell > 10)

    print(f"\n--- {tag} ---")

    print(f"  C_ℓ errors:")
    rel = Cl_err[valid] / Cl[valid]
    print(f"    Relative : {np.nanmean(rel)*100:.1f}% mean  |  "
          f"{np.nanmin(rel)*100:.1f}% – {np.nanmax(rel)*100:.1f}%")

    print(f"  D_ℓ errors [μK²]:")
    print(f"    Absolute : {np.nanmin(Dl_uK2_err[valid]):.4e} – "
          f"{np.nanmax(Dl_uK2_err[valid]):.4e} μK²")
    print(f"    Relative : {np.nanmean(rel)*100:.1f}% mean  "
          f"(same as C_ℓ by construction)")

print(f"\n{'='*60}")


# =============================================================================
# Plot — D_ℓ with error bands, unrotated vs rotated
# =============================================================================

def plot_Dl_with_errors(ax):
    vu = ~np.isnan(Dl_unrot_uK2) & (Dl_unrot_uK2 > 0) & (ell_unrot > 10)
    vr = ~np.isnan(Dl_rot_uK2)   & (Dl_rot_uK2   > 0) & (ell_rot   > 10)

    ax.loglog(ell_unrot[vu], Dl_unrot_uK2[vu],
              'o-', color='steelblue', lw=2, ms=5,
              label=r'Unrotated ($\theta=0^{\circ}$)')
    ax.fill_between(ell_unrot[vu],
                    (Dl_unrot_uK2 - Dl_unrot_uK2_err)[vu],
                    (Dl_unrot_uK2 + Dl_unrot_uK2_err)[vu],
                    alpha=0.25, color='steelblue')

    ax.loglog(ell_rot[vr], Dl_rot_uK2[vr],
              's--', color='darkorange', lw=2, ms=5,
              label=rf'Rotated ($\theta={angle_deg}^{{\circ}}$)')
    ax.fill_between(ell_rot[vr],
                    (Dl_rot_uK2 - Dl_rot_uK2_err)[vr],
                    (Dl_rot_uK2 + Dl_rot_uK2_err)[vr],
                    alpha=0.25, color='darkorange')

    ax.axvline(ell_box,     color='steelblue',  lw=1.2, ls=':', alpha=0.8,
               label=f'Artefact ℓ (unrot) ~ {ell_box:.0f}')
    ax.axvline(ell_box_rot, color='darkorange', lw=1.2, ls=':', alpha=0.8,
               label=f'Artefact ℓ (rot) ~ {ell_box_rot:.0f}')

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell\ [\mu\mathrm{K}^2]$')
    ax.legend(loc='best')


def plot_Cl_with_errors(ax):
    vu = ~np.isnan(Cl_unrot) & (Cl_unrot > 0) & (ell_unrot > 10)
    vr = ~np.isnan(Cl_rot)   & (Cl_rot   > 0) & (ell_rot   > 10)

    ax.loglog(ell_unrot[vu], Cl_unrot[vu],
              'o-', color='steelblue', lw=2, ms=5,
              label=r'Unrotated ($\theta=0^{\circ}$)')
    ax.fill_between(ell_unrot[vu],
                    (Cl_unrot - Cl_unrot_err)[vu],
                    (Cl_unrot + Cl_unrot_err)[vu],
                    alpha=0.25, color='steelblue')

    ax.loglog(ell_rot[vr], Cl_rot[vr],
              's--', color='darkorange', lw=2, ms=5,
              label=rf'Rotated ($\theta={angle_deg}^{{\circ}}$)')
    ax.fill_between(ell_rot[vr],
                    (Cl_rot - Cl_rot_err)[vr],
                    (Cl_rot + Cl_rot_err)[vr],
                    alpha=0.25, color='darkorange')

    ax.axvline(ell_box,     color='steelblue',  lw=1.2, ls=':', alpha=0.8,
               label=f'Artefact ℓ (unrot) ~ {ell_box:.0f}')
    ax.axvline(ell_box_rot, color='darkorange', lw=1.2, ls=':', alpha=0.8,
               label=f'Artefact ℓ (rot) ~ {ell_box_rot:.0f}')

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell$')
    ax.legend(loc='best')


_plots = [
    ("cell9p5_Dl_comparison_with_errors",
     plot_Dl_with_errors,
     r"$D_\ell\ [\mu\mathrm{K}^2]$: unrotated vs rotated with errors"),
    ("cell9p5_Cl_comparison_with_errors",
     plot_Cl_with_errors,
     r"$C_\ell$: unrotated vs rotated with errors"),
]

for plot_name, plot_func, title in _plots:
    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
        plot_func(ax)
        plt.show()
    save_pdf_png(plot_func=plot_func, plot_dir=plot_dir,
                 plot_name=plot_name, title=title)
    print(f"Saved: {plot_name}")

# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"CELL 9.5 SUMMARY")
print(f"{'='*60}")
for tag, Cl, Cl_err, ell in [
    ("Unrotated", Cl_unrot, Cl_unrot_err, ell_unrot),
    ("Rotated  ", Cl_rot,   Cl_rot_err,   ell_rot),
]:
    valid = ~np.isnan(Cl) & (Cl > 0) & (ell > 10)
    rel   = Cl_err[valid] / Cl[valid]
    print(f"  {tag} : mean relative error = {np.nanmean(rel)*100:.1f}%  |  "
          f"range {np.nanmin(rel)*100:.1f}% – {np.nanmax(rel)*100:.1f}%")
print(f"{'='*60}")

# %%
# =============================================================================
# Hoist Dl_orig_uK2 from Cell 8.5's local scope — needed for Cell 9b.5
# =============================================================================
# =============================================================================
# Hoist variables needed from Cell 8 and Cell 9 scope
# =============================================================================
T_CMB_uK      = 2.725e6
Dl_orig_uK2   = Dl_kSZ_muK2_DA * T_CMB_uK**2

# Error on original — mirrors Cell 9 propagation
Cl_kSZ_DA_err    = P1d_err * 0.67**2 *36 / chi_comoving_Mpc**2
Dl_kSZ_DA_err    = ell_from_k * (ell_from_k + 1) * Cl_kSZ_DA_err / (2 * np.pi)
Dl_uK2_DA_err    = Dl_kSZ_DA_err * T_CMB_uK**2


def plot_Dl_comparison_with_obs(ax):

    # --- Original full lightcone ---
    vo = (~np.isnan(Dl_orig_uK2) & (Dl_orig_uK2 > 0) & (ell_from_k > 10))
    ax.errorbar(
        ell_from_k[vo], Dl_orig_uK2[vo],
        yerr=Dl_uK2_DA_err[vo],
        fmt='^-', color='black', lw=1.5, ms=4,
        capsize=3, capthick=1,
        label=r'Original (full box, $128\times128$)',
    )
    ax.fill_between(
        ell_from_k[vo],
        (Dl_orig_uK2 - Dl_uK2_DA_err)[vo],
        (Dl_orig_uK2 + Dl_uK2_DA_err)[vo],
        alpha=0.2, color='black',
    )

    # --- Unrotated ---
    vu = (~np.isnan(Dl_unrot_uK2) & (Dl_unrot_uK2 > 0) & (ell_unrot > 10))
    ax.errorbar(
        ell_unrot[vu], Dl_unrot_uK2[vu],
        yerr=Dl_unrot_uK2_err[vu],
        fmt='o-', color='steelblue', lw=1.5, ms=4,
        capsize=3, capthick=1,
        label=r'Unrotated ($\theta=0^{\circ}$)',
    )
    ax.fill_between(
        ell_unrot[vu],
        (Dl_unrot_uK2 - Dl_unrot_uK2_err)[vu],
        (Dl_unrot_uK2 + Dl_unrot_uK2_err)[vu],
        alpha=0.2, color='steelblue',
    )

    # --- Rotated ---
    vr = (~np.isnan(Dl_rot_uK2) & (Dl_rot_uK2 > 0) & (ell_rot > 10))
    ax.errorbar(
        ell_rot[vr], Dl_rot_uK2[vr],
        yerr=Dl_rot_uK2_err[vr],
        fmt='s--', color='darkorange', lw=1.5, ms=4,
        capsize=3, capthick=1,
        label=rf'Rotated ($\theta={angle_deg}^{{\circ}}$)',
    )
    ax.fill_between(
        ell_rot[vr],
        (Dl_rot_uK2 - Dl_rot_uK2_err)[vr],
        (Dl_rot_uK2 + Dl_rot_uK2_err)[vr],
        alpha=0.2, color='darkorange',
    )

    # --- Reichardt+2021 patchy kSZ band ---
    ax.fill_between(
        ell_lo, Dl_lo, Dl_hi_interp,
        alpha=0.25, color='red',
        label='Patchy kSZ (Reichardt+ 2021)',
    )
    ax.plot(ell_lo, Dl_lo,        color='red', lw=1.0, ls='--')
    ax.plot(ell_lo, Dl_hi_interp, color='red', lw=1.0, ls='--')

    ax.set_xlim(1e2, 1e4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell / 2\pi\ [\mu\mathrm{K}^2]$')
    ax.legend(loc='lower center')


plot_name = "cell9b5_Dl_unrot_vs_rot_with_obs"

with mpl.rc_context(PNG_STYLE):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    plot_Dl_comparison_with_obs(ax)
    plt.show()

save_pdf_png(
    plot_func=plot_Dl_comparison_with_obs,
    plot_dir=plot_dir,
    plot_name=plot_name,
    title=r"kSZ $D_\ell$: unrotated vs rotated vs observations",
)

print(f"Saved: {plot_name}")

# %%
import py21cmfast as p21c
import inspect, os

# ── 1. What attributes does LightCone actually have? ──────────────────────────
print("=== LightCone attributes ===")
print([a for a in dir(lightcone) if not a.startswith('__')])

# ── 2. Sanity check density (known units: δ, so range ~ -1 to few) ────────────
print(f"\ndensity  range: {lightcone.density.min():.3f} to {lightcone.density.max():.3f}")
print(f"velocity range: {lightcone.velocity.min():.3e} to {lightcone.velocity.max():.3e}")
print(f"brightness_temp range: {lightcone.brightness_temp.min():.3f} to {lightcone.brightness_temp.max():.3f} mK")

# ── 3. Find the py21cmfast source file and grep for 'velocity' units ──────────
src_file = inspect.getfile(p21c)
src_dir  = os.path.dirname(src_file)
print(f"\npy21cmfast source dir: {src_dir}")
print("Files:", os.listdir(src_dir))

# ── 4. Search for velocity unit definition in source ─────────────────────────
for fname in os.listdir(src_dir):
    if fname.endswith('.py'):
        fpath = os.path.join(src_dir, fname)
        with open(fpath) as f:
            for i, line in enumerate(f, 1):
                if 'velocity' in line.lower() and ('unit' in line.lower() 
                    or 'km' in line.lower() or 'mpc' in line.lower()
                    or 'h0' in line.lower() or 'hubble' in line.lower()):
                    print(f"  {fname}:{i}: {line.rstrip()}")

# %%
import py21cmfast as p21c
import inspect, os

src_dir = os.path.dirname(inspect.getfile(p21c))

# Search inputs.py for BOX_LEN
with open(os.path.join(src_dir, 'inputs.py')) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'BOX_LEN' in line or 'box_len' in line.lower():
        start = max(0, i-2)
        end   = min(len(lines), i+5)
        print(f"--- line {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:4d}: {lines[j]}", end='')
        print()


