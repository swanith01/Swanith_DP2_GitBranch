import py21cmfast as p21c
from py21cmfast import plotting
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print(f"21cmFAST version is {p21c.__version__}")
print(f"Starting run at {datetime.now()}")

# Create directory structure
output_dir = "v3_lightcones_20Nov"
cache_dir = os.path.join(output_dir, "_cache")
os.makedirs(cache_dir, exist_ok=True)
print(f"Output directory: {output_dir}")
print(f"Cache directory: {cache_dir}")

# Set up user parameters for 1 Gpc box with 512³ resolution
user_params = p21c.UserParams(
    HII_DIM=512, 
    BOX_LEN=1000,  # 1 Gpc = 1000 Mpc
    KEEP_3D_VELOCITIES=True,
    USE_INTERPOLATION_TABLES=True  # Explicitly set to avoid warning
)

print(f"\nUser parameters:")
print(f"  HII_DIM: {user_params.HII_DIM}")
print(f"  BOX_LEN: {user_params.BOX_LEN} Mpc")
print(f"  Cell size: {user_params.cell_size} Mpc")

# Set up lightconer with velocity
lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
    min_redshift=7.0,
    max_redshift=12.0,
    quantities=('brightness_temp', 'density', 'velocity_z', 'xH_box'),
    resolution=user_params.cell_size,
    get_los_velocity=True,  # Enable LOS velocity
)

print(f"\nLightconer setup:")
print(f"  Redshift range: {lcn.lc_redshifts.min():.2f} - {lcn.lc_redshifts.max():.2f}")
print(f"  Number of slices: {len(lcn.lc_redshifts)}")
print(f"  get_los_velocity: True")

# Run lightcone (this will take a while!)
print(f"\nStarting lightcone calculation at {datetime.now()}")
print("This may take several hours for a 1 Gpc, 512³ box...")

lightcone = p21c.run_lightcone(
    lightconer=lcn,
    global_quantities=("brightness_temp", 'density', 'xH_box'),
    direc=cache_dir,
    user_params=user_params,
    write=True  # Write cache files
)

print(f"Lightcone calculation completed at {datetime.now()}")

# Check what fields are available
print("\nAvailable lightcone fields:")
for attr in dir(lightcone):
    if not attr.startswith('_') and hasattr(getattr(lightcone, attr), 'shape'):
        field = getattr(lightcone, attr)
        if hasattr(field, 'shape') and len(field.shape) == 3:
            print(f"  {attr}: {field.shape}")

# Save lightcone as HDF5
lightcone_file = os.path.join(output_dir, "lightcone_1Gpc_512.h5")
lightcone.save(lightcone_file)
print(f"\nLightcone saved to: {lightcone_file}")

# Create diagnostic plots
print("\nCreating diagnostic plots...")

# 1. Brightness temperature lightcone
fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
plotting.lightcone_sliceplot(lightcone, 'brightness_temp', ax=ax, fig=fig)
plt.savefig(os.path.join(output_dir, 'brightness_temp_lightcone.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: brightness_temp_lightcone.png")

# 2. Density lightcone
fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
plotting.lightcone_sliceplot(lightcone, 'density', ax=ax, fig=fig)
plt.savefig(os.path.join(output_dir, 'density_lightcone.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: density_lightcone.png")

# 3. xH_box lightcone
fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
plotting.lightcone_sliceplot(lightcone, 'xH_box', ax=ax, fig=fig)
plt.savefig(os.path.join(output_dir, 'xH_lightcone.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: xH_lightcone.png")

# 4. Velocity_z lightcone
if hasattr(lightcone, 'velocity_z'):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    plotting.lightcone_sliceplot(lightcone, 'velocity_z', ax=ax, fig=fig)
    plt.savefig(os.path.join(output_dir, 'velocity_z_lightcone.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: velocity_z_lightcone.png")

# 5. LOS velocity lightcone (if available)
if hasattr(lightcone, 'los_velocity'):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    plotting.lightcone_sliceplot(lightcone, 'los_velocity', ax=ax, fig=fig)
    plt.savefig(os.path.join(output_dir, 'los_velocity_lightcone.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: los_velocity_lightcone.png")

# 6. kSZ product field
print("\nCalculating kSZ product field...")
density = lightcone.density
xH = lightcone.xH_box
x_e = 1.0 - xH

# Get velocity (check which field is available)
if hasattr(lightcone, 'los_velocity'):
    v_los = lightcone.los_velocity
elif hasattr(lightcone, 'velocity_z'):
    v_los = lightcone.velocity_z
else:
    print("  Warning: No velocity field found!")
    v_los = None

if v_los is not None:
    # Convert velocity from Mpc/s to km/s
    Mpc_to_km = 3.086e19
    v_los_km_s = v_los * Mpc_to_km
    c_km_s = 3e5
    
    # kSZ in μK
    T_CMB_uK = 2.7255e6
    kSZ_uK = (1 + density) * x_e * v_los_km_s / c_km_s * T_CMB_uK
    
    # Add to lightcone
    lightcone.kSZ_uK = kSZ_uK
    
    # Plot kSZ
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    plotting.lightcone_sliceplot(lightcone, 'kSZ_uK', ax=ax, fig=fig)
    plt.savefig(os.path.join(output_dir, 'kSZ_lightcone.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: kSZ_lightcone.png")

print(f"\nAll done at {datetime.now()}!")
print(f"Output files saved in: {output_dir}/")
