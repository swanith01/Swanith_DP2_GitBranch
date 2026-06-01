#!/usr/bin/env python
# =============================================================================
# run_cell9_only.py
# Standalone script: loads all cached results and runs ONLY Cell 9
# (wedge filter diagnostic for kSZ²×21cm unsquared)
#
# Run on interactive node:
#   qsub -I -q workq -l select=1:ncpus=4:mem=16gb -l walltime=00:30:00
#   cd /user1/swanith/1Jun2026_kSZ_Sqr_21cm_sqr_code
#   conda activate p21c_v3
#   python run_cell9_only.py
# =============================================================================

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import os
import sys

# =============================================================================
# 1. Config — point at production cache
# =============================================================================
RUN_MODE      = "production"
CODE_DIR      = "/user1/swanith/1Jun2026_kSZ_Sqr_21cm_sqr_code"
CACHE_DIR     = os.path.join(CODE_DIR, "1Jun2026_kSZ_sqr_21cm_sqr", "cache")
PLOT_DIR      = os.path.join(CODE_DIR, "1Jun2026_kSZ_sqr_21cm_sqr", "plots")
KSZMAPS_DIR   = os.path.join(CACHE_DIR, "kSZ_maps")
Z_OBS         = 5.0
RANDOM_SEEDS  = list(range(1, 21))
HII_DIM       = 128
BOX_LEN       = 800.0

os.makedirs(PLOT_DIR, exist_ok=True)

print("="*60)
print("  run_cell9_only.py")
print(f"  CACHE  : {CACHE_DIR}")
print(f"  PLOTS  : {PLOT_DIR}")
print("="*60)

# =============================================================================
import h5py

seed           = 1
seed_cache_dir = os.path.join(CACHE_DIR, f"seed_{seed}")
lc_files       = [f for f in os.listdir(seed_cache_dir)
                  if f.startswith("LightCone") and f.endswith(".h5")]
if len(lc_files) == 0:
    print(f"✗ No LightCone h5 found in {seed_cache_dir}")
    sys.exit(1)

lc_file = os.path.join(seed_cache_dir, lc_files[0])
print(f"  Loading: {lc_file}")

with h5py.File(lc_file, "r") as f:
    brightness_temp = f["lightcones"]["brightness_temp"][:]
    xH_node         = f["global_quantities"]["xH_box"][:]
    node_redshifts  = f["node_redshifts"][:]

n_slices         = brightness_temp.shape[2]
lc_redshifts_arr = np.linspace(node_redshifts.min(), node_redshifts.max(), n_slices)

class LC:
    pass
lc                     = LC()
lc.brightness_temp     = brightness_temp
lc.lightcone_redshifts = lc_redshifts_arr
lc.node_redshifts      = node_redshifts
lc.global_xH           = xH_node
print(f"  ✓ brightness_temp {brightness_temp.shape}")
print(f"  z range: {lc_redshifts_arr.min():.2f} → {lc_redshifts_arr.max():.2f}")

ksz_map_file = os.path.join(KSZMAPS_DIR, f"kSZ_map_z{Z_OBS:.1f}_seed{seed}.npy")
if not os.path.exists(ksz_map_file):
    print(f"✗ kSZ map not found: {ksz_map_file}")
    sys.exit(1)
kSZ_map = np.load(ksz_map_file)
print(f"  ✓ kSZ map {kSZ_map.shape}  RMS={np.sqrt(np.mean(kSZ_map**2)):.3e}")

# =============================================================================
# 3. Plot style (minimal — no PDF_STYLE/PNG_STYLE needed)
# =============================================================================
plt.rcParams.update({
    'font.family'        : 'serif',
    'font.size'          : 14,
    'axes.labelsize'     : 14,
    'axes.titlesize'     : 13,
    'xtick.labelsize'    : 12,
    'ytick.labelsize'    : 12,
    'legend.fontsize'    : 11,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'axes.grid'          : False,
    'mathtext.fontset'   : 'cm',
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
})

# =============================================================================
# 4. Grid setup
# =============================================================================
pix_size_Mpc = BOX_LEN / HII_DIM
pix_area     = pix_size_Mpc**2
dk           = 2 * np.pi / (HII_DIM * pix_size_Mpc)
kx_2d        = np.fft.fftshift(np.fft.fftfreq(HII_DIM)) * HII_DIM * dk
ky_2d        = np.fft.fftshift(np.fft.fftfreq(HII_DIM)) * HII_DIM * dk
kgrid_2d     = np.sqrt(kx_2d[:, None]**2 + ky_2d[None, :]**2)
k_bins       = np.logspace(np.log10(dk), np.log10(kgrid_2d.max() * 0.9), 30)
k_centers    = 0.5 * (k_bins[:-1] + k_bins[1:])
kx_1d        = np.fft.fftfreq(HII_DIM, d=pix_size_Mpc) * 2 * np.pi
ky_1d        = np.fft.fftfreq(HII_DIM, d=pix_size_Mpc) * 2 * np.pi
cosmo        = FlatLambdaCDM(H0=67.77, Om0=0.3086)

# Squared kSZ map
kSZ2_map         = kSZ_map**2
kSZ2_centered    = kSZ2_map - np.mean(kSZ2_map)
fft_kSZ2_shifted = np.fft.fftshift(np.fft.fft2(kSZ2_centered))

# Redshift scan
lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
delta_z_thin = 0.5
z_centres    = np.arange(
    np.ceil(lc_redshifts[lc_redshifts > 0].min() / delta_z_thin) * delta_z_thin,
    min(lc_redshifts.max(), 20.0),
    delta_z_thin
)
print(f"\n  Scanning {len(z_centres)} redshift slices "
      f"(z={z_centres[0]:.1f}→{z_centres[-1]:.1f}, Δz={delta_z_thin})")

ell_target  = 3000
z_nodes_lc  = lc.node_redshifts[::-1]
xe_nodes_lc = 1.0 - lc.global_xH[::-1]

# =============================================================================
# 5. Filter scenarios
# =============================================================================
scenarios = {
    'A_nofilter'  : {'label': 'No filter\n(raw signal)',
                     'color': 'black',      'ls': '-',  'lw': 2.5},
    'B_optimistic': {'label': r'$k_\parallel > 0.01$ Mpc$^{-1}$',
                     'color': 'steelblue',  'ls': '--', 'lw': 2.0},
    'C_wedge_m3'  : {'label': r'Wedge $m=3$',
                     'color': 'darkorange', 'ls': '-.', 'lw': 2.0},
    'D_wedge_m5'  : {'label': r'Wedge $m=5$',
                     'color': 'crimson',    'ls': ':',  'lw': 2.0},
}

results_z = {k: {'z': [], 'D': []} for k in scenarios}

# =============================================================================
# 6. Run pipeline for each scenario
# =============================================================================
for key, meta in scenarios.items():
    label_short = meta['label'].replace('\n', ' ')
    print(f"\n  Scenario: {label_short}")

    for z0 in z_centres:
        z_lo = z0 - delta_z_thin / 2.0
        z_hi = z0 + delta_z_thin / 2.0
        idx_chunk = np.where(
            (lc_redshifts >= z_lo) & (lc_redshifts < z_hi)
        )[0]
        if len(idx_chunk) < 2:
            continue

        T21_chunk = np.asarray(
            lc.brightness_temp[:, :, idx_chunk], dtype=np.float64
        )
        n_los        = T21_chunk.shape[2]
        pix_size_los = pix_size_Mpc

        kz_1d    = np.fft.fftfreq(n_los, d=pix_size_los) * 2 * np.pi
        kx_3d    = kx_1d[:, None, None]
        ky_3d    = ky_1d[None, :, None]
        kz_3d    = kz_1d[None, None, :]
        kperp_3d = np.sqrt(kx_3d**2 + ky_3d**2)
        kpar_3d  = np.abs(kz_3d)

        T21_fft3d = np.fft.fftn(T21_chunk)

        if key == 'A_nofilter':
            filt3d = np.ones_like(kpar_3d)
        elif key == 'B_optimistic':
            filt3d = (kpar_3d > 0.01).astype(float)
        elif key == 'C_wedge_m3':
            filt3d = (kpar_3d > 3.0 * kperp_3d).astype(float)
        elif key == 'D_wedge_m5':
            filt3d = (kpar_3d > 5.0 * kperp_3d).astype(float)

        T21_filtered = np.real(np.fft.ifftn(T21_fft3d * filt3d))
        T21_2d       = np.mean(T21_filtered, axis=2)
        T21_cen      = T21_2d - np.mean(T21_2d)
        fft_T21_sh   = np.fft.fftshift(np.fft.fft2(T21_cen))

        cross_ps2d = (np.real(np.conj(fft_kSZ2_shifted) * fft_T21_sh)
                      * pix_area / HII_DIM**2)

        C_cross = np.full(len(k_centers), np.nan)
        for j in range(len(k_centers)):
            mask  = (kgrid_2d >= k_bins[j]) & (kgrid_2d < k_bins[j+1])
            n_pix = int(np.sum(mask))
            if n_pix > 0:
                C_cross[j] = np.mean(cross_ps2d[mask])

        chi_z0  = float(cosmo.comoving_distance(z0).value)
        ell_arr = k_centers * chi_z0
        D_cross = ell_arr * (ell_arr + 1) * C_cross / (2 * np.pi)

        idx_ell = np.argmin(np.abs(ell_arr - ell_target))
        if np.isfinite(D_cross[idx_ell]):
            results_z[key]['z'].append(z0)
            results_z[key]['D'].append(D_cross[idx_ell])

    D_arr = np.array(results_z[key]['D'])
    print(f"    {len(D_arr)} points  max|D|={np.nanmax(np.abs(D_arr)):.3e}")


# =============================================================================
# 7. Plot — 2 panels
# =============================================================================
print("\n  Plotting ...")

D_nofilt  = np.array(results_z['A_nofilter']['D'])
linthresh = float(np.nanpercentile(np.abs(D_nofilt[D_nofilt != 0]), 5)) \
            if len(D_nofilt) > 0 else 1e-9
linthresh = max(linthresh, 1e-12)
print(f"  symlog linthresh = {linthresh:.3e}")

xe_markers    = [(0.2, 'royalblue'), (0.5, 'forestgreen'), (0.9, 'firebrick')]
scenario_list = list(scenarios.items())

fig, (ax_sig, ax_xe) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# left panel — all scenarios overplotted
for key, meta in scenario_list:
    z_arr = np.array(results_z[key]['z'])
    D_arr = np.array(results_z[key]['D'])
    if len(z_arr) > 0:
        sort = np.argsort(z_arr)[::-1]
        ax_sig.plot(z_arr[sort], D_arr[sort],
                    color=meta['color'], lw=meta['lw'], ls=meta['ls'],
                    label=meta['label'].replace('\n', ' '))

ax_sig.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax_sig.set_yscale('symlog', linthresh=linthresh)
ax_sig.set_xlabel(r'Redshift $z$', fontsize=13)
ax_sig.set_ylabel(
    r'$\ell(\ell+1)C_\ell^{\rm kSZ^2\times 21cm}/2\pi$'
    f'  at $\\ell={ell_target}$', fontsize=12)
ax_sig.set_title(r'kSZ$^2\times$21cm (unsquared): filter comparison', fontsize=13)
ax_sig.invert_xaxis()
ax_sig.legend(loc='lower left', fontsize=11, framealpha=0.9)

for xe_val, color in xe_markers:
    z_xe = float(np.interp(xe_val, xe_nodes_lc, z_nodes_lc))
    ax_sig.axvline(z_xe, color=color, ls=':', lw=1.2, alpha=0.7)

ax_sig.text(0.03, 0.97,
            'No filter: physical signal survives\n'
            r'Any filter kills $k_\parallel\approx 0$ $\Rightarrow$ signal $\to 0$',
            transform=ax_sig.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# right panel — reionisation history
z_plot_xe = np.linspace(z_nodes_lc.min(), z_nodes_lc.max(), 500)
xe_plot   = np.interp(z_plot_xe, z_nodes_lc, xe_nodes_lc)
ax_xe.plot(z_plot_xe, xe_plot, color='black', lw=2.5)

for xe_val, color in xe_markers:
    z_xe = float(np.interp(xe_val, xe_nodes_lc, z_nodes_lc))
    ax_xe.axhline(xe_val, color=color, ls=':', lw=1.2, alpha=0.8)
    ax_xe.axvline(z_xe,   color=color, ls=':', lw=1.2, alpha=0.8)
    ax_xe.scatter([z_xe], [xe_val], color=color, s=60, zorder=5)
    ax_xe.text(z_xe + 0.2, xe_val + 0.03,
               rf'$x_e={xe_val}$', color=color, fontsize=11)

ax_xe.set_xlabel(r'Redshift $z$', fontsize=13)
ax_xe.set_ylabel(r'Ionisation fraction $x_e$', fontsize=13)
ax_xe.set_title('Reionisation history', fontsize=13)
ax_xe.set_ylim(-0.05, 1.15)
ax_xe.invert_xaxis()

fig.suptitle(
    r'kSZ$^2\times$21cm (unsquared): wedge removes $k_\parallel\approx 0$'
    r' $\Rightarrow$ signal collapses  (Zhou+25)',
    fontsize=13, fontweight='bold'
)

for ext, dpi in [('pdf', 300), ('png', 300)]:
    fname = os.path.join(PLOT_DIR, f'wedge_kills_unsquared_2panel.{ext}')
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    print(f'  saved: {fname}')
plt.close(fig)


print("\n" + "="*60)
print(f"SUPPRESSION SUMMARY  (RMS D_ell at ell={ell_target})")
print(f"{'Scenario':<32} {'RMS':>10} {'vs no-filter':>14}")
print("-"*60)
for key, meta in scenarios.items():
    D_arr = np.array(results_z[key]['D'])
    rms   = np.sqrt(np.nanmean(D_arr**2)) if len(D_arr) > 0 else np.nan
    supp  = rms / D_ref_rms if D_ref_rms > 0 else np.nan
    label = meta['label'].replace('\n', ' ')
    print(f"  {label:<30} {rms:>10.3e} {supp:>13.1%}")
print("="*60)
print("\n✓ DONE — plot saved to:", PLOT_DIR)

# =============================================================================
# 9. Per-scenario zoom plots — own y-axis so wiggles are visible
#    No-filter shown faintly in background for scale reference
# =============================================================================
print("\n  Plotting per-scenario zoom figures ...")

filtered_keys = ['B_optimistic', 'C_wedge_m3', 'D_wedge_m5']

z_A  = np.array(results_z['A_nofilter']['z'])
D_A  = np.array(results_z['A_nofilter']['D'])
sort_A = np.argsort(z_A)[::-1]

for key in filtered_keys:
    meta  = scenarios[key]
    z_arr = np.array(results_z[key]['z'])
    D_arr = np.array(results_z[key]['D'])

    if len(z_arr) < 3:
        print(f"  Skipping {key} — too few points")
        continue

    sort  = np.argsort(z_arr)[::-1]

    # auto linthresh from this signal's own amplitude
    nonzero = D_arr[D_arr != 0]
    lt = float(np.nanpercentile(np.abs(nonzero), 5)) if len(nonzero) > 0 else 1e-20
    lt = max(lt, 1e-25)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # faint no-filter background
    ax.plot(z_A[sort_A], D_A[sort_A],
            color='black', lw=1.0, alpha=0.18, ls='-',
            label='No filter (background ref)')

    # filtered signal — full colour
    ax.plot(z_arr[sort], D_arr[sort],
            color=meta['color'], lw=2.2, ls=meta['ls'],
            label=meta['label'].replace('\n', ' '))

    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.set_yscale('symlog', linthresh=lt)
    ax.set_xlabel(r'Redshift $z$', fontsize=13)
    ax.set_ylabel(
        r'$\ell(\ell+1)C_\ell^{\rm kSZ^2\times 21cm}/2\pi$'
        f'  at $\\ell={ell_target}$', fontsize=12)
    ax.invert_xaxis()

    # xe vertical lines
    for xe_val, color in xe_markers:
        z_xe = float(np.interp(xe_val, xe_nodes_lc, z_nodes_lc))
        ax.axvline(z_xe, color=color, ls=':', lw=1.2, alpha=0.8)
        ax.text(z_xe + 0.15, ax.get_ylim()[1],
                rf'$x_e={xe_val}$', color=color,
                fontsize=9, rotation=90, va='top')

    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_title(
        rf'kSZ$^2\times$21cm (unsquared, $\ell={ell_target}$) — '
        + meta['label'].replace('\n', ' ') + '\n(own y-axis — no-filter shown faintly for scale)',
        fontsize=12)

    safe_key = key.replace('/', '_')
    for ext, dpi in [('pdf', 300), ('png', 300)]:
        fname = os.path.join(PLOT_DIR, f'wedge_zoom_{safe_key}.{ext}')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: wedge_zoom_{safe_key}")

print("\n✓ Per-scenario zoom plots complete")
