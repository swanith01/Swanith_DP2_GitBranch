# ============================================================
# kSZ Power Spectrum: RESOLUTION CONVERGENCE TEST
# ============================================================
# Fixed box: 800 Mpc
# Resolutions: 32³, 64³, 128³, 256³, 512³
# Output:
#   • P_{q⊥}(k) at selected redshifts for each resolution
#   • D_ell(ℓ) for each resolution
#   • D_ell ratio relative to 128³ reference
#   • D_3000 and P_{q⊥}(k=0.1 Mpc⁻¹) vs N (convergence)
#   • k_Nyquist and k_box marked on spectrum plots
# ============================================================

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const
from scipy.integrate import quad
import astropy.units as u

try:
    import py21cmfast as p21
except Exception as e:
    raise RuntimeError("py21cmfast is required.") from e

# ============================================================
# Plot style (mirrors parent code)
# ============================================================
plt.rcParams.update({
    'font.family'       : 'serif',
    'font.serif'        : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'  : 'cm',
    'font.size'         : 20,
    'axes.labelsize'    : 20,
    'axes.titlesize'    : 20,
    'xtick.labelsize'   : 14,
    'ytick.labelsize'   : 14,
    'legend.fontsize'   : 14,
    'figure.titlesize'  : 20,
    'xtick.direction'   : 'in',   'ytick.direction'   : 'in',
    'xtick.top'         : True,   'ytick.right'       : True,
    'xtick.major.size'  : 6,      'ytick.major.size'  : 6,
    'xtick.minor.size'  : 3,      'ytick.minor.size'  : 3,
    'xtick.major.width' : 1.0,    'ytick.major.width' : 1.0,
    'xtick.minor.width' : 0.8,    'ytick.minor.width' : 0.8,
    'axes.linewidth'    : 1.2,
    'lines.linewidth'   : 2.0,
    'lines.markersize'  : 5,
    'axes.grid'         : False,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.05,
})
import matplotlib as mpl
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True

# ============================================================
# Directories
# ============================================================
BASE_DIR  = "ksz_convergence/resolution/"
CACHE_DIR = os.path.join(BASE_DIR, "cache/")
PLOT_DIR  = os.path.join(BASE_DIR, "plots/")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

p21.config['direc'] = CACHE_DIR
os.environ['OMP_NUM_THREADS'] = '32'

# ============================================================
# Physical constants
# ============================================================
sigma_T  = const.sigma_T.cgs.value
MPC_CM   = 3.0856775814913673e24
c_cms    = const.c.cgs.value
T_CMB_K  = 2.7255

def ne0_cgs(Y_He=0.24, include_He=True):
    m_p    = const.m_p.cgs.value
    rho_c0 = cosmo.critical_density0.cgs.value
    X_H    = 1.0 - Y_He
    n_H0   = X_H * (cosmo.Ob0 * rho_c0) / m_p
    y      = Y_He / (4.0 * X_H) if include_He else 0.0
    return n_H0 * (1.0 + y)

ne0 = ne0_cgs()

# ============================================================
# Growth factor / velocity conversion
# ============================================================
def growth_factor(z):
    def integrand(zp):
        return (1.0 + zp) / cosmo.H(zp).value**3
    val,  _ = quad(integrand, z,   np.inf)
    norm, _ = quad(integrand, 0.0, np.inf)
    return (cosmo.H(z).value * val) / (cosmo.H(0).value * norm)

def growth_rate(z):
    return cosmo.Om(z) ** 0.55

def velocity_conversion_factor(z):
    return growth_factor(z) * growth_rate(z) * cosmo.H(z).value / (1.0 + z)

# ============================================================
# Scan configuration
# ============================================================
BOX_LEN  = 800.0                        # [Mpc] — fixed
HII_DIMS = [32, 64, 128, 256, 512]      # resolution scan

print("Resolution convergence scan configuration:")
print(f"  Fixed BOX_LEN = {BOX_LEN:.0f} Mpc")
for N in HII_DIMS:
    dx  = BOX_LEN / N
    kNy = np.pi / dx                    # Nyquist wavenumber [Mpc⁻¹]
    kbx = 2.0 * np.pi / BOX_LEN
    print(f"  N={N:>4}³  →  Δx={dx:7.3f} Mpc  "
          f"k_box={kbx:.4f}  k_Nyq={kNy:.4f} Mpc⁻¹")

# EoR redshifts (z ≥ 5)
ZS_EOR = sorted([z for z in np.arange(5.0, 15.5, 0.5)], reverse=True)
print(f"\nRedshift slices: {ZS_EOR}")

# Reference resolution for ratio plots (128³ matches parent code)
REF_N = 128

# ============================================================
# Field & spectrum helpers (identical to parent + boxsize script)
# ============================================================
def run_coeval_fields(z, HII_DIM, BOX_LEN):
    coeval = p21.run_coeval(
        redshift    = float(z),
        user_params = {"HII_DIM": int(HII_DIM), "BOX_LEN": float(BOX_LEN)},
        write       = False,
    )
    fac   = velocity_conversion_factor(z)
    delta = coeval.density
    xH    = coeval.xH_box
    vx    = coeval.lowres_vx * fac * 1e5
    vy    = coeval.lowres_vy * fac * 1e5
    vz    = coeval.lowres_vz * fac * 1e5
    return delta, xH, vx, vy, vz


def qperp_power(delta, xH, vx, vy, vz, BOX_LEN, nbins=None):
    chi = 1.0 - xH
    w   = (1.0 + delta) * chi
    qx  = w * vx;  qy = w * vy;  qz = w * vz

    N = qx.shape[0]
    L = float(BOX_LEN)
    d = L / N
    V = L**3

    kfreq      = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2         = kx**2 + ky**2 + kz**2
    k_mag      = np.sqrt(k2)
    k2_safe    = np.where(k2 == 0.0, np.inf, k2)

    Qx = np.fft.fftn(qx) * d**3
    Qy = np.fft.fftn(qy) * d**3
    Qz = np.fft.fftn(qz) * d**3

    kdotQ_k2 = (Qx*kx + Qy*ky + Qz*kz) / k2_safe
    Qx_perp  = Qx - kdotQ_k2 * kx
    Qy_perp  = Qy - kdotQ_k2 * ky
    Qz_perp  = Qz - kdotQ_k2 * kz

    Qperp2 = (np.abs(Qx_perp)**2 +
               np.abs(Qy_perp)**2 +
               np.abs(Qz_perp)**2)
    p_flat = (Qperp2 / V / 2.0).ravel()
    k_flat = k_mag.ravel()

    if nbins is None:
        nbins = max(2, int(np.ceil(np.cbrt(N) * 8)))

    pos_k = np.abs(kfreq[kfreq > 0.0])
    kmin  = pos_k.min() if pos_k.size > 0 else 1e-6
    kmax  = np.abs(kfreq).max() * np.sqrt(3.0)
    if kmax <= kmin:
        kmax = kmin * 10.0

    bins  = np.geomspace(kmin, kmax, nbins)
    digit = np.digitize(k_flat, bins)

    k_bins, P_bins, P_std = [], [], []
    for i in range(1, len(bins)):
        mask = digit == i
        if not np.any(mask):
            continue
        k_bins.append(k_flat[mask].mean())
        P_bins.append(p_flat[mask].mean())
        P_std.append(p_flat[mask].std())

    return np.array(k_bins), np.array(P_bins), np.array(P_std)


def interp_loglog(xq, xp, fp):
    xp = np.asarray(xp);  fp = np.asarray(fp)
    m  = (xp > 0) & (fp > 0)
    lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
    return np.exp(np.interp(lq, np.log(xp[m]), np.log(fp[m])))


def compute_Dell(results_qperp_res, ZS_asc):
    pref = (sigma_T * ne0 / c_cms)**2

    chi_mpc  = np.array([cosmo.comoving_distance(z).value for z in ZS_asc])
    dchi_mpc = np.abs(np.gradient(chi_mpc))
    dchi_cm  = dchi_mpc * MPC_CM

    xe_arr = np.array([1.0 - results_qperp_res[z]['xH_mean']
                       for z in ZS_asc])

    tau = np.zeros(len(ZS_asc))
    for i in range(len(ZS_asc) - 1):
        zmid   = 0.5 * (ZS_asc[i]  + ZS_asc[i+1])
        xe_mid = 0.5 * (xe_arr[i]  + xe_arr[i+1])
        tau[i+1] = tau[i] + sigma_T * ne0 * xe_mid * (1.0+zmid)**2 * dchi_cm[i]

    k_max_sim = min(results_qperp_res[z]['k'].max() for z in ZS_asc)
    k_min_sim = max(results_qperp_res[z]['k'].min() for z in ZS_asc)
    s_min     = chi_mpc.min()
    s_max     = chi_mpc.max()

    ells_full = np.unique(
        np.round(np.logspace(2.0, 4.5, 80)).astype(int)).astype(int)
    ell_max_valid = int(k_max_sim * s_min)
    ell_min_valid = int(k_min_sim * s_max)
    ells = ells_full[(ells_full >= ell_min_valid) &
                     (ells_full <= ell_max_valid)]

    if len(ells) == 0:
        return np.array([]), np.array([]), np.array([])

    k_list = [results_qperp_res[z]['k']      for z in ZS_asc]
    P_list = [results_qperp_res[z]['Pqperp'] for z in ZS_asc]
    S_list = [results_qperp_res[z]['Pstd']   for z in ZS_asc]

    C_ell     = np.zeros(len(ells))
    var_C_ell = np.zeros(len(ells))

    for i in range(len(ZS_asc)):
        s_mpc = chi_mpc[i]
        if s_mpc <= 0.0:
            continue
        a_i  = 1.0 / (1.0 + ZS_asc[i])
        vis2 = np.exp(-2.0 * tau[i])
        w    = vis2 / (s_mpc**2 * a_i**4) * dchi_mpc[i]
        k_ell = ells / s_mpc

        P_now      = interp_loglog(k_ell, k_list[i], P_list[i])
        S_now      = interp_loglog(k_ell, k_list[i], S_list[i])
        C_ell     += pref * w * P_now * MPC_CM**2
        var_C_ell += (pref * w * S_now * MPC_CM**2)**2

    sigma_C = np.sqrt(var_C_ell)
    prefD   = ells * (ells + 1.0) / (2.0 * np.pi) * T_CMB_K**2 * 1e12
    return ells, prefD * C_ell, prefD * sigma_C


# ============================================================
# Main loop
# ============================================================
all_results  = {}
dell_results = {}

for N in HII_DIMS:
    dx  = BOX_LEN / N
    tag = f"N{N}_box{int(BOX_LEN)}"
    cache_file = os.path.join(CACHE_DIR, f"qperp_{tag}.pkl")

    print(f"\n{'='*60}")
    print(f"HII_DIM={N}³  |  Δx={dx:.3f} Mpc  |  BOX_LEN={BOX_LEN:.0f} Mpc")
    print(f"  k_box  = {2*np.pi/BOX_LEN:.4f} Mpc⁻¹")
    print(f"  k_Nyq  = {np.pi/dx:.4f} Mpc⁻¹")
    print(f"{'='*60}")

    # ── Load or compute ──────────────────────────────────────
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            results_res = pickle.load(f)
        print(f"  Cache loaded: {len(results_res)} redshifts")
    else:
        results_res = {}

    missing = [z for z in ZS_EOR if z not in results_res]
    if missing:
        print(f"  Computing {len(missing)} redshifts...")
        for z in missing:
            print(f"    z={z:.1f}...", end=' ', flush=True)
            try:
                delta, xH, vx, vy, vz = run_coeval_fields(z, N, BOX_LEN)
                k_q, P_q, P_std = qperp_power(delta, xH, vx, vy, vz, BOX_LEN)
                results_res[z] = {
                    'k'      : k_q,
                    'Pqperp' : P_q,
                    'Pstd'   : P_std,
                    'xH_mean': float(xH.mean()),
                    'HII_DIM': N,
                    'dx_mpc' : dx,
                }
                print(f"<xH>={xH.mean():.3f}  "
                      f"k_range=[{k_q.min():.4f}, {k_q.max():.4f}] Mpc⁻¹")
            except Exception as e:
                print(f"FAILED: {e}")

        with open(cache_file, 'wb') as f:
            pickle.dump(results_res, f)
        print(f"  Cache saved → {cache_file}")
    else:
        print("  All redshifts already cached.")

    all_results[N] = results_res

    # ── D_ell ────────────────────────────────────────────────
    ZS_asc = sorted([z for z in results_res.keys() if z >= 5.0])
    if len(ZS_asc) >= 2:
        ells, D_ell, sigma_D = compute_Dell(results_res, ZS_asc)
        dell_results[N] = (ells, D_ell, sigma_D)
        if len(ells) > 0:
            D3000 = float(np.interp(3000.0, ells, D_ell))
            print(f"  D_3000 = {D3000:.4g} μK²")


# ============================================================
# Summary table
# ============================================================
print("\n" + "="*70)
print(f"{'N':>6}  {'Δx [Mpc]':>10}  {'k_Nyq':>8}  "
      f"{'D_3000 [μK²]':>14}  {'ell_min':>8}  {'ell_max':>8}")
print("-"*70)
for N in HII_DIMS:
    dx  = BOX_LEN / N
    kNy = np.pi / dx
    if N in dell_results and len(dell_results[N][0]) > 0:
        ells, D_ell, _ = dell_results[N]
        D3000 = float(np.interp(3000.0, ells, D_ell))
        print(f"{N:>6}  {dx:>10.3f}  {kNy:>8.4f}  "
              f"{D3000:>14.4g}  {ells.min():>8}  {ells.max():>8}")
    else:
        print(f"{N:>6}  {dx:>10.3f}  {kNy:>8.4f}  {'N/A':>14}")


# ============================================================
# Plot colours and linestyles
# ============================================================
colors_N  = plt.cm.plasma(np.linspace(0.1, 0.9, len(HII_DIMS)))
ls_styles = ['-', '--', '-.', ':', (0,(3,1,1,1))]


# ============================================================
# Plot 1: D_ell for all resolutions
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xscale('log');  ax.set_yscale('log')
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$D_\ell\ [\mu\mathrm{K}^2]$')
ax.set_title(rf'Patchy kSZ: Resolution Convergence'
             f'\n$L = {BOX_LEN:.0f}$ Mpc (fixed box)')

for (N, col, ls) in zip(HII_DIMS, colors_N, ls_styles):
    if N not in dell_results or len(dell_results[N][0]) == 0:
        continue
    ells, D_ell, sigma_D = dell_results[N]
    dx = BOX_LEN / N
    ax.plot(ells, D_ell, color=col, ls=ls, lw=2.0,
            label=rf'$N={N}^3$, $\Delta x={dx:.2f}$ Mpc')
    ax.fill_between(ells,
                    np.maximum(D_ell - sigma_D, 1e-8),
                    D_ell + sigma_D,
                    color=col, alpha=0.12)

ax.errorbar(3000, 1.1, yerr=[[0.7], [1.0]],
            fmt='s', ms=8, capsize=5, capthick=2,
            color='red', zorder=10, label='Reichardt+2021')

ax.legend(frameon=False, fontsize=12)
plt.tight_layout()
stem = os.path.join(PLOT_DIR, "Dell_resolution_convergence")
plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
print(f"\nSaved → {stem}.pdf / .png")
plt.close()


# ============================================================
# Plot 2: D_ell ratio relative to reference (128³)
# ============================================================
if REF_N in dell_results and len(dell_results[REF_N][0]) > 0:
    ells_ref, D_ref, _ = dell_results[REF_N]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(rf'$D_\ell / D_\ell^{{N={REF_N}^3}}$')
    ax.set_title(rf'Resolution Convergence: $D_\ell$ Ratio'
                 f'\nReference: $N={REF_N}^3$, $L={BOX_LEN:.0f}$ Mpc')
    ax.axhline(1.0, color='k', lw=1.0, ls=':')
    ax.axhspan(0.95, 1.05, color='gray', alpha=0.10, label='±5%')
    ax.axhspan(0.90, 1.10, color='gray', alpha=0.05, label='±10%')

    for (N, col, ls) in zip(HII_DIMS, colors_N, ls_styles):
        if N == REF_N or N not in dell_results:
            continue
        ells_N, D_N, _ = dell_results[N]
        if len(ells_N) == 0:
            continue
        ell_common   = ells_N[(ells_N >= ells_ref.min()) &
                               (ells_N <= ells_ref.max())]
        if len(ell_common) == 0:
            continue
        D_N_interp   = np.interp(ell_common, ells_N,   D_N)
        D_ref_interp = np.interp(ell_common, ells_ref, D_ref)
        ratio = D_N_interp / np.where(D_ref_interp > 0, D_ref_interp, np.nan)
        dx = BOX_LEN / N
        ax.plot(ell_common, ratio, color=col, ls=ls, lw=2.0,
                label=rf'$N={N}^3$, $\Delta x={dx:.2f}$ Mpc')

    ax.legend(frameon=False, fontsize=12)
    ax.set_ylim(0.4, 2.0)
    plt.tight_layout()
    stem = os.path.join(PLOT_DIR, "Dell_ratio_resolution")
    plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
    print(f"Saved → {stem}.pdf / .png")
    plt.close()


# ============================================================
# Plot 3: P_{q⊥}(k) at z=7 for all resolutions
#         (k_box and k_Nyq marked per resolution)
# ============================================================
Z_SHOW = 7.0

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xscale('log');  ax.set_yscale('log')
ax.set_xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
ax.set_ylabel(r'$P_{q_\perp}(k)\ [\mathrm{cm}^2\,\mathrm{s}^{-2}\,\mathrm{Mpc}^3]$')
ax.set_title(rf'$P_{{q_\perp}}(k)$ at $z={Z_SHOW}$: Resolution Convergence'
             f'\n$L = {BOX_LEN:.0f}$ Mpc (fixed box)')

for (N, col, ls) in zip(HII_DIMS, colors_N, ls_styles):
    if N not in all_results or Z_SHOW not in all_results[N]:
        continue
    res = all_results[N][Z_SHOW]
    k, P, S = res['k'], res['Pqperp'], res['Pstd']
    dx = BOX_LEN / N
    ax.plot(k, P, color=col, ls=ls, lw=2.0,
            label=rf'$N={N}^3$, $\Delta x={dx:.2f}$ Mpc')
    ax.fill_between(k, P - S, P + S, color=col, alpha=0.12)

    # Mark k_Nyquist for each resolution as vertical dashed line
    k_nyq = np.pi / dx
    ax.axvline(k_nyq, color=col, ls=':', lw=0.8, alpha=0.7)

# Mark k_box (same for all: fixed BOX_LEN)
k_box = 2.0 * np.pi / BOX_LEN
ax.axvline(k_box, color='gray', ls=':', lw=1.2, alpha=0.8,
           label=rf'$k_{{\rm box}} = 2\pi/L$')

ax.legend(frameon=False, fontsize=11)
plt.tight_layout()
stem = os.path.join(PLOT_DIR, f"Pqperp_z{Z_SHOW}_resolution")
plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
print(f"Saved → {stem}.pdf / .png")
plt.close()


# ============================================================
# Plot 4: D_3000 vs N (resolution convergence diagnostic)
# ============================================================
N_vals, D3000_vals, dx_vals = [], [], []
for N in HII_DIMS:
    if N not in dell_results or len(dell_results[N][0]) == 0:
        continue
    ells, D_ell, _ = dell_results[N]
    N_vals.append(N)
    D3000_vals.append(float(np.interp(3000.0, ells, D_ell)))
    dx_vals.append(BOX_LEN / N)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: D_3000 vs N (cell count)
ax = axes[0]
ax.plot(N_vals, D3000_vals, 'o-', color='steelblue', lw=2.0, ms=8)
if REF_N in dell_results and len(dell_results[REF_N][0]) > 0:
    ells_r, D_r, _ = dell_results[REF_N]
    D3000_ref = float(np.interp(3000.0, ells_r, D_r))
    ax.axhline(D3000_ref, color='gray', ls='--', lw=1.2,
               label=rf'$N={REF_N}^3$ reference')
    ax.axhspan(D3000_ref * 0.95, D3000_ref * 1.05,
               color='gray', alpha=0.12, label='±5%')
ax.set_xlabel(r'Grid cells per side $N$')
ax.set_ylabel(r'$D_{3000}\ [\mu\mathrm{K}^2]$')
ax.set_title(r'$D_{3000}$ vs. Resolution')
ax.legend(frameon=False, fontsize=12)

# Right: D_3000 vs Δx (cell size)
ax = axes[1]
ax.plot(dx_vals, D3000_vals, 'o-', color='darkorange', lw=2.0, ms=8)
if REF_N in dell_results and len(dell_results[REF_N][0]) > 0:
    ells_r, D_r, _ = dell_results[REF_N]
    D3000_ref = float(np.interp(3000.0, ells_r, D_r))
    ax.axhline(D3000_ref, color='gray', ls='--', lw=1.2,
               label=rf'$N={REF_N}^3$ reference')
    ax.axhspan(D3000_ref * 0.95, D3000_ref * 1.05,
               color='gray', alpha=0.12, label='±5%')
ax.invert_xaxis()                              # smaller Δx = better resolution → right
ax.set_xlabel(r'Cell size $\Delta x\ [\mathrm{Mpc}]$')
ax.set_ylabel(r'$D_{3000}\ [\mu\mathrm{K}^2]$')
ax.set_title(r'$D_{3000}$ vs. Cell Size')
ax.legend(frameon=False, fontsize=12)

plt.suptitle(rf'Resolution Convergence  ($L={BOX_LEN:.0f}$ Mpc)',
             fontsize=20, y=1.01)
plt.tight_layout()
stem = os.path.join(PLOT_DIR, "D3000_vs_resolution")
plt.savefig(stem + ".pdf", bbox_inches='tight')
plt.savefig(stem + ".png", dpi=300, bbox_inches='tight')
print(f"Saved → {stem}.pdf / .png")
plt.close()


# ============================================================
# Plot 5: P_{q⊥}(k=0.1 Mpc⁻¹) at z=7 vs N
#         (intermediate-scale mode convergence)
# ============================================================
K_PROBE = 0.1   # [Mpc⁻¹]

N_vals2, Pk_vals = [], []
for N in HII_DIMS:
    if N not in all_results or Z_SHOW not in all_results[N]:
        continue
    res = all_results[N][Z_SHOW]
    k, P = res['k'], res['Pqperp']
    if k.min() <= K_PROBE <= k.max():
        N_vals2.append(N)
        Pk_vals.append(float(np.interp(K_PROBE, k, P)))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(N_vals2, Pk_vals, 's-', color='steelblue', lw=2.0, ms=9)
ax.set_xlabel(r'Grid cells per side $N$')
ax.set_ylabel(rf'$P_{{q_\perp}}(k={K_PROBE}\ \mathrm{{Mpc}}^{{-1}})$'
              r'  $[\mathrm{cm}^2\,\mathrm{s}^{-2}\,\mathrm{Mpc}^3]$')
ax.set_title(rf'Resolution Convergence at $k={K_PROBE}$ Mpc$^{{-1}}$, $z={Z_SHOW}$'
             f'\n$L = {BOX_LEN:.0f}$ Mpc')

# Reference line at REF_N value
if REF_N in [n for n in N_vals2]:
    ref_idx = N_vals2.index(REF_N)
    ax.axhline(Pk_vals[ref_idx], color='gray', ls='--', lw=1.2,
               label=rf'$N={REF_N}^3$ reference')
    ax.axhspan(Pk_vals[ref_idx]*0.95, Pk_vals[ref_idx]*1.05,
               color='gray', alpha=0.12, label='±5%')
    ax.legend(frameon=False, fontsize=12)

plt.tight_layout()
stem = os.path.join(PLOT_DIR, f"Pk01_z{Z_SHOW}_vs_resolution")
plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
print(f"Saved → {stem}.pdf / .png")
plt.close()


# ============================================================
# Plot 6: Ionisation history <x_e>(z) for each resolution
#         (sanity check: physics should not depend on N)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlabel(r'Redshift $z$')
ax.set_ylabel(r'$\langle x_e \rangle$')
ax.set_title(rf'Ionisation history: Resolution Consistency Check'
             f'\n$L={BOX_LEN:.0f}$ Mpc')

for (N, col, ls) in zip(HII_DIMS, colors_N, ls_styles):
    if N not in all_results:
        continue
    res = all_results[N]
    zs_sorted = sorted(res.keys())
    xe_vals   = [1.0 - res[z]['xH_mean'] for z in zs_sorted]
    dx = BOX_LEN / N
    ax.plot(zs_sorted, xe_vals, color=col, ls=ls, lw=2.0,
            label=rf'$N={N}^3$, $\Delta x={dx:.2f}$ Mpc')

ax.axhline(0.5, color='gray', ls=':', lw=1.0, alpha=0.6)
ax.legend(frameon=False, fontsize=12)
plt.tight_layout()
stem = os.path.join(PLOT_DIR, "xe_history_resolution")
plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
print(f"Saved → {stem}.pdf / .png")
plt.close()

print("\nResolution convergence test complete.")
print(f"All outputs in: {PLOT_DIR}")
