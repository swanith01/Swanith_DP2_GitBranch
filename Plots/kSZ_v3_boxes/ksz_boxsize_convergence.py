# ============================================================
# kSZ Power Spectrum: BOX-SIZE CONVERGENCE TEST
# ============================================================
# Boxes: 200, 400, 600, 800, 1000 Mpc
# Fixed resolution: Δx = 800/128 ≈ 6.25 Mpc  →  HII_DIM = BOX_LEN / Δx
# Redshifts: EoR only (z ≥ 5), same grid as parent code
# Output:
#   • Per-box P_{q⊥}(k) at selected redshifts
#   • D_ell(ℓ) for each box, all on one plot
#   • D_ell ratio relative to 800 Mpc reference box
#   • D_3000 table and convergence plot
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
BASE_DIR  = "ksz_convergence/boxsize/"
CACHE_DIR = os.path.join(BASE_DIR, "cache/")
PLOT_DIR  = os.path.join(BASE_DIR, "plots/")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

p21.config['direc'] = CACHE_DIR
os.environ['OMP_NUM_THREADS'] = '32'

# ============================================================
# Physical constants
# ============================================================
sigma_T  = const.sigma_T.cgs.value          # [cm²]
MPC_CM   = 3.0856775814913673e24            # [cm/Mpc]
c_cms    = const.c.cgs.value                # [cm/s]
T_CMB_K  = 2.7255                           # [K]

# ── Electron number density today ────────────────────────────
def ne0_cgs(Y_He=0.24, include_He=True):
    m_p    = const.m_p.cgs.value
    rho_c0 = cosmo.critical_density0.cgs.value
    X_H    = 1.0 - Y_He
    n_H0   = X_H * (cosmo.Ob0 * rho_c0) / m_p
    y      = Y_He / (4.0 * X_H) if include_He else 0.0
    return n_H0 * (1.0 + y)

ne0 = ne0_cgs()

# ============================================================
# Growth factor / rate / velocity conversion
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
    """D(z)·f(z)·H(z)/(1+z)  [km/s]  →  comoving peculiar velocity"""
    return growth_factor(z) * growth_rate(z) * cosmo.H(z).value / (1.0 + z)

# ============================================================
# Scan configuration
# ============================================================
# Fixed resolution: Δx = 800/128 Mpc ≈ 6.25 Mpc
DELTA_X  = 800.0 / 128.0                   # [Mpc]
BOX_LENS = [200, 400, 600, 800, 1000]       # [Mpc]

# HII_DIM for each box (round to nearest power of 2 or even int)
# We enforce exact Δx by using round(BOX_LEN / DELTA_X)
DIMS = {L: max(4, int(round(L / DELTA_X))) for L in BOX_LENS}
print("Box-size scan configuration:")
print(f"  Fixed Δx = {DELTA_X:.3f} Mpc")
for L in BOX_LENS:
    N = DIMS[L]
    print(f"  BOX_LEN={L:5d} Mpc  →  HII_DIM={N:4d}  "
          f"(actual Δx = {L/N:.3f} Mpc)")

# Redshift grid — EoR (z ≥ 5), same as parent code
ZS_EOR = sorted([z for z in np.arange(5.0, 15.5, 0.5)], reverse=True)
print(f"\nRedshift slices (EoR): {ZS_EOR}")

# Reference box for ratio plots
REF_BOX = 800

# ============================================================
# Field & P_{q⊥} helpers
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
    vx    = coeval.lowres_vx * fac * 1e5     # [cm/s]
    vy    = coeval.lowres_vy * fac * 1e5
    vz    = coeval.lowres_vz * fac * 1e5
    return delta, xH, vx, vy, vz


def qperp_power(delta, xH, vx, vy, vz, BOX_LEN, nbins=None):
    """
    Spherically averaged P_{q⊥}(k) with per-bin std.
    Identical estimator to parent code (Park+2013 convention).
    """
    chi = 1.0 - xH
    w   = (1.0 + delta) * chi

    qx = w * vx;  qy = w * vy;  qz = w * vz    # [cm/s]

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


# ============================================================
# C_ell / D_ell integrator (Cain+2024 Eq. 3 / Park+2013)
# ============================================================
def compute_Dell(results_qperp_box, ZS_asc):
    """
    Given a dict keyed by z → {'k', 'Pqperp', 'Pstd', 'xH_mean'},
    compute D_ell [μK²] via the Limber integral.
    Returns ells, D_ell, sigma_D_ell arrays.
    """
    pref = (sigma_T * ne0 / c_cms)**2

    chi_mpc  = np.array([cosmo.comoving_distance(z).value for z in ZS_asc])
    dchi_mpc = np.abs(np.gradient(chi_mpc))
    dchi_cm  = dchi_mpc * MPC_CM

    xe_arr = np.array([1.0 - results_qperp_box[z]['xH_mean']
                       for z in ZS_asc])

    # optical depth τ(z)
    tau = np.zeros(len(ZS_asc))
    for i in range(len(ZS_asc) - 1):
        zmid   = 0.5 * (ZS_asc[i]  + ZS_asc[i+1])
        xe_mid = 0.5 * (xe_arr[i]  + xe_arr[i+1])
        tau[i+1] = tau[i] + sigma_T * ne0 * xe_mid * (1.0+zmid)**2 * dchi_cm[i]

    # valid ell range
    k_max_sim = min(results_qperp_box[z]['k'].max() for z in ZS_asc)
    k_min_sim = max(results_qperp_box[z]['k'].min() for z in ZS_asc)
    s_min     = chi_mpc.min()
    s_max     = chi_mpc.max()

    ells_full = np.unique(
        np.round(np.logspace(2.0, 4.5, 80)).astype(int)).astype(int)
    ell_max_valid = int(k_max_sim * s_min)
    ell_min_valid = int(k_min_sim * s_max)
    ells = ells_full[(ells_full >= ell_min_valid) &
                     (ells_full <= ell_max_valid)]

    if len(ells) == 0:
        print("  WARNING: no valid ell range — box may be too small.")
        return np.array([]), np.array([]), np.array([])

    k_list = [results_qperp_box[z]['k']      for z in ZS_asc]
    P_list = [results_qperp_box[z]['Pqperp'] for z in ZS_asc]
    S_list = [results_qperp_box[z]['Pstd']   for z in ZS_asc]

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

    sigma_C_ell = np.sqrt(var_C_ell)
    prefD = ells * (ells + 1.0) / (2.0 * np.pi) * T_CMB_K**2 * 1e12
    return ells, prefD * C_ell, prefD * sigma_C_ell


# ============================================================
# Main loop: compute or load P_{q⊥} for each (BOX_LEN, z)
# ============================================================
all_results  = {}   # all_results[BOX_LEN][z] = {k, Pqperp, Pstd, xH_mean}
dell_results = {}   # dell_results[BOX_LEN] = (ells, D_ell, sigma_D)

for BOX_LEN in BOX_LENS:
    N    = DIMS[BOX_LEN]
    tag  = f"box{BOX_LEN}_N{N}"
    cache_file = os.path.join(CACHE_DIR, f"qperp_{tag}.pkl")

    print(f"\n{'='*60}")
    print(f"BOX_LEN = {BOX_LEN} Mpc  |  HII_DIM = {N}  |  Δx = {BOX_LEN/N:.3f} Mpc")
    print(f"{'='*60}")

    # ── Load or compute ──────────────────────────────────────
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            results_box = pickle.load(f)
        print(f"  Cache loaded: {len(results_box)} redshifts")
    else:
        results_box = {}

    missing = [z for z in ZS_EOR if z not in results_box]
    if missing:
        print(f"  Computing {len(missing)} redshifts...")
        for z in missing:
            print(f"    z={z:.1f}...", end=' ', flush=True)
            try:
                delta, xH, vx, vy, vz = run_coeval_fields(z, N, BOX_LEN)
                k_q, P_q, P_std = qperp_power(delta, xH, vx, vy, vz, BOX_LEN)
                results_box[z] = {
                    'k'      : k_q,
                    'Pqperp' : P_q,
                    'Pstd'   : P_std,
                    'xH_mean': float(xH.mean()),
                }
                print(f"<xH>={xH.mean():.3f}")
            except Exception as e:
                print(f"FAILED: {e}")

        with open(cache_file, 'wb') as f:
            pickle.dump(results_box, f)
        print(f"  Cache saved → {cache_file}")
    else:
        print("  All redshifts already cached.")

    all_results[BOX_LEN] = results_box

    # ── D_ell ────────────────────────────────────────────────
    ZS_asc = sorted([z for z in results_box.keys() if z >= 5.0])
    if len(ZS_asc) >= 2:
        ells, D_ell, sigma_D = compute_Dell(results_box, ZS_asc)
        dell_results[BOX_LEN] = (ells, D_ell, sigma_D)
        if len(ells) > 0:
            D3000 = float(np.interp(3000.0, ells, D_ell))
            print(f"  D_3000 = {D3000:.4g} μK²  "
                  f"(ell range: {ells.min()}–{ells.max()})")
    else:
        print("  Not enough redshifts for D_ell integration.")


# ============================================================
# Summary table
# ============================================================
print("\n" + "="*65)
print(f"{'BOX_LEN':>10}  {'HII_DIM':>8}  {'Δx [Mpc]':>10}  "
      f"{'D_3000 [μK²]':>14}  {'ell_min':>8}  {'ell_max':>8}")
print("-"*65)
for L in BOX_LENS:
    N = DIMS[L]
    if L in dell_results and len(dell_results[L][0]) > 0:
        ells, D_ell, _ = dell_results[L]
        D3000 = float(np.interp(3000.0, ells, D_ell))
        print(f"{L:>10}  {N:>8}  {L/N:>10.3f}  "
              f"{D3000:>14.4g}  {ells.min():>8}  {ells.max():>8}")
    else:
        print(f"{L:>10}  {N:>8}  {L/N:>10.3f}  {'N/A':>14}")


# ============================================================
# Plot 1: D_ell for all box sizes on one panel
# ============================================================
colors_box = plt.cm.viridis(np.linspace(0.1, 0.9, len(BOX_LENS)))
ls_styles  = ['-', '--', '-.', ':', (0,(3,1,1,1))]

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xscale('log');  ax.set_yscale('log')
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$D_\ell\ [\mu\mathrm{K}^2]$')
ax.set_title(r'Patchy kSZ: Box-size Convergence'
             f'\n$\Delta x = {DELTA_X:.2f}$ Mpc (fixed resolution)')

for (L, col, ls) in zip(BOX_LENS, colors_box, ls_styles):
    if L not in dell_results or len(dell_results[L][0]) == 0:
        continue
    ells, D_ell, sigma_D = dell_results[L]
    N = DIMS[L]
    ax.plot(ells, D_ell, color=col, ls=ls, lw=2.0,
            label=rf'$L={L}$ Mpc, $N={N}^3$')
    ax.fill_between(ells,
                    np.maximum(D_ell - sigma_D, 1e-8),
                    D_ell + sigma_D,
                    color=col, alpha=0.12)

ax.errorbar(3000, 1.1, yerr=[[0.7], [1.0]],
            fmt='s', ms=8, capsize=5, capthick=2,
            color='red', zorder=10, label='Reichardt+2021')

ax.legend(frameon=False, fontsize=12, ncol=2)
plt.tight_layout()
stem = os.path.join(PLOT_DIR, "Dell_boxsize_convergence")
plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
print(f"\nSaved → {stem}.pdf / .png")
plt.close()


# ============================================================
# Plot 2: D_ell ratio relative to 800 Mpc reference box
# ============================================================
if REF_BOX in dell_results and len(dell_results[REF_BOX][0]) > 0:
    ells_ref, D_ref, _ = dell_results[REF_BOX]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(rf'$D_\ell / D_\ell^{{L={REF_BOX}\,\mathrm{{Mpc}}}}$')
    ax.set_title(r'Box-size Convergence: $D_\ell$ Ratio'
                 f'\nReference box: $L = {REF_BOX}$ Mpc')
    ax.axhline(1.0, color='k', lw=1.0, ls=':')
    ax.axhspan(0.95, 1.05, color='gray', alpha=0.10, label='±5%')
    ax.axhspan(0.90, 1.10, color='gray', alpha=0.05, label='±10%')

    for (L, col, ls) in zip(BOX_LENS, colors_box, ls_styles):
        if L == REF_BOX or L not in dell_results:
            continue
        ells_L, D_L, _ = dell_results[L]
        if len(ells_L) == 0:
            continue
        # Interpolate both onto the overlapping ell grid
        ell_common = ells_L[(ells_L >= ells_ref.min()) &
                             (ells_L <= ells_ref.max())]
        if len(ell_common) == 0:
            continue
        D_L_interp   = np.interp(ell_common, ells_L,   D_L)
        D_ref_interp = np.interp(ell_common, ells_ref, D_ref)
        ratio = D_L_interp / np.where(D_ref_interp > 0, D_ref_interp, np.nan)
        ax.plot(ell_common, ratio, color=col, ls=ls, lw=2.0,
                label=rf'$L={L}$ Mpc')

    ax.legend(frameon=False, fontsize=12)
    ax.set_ylim(0.5, 1.7)
    plt.tight_layout()
    stem = os.path.join(PLOT_DIR, "Dell_ratio_boxsize")
    plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
    print(f"Saved → {stem}.pdf / .png")
    plt.close()


# ============================================================
# Plot 3: P_{q⊥}(k) at a representative redshift (z=7)
# ============================================================
Z_SHOW = 7.0
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xscale('log');  ax.set_yscale('log')
ax.set_xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
ax.set_ylabel(r'$P_{q_\perp}(k)\ [\mathrm{cm}^2\,\mathrm{s}^{-2}\,\mathrm{Mpc}^3]$')
ax.set_title(rf'$P_{{q_\perp}}(k)$ at $z={Z_SHOW}$: Box-size Convergence'
             f'\n$\Delta x = {DELTA_X:.2f}$ Mpc (fixed)')

for (L, col, ls) in zip(BOX_LENS, colors_box, ls_styles):
    if L not in all_results or Z_SHOW not in all_results[L]:
        continue
    res = all_results[L][Z_SHOW]
    k, P, S = res['k'], res['Pqperp'], res['Pstd']
    N = DIMS[L]
    ax.plot(k, P, color=col, ls=ls, lw=2.0,
            label=rf'$L={L}$ Mpc, $N={N}^3$')
    ax.fill_between(k, P - S, P + S, color=col, alpha=0.12)

ax.legend(frameon=False, fontsize=12, ncol=2)
plt.tight_layout()
stem = os.path.join(PLOT_DIR, f"Pqperp_z{Z_SHOW}_boxsize")
plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
print(f"Saved → {stem}.pdf / .png")
plt.close()


# ============================================================
# Plot 4: D_3000 vs BOX_LEN  (convergence diagnostic)
# ============================================================
L_vals, D3000_vals = [], []
for L in BOX_LENS:
    if L not in dell_results or len(dell_results[L][0]) == 0:
        continue
    ells, D_ell, _ = dell_results[L]
    L_vals.append(L)
    D3000_vals.append(float(np.interp(3000.0, ells, D_ell)))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(L_vals, D3000_vals, 'o-', color='steelblue', lw=2.0, ms=8)
if REF_BOX in dell_results and len(dell_results[REF_BOX][0]) > 0:
    ells_r, D_r, _ = dell_results[REF_BOX]
    D3000_ref = float(np.interp(3000.0, ells_r, D_r))
    ax.axhline(D3000_ref, color='gray', ls='--', lw=1.2,
               label=rf'$L={REF_BOX}$ Mpc reference')
    ax.axhspan(D3000_ref * 0.95, D3000_ref * 1.05,
               color='gray', alpha=0.12, label='±5%')

ax.set_xlabel(r'Box size $L\ [\mathrm{Mpc}]$')
ax.set_ylabel(r'$D_{3000}\ [\mu\mathrm{K}^2]$')
ax.set_title(r'Box-size Convergence: $D_{3000}$'
             f'\n$\Delta x = {DELTA_X:.2f}$ Mpc (fixed resolution)')
ax.legend(frameon=False, fontsize=12)
plt.tight_layout()
stem = os.path.join(PLOT_DIR, "D3000_vs_boxsize")
plt.savefig(stem + ".pdf");  plt.savefig(stem + ".png", dpi=300)
print(f"Saved → {stem}.pdf / .png")
plt.close()

print("\nBox-size convergence test complete.")
print(f"All outputs in: {PLOT_DIR}")
