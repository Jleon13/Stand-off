# ========================= INITIAL SETUP =========================
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import wofz
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# ========================= CONSTANTS =========================
TREF = 296.0
K_B_erg = 1.380649e-16
C_M_S   = 299792458.0
C_CM_S  = C_M_S * 100.0
C2      = 1.43880285
N_A     = 6.02214086e23
N_L     = 2.47937196e19

# ========================= DATACLASS =========================
@dataclass
class Species:
    name: str
    mol: int
    iso: int
    qfile: str
    Wg: float
    Pmol: float
    Qref: float = field(init=False, default=np.nan)
    QT: float = field(init=False, default=np.nan)
    idx_all: np.ndarray | None = field(init=False, default=None)

# ========================= FUNCTIONS =========================
def load_Q_vals(qfile: str, Tref: float, T: float) -> Tuple[float, float]:
    """Read q*.txt and return Qref=Q(Tref), QT=Q(T)."""
    if not os.path.isfile(qfile):
        raise FileNotFoundError(f"Missing Q(T) file: {qfile}")
    try:
        df = pd.read_csv(qfile, header=None, delim_whitespace=True, comment='#')
        if df.shape[1] < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(qfile, header=None, sep=r"[\s,;]+", engine="python", comment='#')
    Tcol = df.iloc[:, 0].astype(float).to_numpy()
    Qcol = df.iloc[:, 1].astype(float).to_numpy()
    Qref = np.interp(Tref, Tcol, Qcol)
    QT   = np.interp(T,    Tcol, Qcol)
    return float(Qref), float(QT)

def read_hitran_par_minimal(path: str) -> Dict[str, np.ndarray]:
    """Read a classic HITRAN .par file (minimal fields)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find {path}")

    widths = [2,1,12,10,10,5,5,10,4,8, 15,15,15,15,6,12,1,7,7]
    use_up_to = 10
    cum = np.cumsum([0] + widths)
    sl = [(cum[i], cum[i+1]) for i in range(use_up_to)]

    mol, iso, nu0, Sref, A, g_air, g_self, Elow, n_air, shift = ([] for _ in range(10))

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s = [line[a:b] for a, b in sl]
                mol.append(int(s[0])); iso.append(int(s[1]))
                nu0.append(float(s[2])); Sref.append(float(s[3]))
                A.append(float(s[4])); g_air.append(float(s[5]))
                g_self.append(float(s[6])); Elow.append(float(s[7]))
                n_air.append(float(s[8])); shift.append(float(s[9]))
            except Exception:
                continue

    return dict(
        mol=np.asarray(mol, dtype=np.int32), iso=np.asarray(iso, dtype=np.int32),
        nu0=np.asarray(nu0, dtype=np.float64), Sref=np.asarray(Sref, dtype=np.float64),
        A=np.asarray(A, dtype=np.float64), g_air=np.asarray(g_air, dtype=np.float64),
        g_self=np.asarray(g_self, dtype=np.float64), Elow=np.asarray(Elow, dtype=np.float64),
        n_air=np.asarray(n_air, dtype=np.float64), shift=np.asarray(shift, dtype=np.float64)
    )

def voigt_profile(nu: np.ndarray, nu0_shifted: np.ndarray, alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Return normalized Voigt profile f_V(nu)."""
    s2 = np.sqrt(np.log(2.0))
    x = s2 * (nu[:, None] - nu0_shifted[None, :]) / alpha[None, :]
    y = s2 * (gamma[None, :] / alpha[None, :])
    z = x + 1j * y
    w = wofz(z)
    fV = s2 / np.sqrt(np.pi) / alpha[None, :] * np.real(w)
    return fV

def transmittance_for_gas_tile(nu_vec: np.ndarray, H: Dict[str, np.ndarray], sp: Species,
                               Tgas: float, Pair: float, Lm: float, mask_lines: np.ndarray) -> np.ndarray:
    """Compute T(nu) for a single gas over a spectral tile."""
    if not np.any(mask_lines):
        return np.ones_like(nu_vec)

    nu0, Sref, g_air, g_self, Elow, n_air, shift = (
        H['nu0'][mask_lines], H['Sref'][mask_lines], H['g_air'][mask_lines],
        H['g_self'][mask_lines], H['Elow'][mask_lines], H['n_air'][mask_lines],
        H['shift'][mask_lines]
    )

    pres = sp.Pmol + Pair
    nu0s = nu0 + shift * pres

    # Temperature scaling of line intensity
    S_T = (Sref * (sp.Qref / sp.QT) *
           np.exp(-C2 * Elow / Tgas) / np.exp(-C2 * Elow / TREF) *
           (1.0 - np.exp(-C2 * nu0 / Tgas)) / (1.0 - np.exp(-C2 * nu0 / TREF)))

    # Path intensity
    Lcm = Lm * 100.0
    line_intensity = S_T * (TREF / Tgas) * N_L * sp.Pmol * Lcm

    # Widths
    alpha = nu0 / C_CM_S * np.sqrt(2.0 * N_A * K_B_erg * Tgas * np.log(2.0) / sp.Wg)
    gamma = ((TREF / Tgas) ** n_air) * (g_air * (pres - sp.Pmol) + g_self * sp.Pmol)

    # Voigt and transmittance
    fV = voigt_profile(nu_vec, nu0s, alpha, gamma)
    tau = fV @ line_intensity
    return np.exp(-tau)

def bin_average(x_sorted: np.ndarray, y_sorted: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average y(x) over bins defined by edges (x must be sorted)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    yb = np.empty(edges.size - 1)
    left = np.searchsorted(x_sorted, edges[:-1], side='left')
    right = np.searchsorted(x_sorted, edges[1:], side='left')
    for i, (l, r) in enumerate(zip(left, right)):
        if r > l:
            yb[i] = np.mean(y_sorted[l:r])
        else:
            yb[i] = np.interp(centers[i], x_sorted, y_sorted)
    return centers, yb

def default_species() -> List[Species]:
    base = "data/isopo/"
    return [
        Species(name='H2O', mol=1, iso=1, qfile=base+'q1.txt',  Wg=18.0153, Pmol=1.876e+04/1e6),
        Species(name='CO2', mol=2, iso=1, qfile=base+'q7.txt',  Wg=44.0095, Pmol=330/1e6),
        Species(name='O3',  mol=3, iso=1, qfile=base+'q16.txt', Wg=47.9982, Pmol=0.03017/1e6),
        Species(name='N2O', mol=4, iso=1, qfile=base+'q21.txt', Wg=44.0128, Pmol=0.32/1e6),
        Species(name='CO',  mol=5, iso=1, qfile=base+'q26.txt', Wg=28.0101, Pmol=0.15/1e6),
        Species(name='CH4', mol=6, iso=1, qfile=base+'q32.txt', Wg=16.0313, Pmol=1.7/1e6),
        Species(name='O2',  mol=7, iso=1, qfile=base+'q36.txt', Wg=31.9988, Pmol=0.20946),
    ]

# ========================= MAIN FUNCTION =========================
def run_simulation(
    parfile: str,
    species: List[Species],
    nu_min: float = 666.67,
    nu_max: float = 10000.0,
    dnu: float = 0.01,
    tileW: float = 20.0,
    guard: float = 5.0,
    temp_K: float = 296.0,
    L_m: float = 1.0,
    Pair: float = 1.0,
    delta_um: float = 0.020,
    save_csv: bool = False,
    outdir: str = 'out',
    make_plots: bool = True,
) -> Dict[str, Any]:
    """
    Compute transmittances/attenuations and return sampled results.
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError(f"Missing HITRAN .par: {parfile}")
    for sp in species:
        print(f"DEBUG: {sp.name} qfile={sp.qfile} exists={os.path.isfile(sp.qfile)}")
        if not os.path.isfile(sp.qfile):
            raise FileNotFoundError(f"Missing q-file for {sp.name}: {sp.qfile}")

    # Load Q(T)
    for sp in species:
        Qref, QT = load_Q_vals(sp.qfile, TREF, temp_K)
        sp.Qref, sp.QT = Qref, QT

    # Read HITRAN .par
    H = read_hitran_par_minimal(parfile)

    # Line indices per species
    for sp in species:
        sp.idx_all = (H['mol'] == sp.mol) & (H['iso'] == sp.iso)

    # Spectral tiling
    edges_tiles = np.arange(nu_min, nu_max + 1e-9, tileW)
    nu_all_parts, T_prod_all_parts, T_sum_all_parts = [], [], []
    T_each_acc = [[] for _ in species]

    for a in edges_tiles:
        b = min(a + tileW, nu_max)
        a_ext, b_ext = max(nu_min, a - guard), min(nu_max, b + guard)
        nu_ext = np.arange(a_ext, b_ext + 1e-12, dnu)

        T_ext_each = np.ones((len(species), nu_ext.size), dtype=np.float64)
        for k, sp in enumerate(species):
            idx_tile = sp.idx_all & (H['nu0'] >= a_ext) & (H['nu0'] <= b_ext)
            if np.any(idx_tile):
                T_ext_each[k, :] = transmittance_for_gas_tile(nu_ext, H, sp, temp_K, Pair, L_m, idx_tile)

        T_ext_prod, T_ext_sum = np.prod(T_ext_each, axis=0), np.sum(T_ext_each, axis=0)

        keep = (nu_ext >= a) & (nu_ext <= b)
        nu_all_parts.append(nu_ext[keep])
        T_prod_all_parts.append(T_ext_prod[keep])
        T_sum_all_parts.append(T_ext_sum[keep])
        for k in range(len(species)):
            T_each_acc[k].append(T_ext_each[k, keep])

    nu_all = np.concatenate(nu_all_parts)
    T_prod, T_sum = np.concatenate(T_prod_all_parts), np.concatenate(T_sum_all_parts)
    T_each = [np.concatenate(T_each_acc[k]) for k in range(len(species))]

    # Convert to wavelength and sort
    lambda_um = 1e4 / nu_all
    ord_idx = np.argsort(lambda_um)
    lambda_sorted = lambda_um[ord_idx]
    T_prod_lambda, T_sum_lambda = T_prod[ord_idx], T_sum[ord_idx]
    T_each_lambda = [T_each[k][ord_idx] for k in range(len(species))]

    # Bin in wavelength
    lam_min = math.ceil(lambda_sorted.min() / delta_um) * delta_um
    lam_max = math.floor(lambda_sorted.max() / delta_um) * delta_um
    edges = np.arange(lam_min, lam_max + 1e-9, delta_um)
    lambda_centers, T_prod_samp = bin_average(lambda_sorted, T_prod_lambda, edges)
    _, T_sum_samp = bin_average(lambda_sorted, T_sum_lambda, edges)
    T_each_samp = []
    for arr in T_each_lambda:
        _, yy = bin_average(lambda_sorted, arr, edges)
        T_each_samp.append(yy)

    # Attenuation in dB/m
    invL = 1.0 / max(L_m, 1e-300)
    A_dbm_lam_each = [-(10.0 * invL) * np.log10(np.clip(arr, 1e-300, 1.0)) for arr in T_each_samp]
    A_dbm_lam_sum = np.sum(np.stack(A_dbm_lam_each, axis=0), axis=0)

    # ===================== CONSISTENT PLOTTING + EXPORT =====================
    if make_plots:
        # Consistent sizing and high-res export
        os.makedirs(outdir, exist_ok=True)
        FIGSIZE = (14, 7)            # same size for all
        DPI_EXPORT = 600             # high resolution
        EXPORT_FORMATS = ("png", "pdf")  # raster + vector

        plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.edgecolor": "#222",
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        })

        def _positivize(a):
            a = np.asarray(a)
            eps = max(1e-12, np.nanmin(a[a > 0]) * 0.1) if np.any(a > 0) else 1e-12
            return np.clip(a, eps, None)

        def save_fig(fig, basename: str):
            for ext in EXPORT_FORMATS:
                dpi = DPI_EXPORT if ext.lower() in ("png", "jpg", "jpeg", "tif", "tiff") else None
                fig.savefig(
                    os.path.join(outdir, f"{basename}.{ext}"),
                    dpi=dpi, bbox_inches="tight", pad_inches=0.05
                )

        # Shared Y-limits for comparability
        ymin, ymax = 1e-5, 1e3
        y_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        y_tick_labels = ['10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '10⁰', '10¹', '10²', '10³']

        # ---------- Figure 1: total attenuation ----------
        A_sum_plot = _positivize(A_dbm_lam_sum)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.semilogy(lambda_centers, np.where(A_sum_plot >= ymin, A_sum_plot, np.nan),
                    lw=2.5, color="#1f77b4", label="Total")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Attenuation (dB/m)")
        ax.set_title("Combined atmospheric attenuation")
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.grid(True, which='major', axis='both', color='#bbb', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, which='minor', axis='both', color='#eee', linestyle=':', linewidth=0.5, alpha=0.3)
        max_idx = np.nanargmax(A_sum_plot)
        ax.annotate(f"Max: {A_sum_plot[max_idx]:.2e} dB/m",
                    xy=(lambda_centers[max_idx], A_sum_plot[max_idx]),
                    xytext=(lambda_centers[max_idx]+1, A_sum_plot[max_idx]*1.5),
                    arrowprops=dict(arrowstyle="->", color="black"), fontsize=15, color="black")
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        save_fig(fig, "attenuation_total")
        plt.show()
        plt.close(fig)

        # ---------- Figure 2: attenuation by gas ----------
        A_each_plot = [_positivize(arr) for arr in A_dbm_lam_each]
        fig, ax = plt.subplots(figsize=FIGSIZE)
        colors = plt.cm.Set2(np.linspace(0, 1, len(species)))
        for arr, spc, color in zip(A_each_plot, species, colors):
            arr_trunc = np.where(arr >= ymin, arr, np.nan)
            ax.semilogy(lambda_centers, arr_trunc, lw=2, label=spc.name, color=color)
            max_idx = np.nanargmax(arr)
            ax.annotate(f"{spc.name}: {arr[max_idx]:.2e}",
                        xy=(lambda_centers[max_idx], arr[max_idx]),
                        xytext=(lambda_centers[max_idx]+0.5, arr[max_idx]*1.5),
                        textcoords="data",
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
                        fontsize=13, color=color)
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Attenuation (dB/m)")
        ax.set_title("Spectral attenuation by gas")
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.grid(True, which='major', axis='both', color='#bbb', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, which='minor', axis='both', color='#eee', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, ncol=2)
        plt.tight_layout()
        save_fig(fig, "attenuation_by_gas")
        plt.show()
        plt.close(fig)

    # ===================== CSV EXPORT (optional) =====================
    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        # Total
        df_total = pd.DataFrame({
            "lambda_um": lambda_centers,
            "T_total": T_prod_samp,
            "A_total_dbm": A_dbm_lam_sum
        })
        df_total.to_csv(os.path.join(outdir, "attenuation_total.csv"), index=False)

        # By gas (attenuation and transmittance)
        cols = {"lambda_um": lambda_centers}
        for sp, T_arr, A_arr in zip(species, T_each_samp, A_dbm_lam_each):
            cols[f"T_{sp.name}"] = T_arr
            cols[f"A_{sp.name}_dbm"] = A_arr
        pd.DataFrame(cols).to_csv(os.path.join(outdir, "attenuation_by_gas.csv"), index=False)

    return dict(
        lambda_centers=lambda_centers, T_prod_samp=T_prod_samp, T_each_samp=T_each_samp,
        A_dbm_lam_sum=A_dbm_lam_sum, A_dbm_lam_each=A_dbm_lam_each, species=species
    )
