# -*- coding: utf-8 -*-
# forward_lsf.py — Forward con LSF en λ y downsampling robusto
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from scipy.special import wofz

# ========================= CONSTANTES =========================
TREF   = 296.0                         # K
K_B_erg= 1.380649e-16                  # erg/K
C_M_S  = 299792458.0                   # m/s
C_CM_S = C_M_S * 100.0                 # cm/s
C2     = 1.43880285                    # cm*K (constante espectroscópica)
N_A    = 6.02214086e23                 # 1/mol
# N_L: #/cm^3 a 296K y 1 atm (≈ Loschmidt en cgs)
N_L    = 2.47937196e19

# ========================= DATACLASS =========================
@dataclass
class Species:
    name: str
    mol: int
    iso: int
    qfile: str
    Wg: float              # g/mol
    Pmol: float            # fracción molar (χ)
    Qref: float = field(init=False, default=np.nan)
    QT:   float = field(init=False, default=np.nan)
    idx_all: np.ndarray|None = field(init=False, default=None)

# ========================= UTILIDADES I/O =========================
def load_Q_vals(qfile: str, Tref: float, T: float) -> Tuple[float, float]:
    if not os.path.isfile(qfile):
        raise FileNotFoundError(f"Missing Q(T) file: {qfile}")
    try:
        df = pd.read_csv(qfile, header=None, sep=r"\s+", engine="python", comment="#")
        if df.shape[1] < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(qfile, header=None, sep=r"[\s,;]+", engine="python", comment="#")
    Tcol = df.iloc[:,0].astype(float).to_numpy()
    Qcol = df.iloc[:,1].astype(float).to_numpy()
    Qref = float(np.interp(Tref, Tcol, Qcol))
    QT   = float(np.interp(T,    Tcol, Qcol))
    return Qref, QT

def read_hitran_par_minimal(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find {path}")
    widths = [2,1,12,10,10,5,5,10,4,8, 15,15,15,15,6,12,1,7,7]
    use_up_to = 10
    cum = np.cumsum([0]+widths)
    sl  = [(cum[i], cum[i+1]) for i in range(use_up_to)]
    mol,iso,nu0,Sref,A,g_air,g_self,Elow,n_air,shift = ([] for _ in range(10))
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            try:
                s = [line[a:b] for a,b in sl]
                mol.append(int(s[0])); iso.append(int(s[1]))
                nu0.append(float(s[2])); Sref.append(float(s[3]))
                A.append(float(s[4])); g_air.append(float(s[5]))
                g_self.append(float(s[6])); Elow.append(float(s[7]))
                n_air.append(float(s[8])); shift.append(float(s[9]))
            except Exception:
                continue
    return dict(
        mol=np.asarray(mol, np.int32), iso=np.asarray(iso, np.int32),
        nu0=np.asarray(nu0, np.float64), Sref=np.asarray(Sref, np.float64),
        A=np.asarray(A, np.float64), g_air=np.asarray(g_air, np.float64),
        g_self=np.asarray(g_self, np.float64), Elow=np.asarray(Elow, np.float64),
        n_air=np.asarray(n_air, np.float64), shift=np.asarray(shift, np.float64)
    )

# ========================= FÍSICA DE LÍNEA =========================
def voigt_profile(nu: np.ndarray, nu0_shifted: np.ndarray,
                  alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Perfil de Voigt normalizado fV(ν)."""
    s2 = np.sqrt(np.log(2.0))
    x  = s2*(nu[:,None] - nu0_shifted[None,:]) / alpha[None,:]
    y  = s2*(gamma[None,:] / alpha[None,:])
    z  = x + 1j*y
    w  = wofz(z)
    return (s2/np.sqrt(np.pi))*(np.real(w)/alpha[None,:])

def transmittance_for_gas_tile(nu_vec: np.ndarray, H: Dict[str,np.ndarray], sp: Species,
                               Tgas: float, P_atm_total: float, Lm: float,
                               mask_lines: np.ndarray) -> np.ndarray:
    """T(ν) para un gas en un tile (ν en cm^-1)."""
    if not np.any(mask_lines):
        return np.ones_like(nu_vec)

    nu0   = H['nu0'][mask_lines]
    Sref  = H['Sref'][mask_lines]
    g_air = H['g_air'][mask_lines]
    g_self= H['g_self'][mask_lines]
    Elow  = H['Elow'][mask_lines]
    n_air = H['n_air'][mask_lines]
    delta_air = H['shift'][mask_lines]   # δ_air (HITRAN) [cm^-1/atm]

    # Presiones parciales (atm)
    Pself = sp.Pmol * P_atm_total
    Pair  = max(P_atm_total - Pself, 0.0)

    # Corrimiento por presión: δ_air·P_air (+ δ_self·P_self si lo tuvieras)
    nu0s = nu0 + delta_air*Pair

    # Escalado S(T)
    S_T = (Sref * (sp.Qref/sp.QT) *
           np.exp(-C2*Elow/Tgas)/np.exp(-C2*Elow/TREF) *
           (1.0 - np.exp(-C2*nu0/Tgas)) / (1.0 - np.exp(-C2*nu0/TREF)))

    # Columna (usa P_total)
    Lcm = Lm*100.0
    col = (TREF/Tgas) * N_L * P_atm_total * sp.Pmol * Lcm  # #/cm^2 efectivos

    # Anchos
    alpha = nu0/C_CM_S * np.sqrt(2.0*N_A*K_B_erg*Tgas*np.log(2.0)/sp.Wg)  # Doppler
    gamma = ((TREF/Tgas)**n_air) * (g_air*Pair + g_self*Pself)            # colisión

    # Voigt y τ
    fV  = voigt_profile(nu_vec, nu0s, alpha, gamma)
    tau = fV @ (S_T*col)
    return np.exp(-tau)

# ========================= LSF en λ =========================
def _gauss_kernel_lambda(FWHM_um: float, dlam: float, truncate_sigma: float=4.0) -> np.ndarray:
    sigma = FWHM_um / (2.0*np.sqrt(2.0*np.log(2.0)))
    half  = max(2, int(truncate_sigma*sigma/dlam))
    x = np.arange(-half, half+1)*dlam
    k = np.exp(-0.5*(x/sigma)**2)
    k /= k.sum()
    return k

def _triangle_kernel_lambda(W_um: float, dlam: float) -> np.ndarray:
    """W_um = ancho total (base)."""
    half = max(1, int(0.5*W_um/dlam))
    x = np.arange(-half, half+1, dtype=float)
    k = (1.0 - np.abs(x)/half)
    k[k<0] = 0.0
    k /= k.sum()
    return k

def _square_kernel_lambda(W_um: float, dlam: float) -> np.ndarray:
    half = max(1, int(0.5*W_um/dlam))
    k = np.ones(2*half+1, dtype=float)
    k /= k.sum()
    return k

def _sinc_kernel_lambda(W_um: float, dlam: float, truncate_mult: float=2.0) -> np.ndarray:
    """Sinc truncado como en SpectralCalc (~±2W)."""
    half = max(2, int(truncate_mult*W_um/dlam))
    x = np.arange(-half, half+1)*dlam
    # sinc(x) = sin(2π x / W)/ (π x / W)  -> normalizamos para suma=1
    eps = 1e-12
    arg = np.pi*x/(W_um/2.0)  # escala para que cero-> W/2 a media oscilación
    k = np.sinc(arg/np.pi)  # numpy sinc = sin(pi z)/(pi z)
    k /= k.sum() + eps
    return k

def build_lsf_lambda(kind: str, W_um: float, dlam: float) -> np.ndarray:
    kind = (kind or "gaussian").lower()
    if kind == "gaussian":
        return _gauss_kernel_lambda(FWHM_um=W_um, dlam=dlam, truncate_sigma=4.0)
    if kind == "triangle":
        return _triangle_kernel_lambda(W_um=W_um, dlam=dlam)
    if kind == "square":
        return _square_kernel_lambda(W_um=W_um, dlam=dlam)
    if kind == "sinc":
        return _sinc_kernel_lambda(W_um=W_um, dlam=dlam, truncate_mult=2.0)
    raise ValueError(f"LSF kind desconocido: {kind}")

# ========================= RESAMPLE =========================
def describe_spacing(x: np.ndarray, name: str, tol: float=0.02) -> None:
    dx = np.diff(x); dx = dx[np.isfinite(dx)]
    if dx.size == 0: return
    r = np.max(dx)/max(np.min(dx), 1e-30)
    uniform = "≈ uniforme" if (r <= 1.0+tol) else "no uniforme"
    print(f"[Grid {name}] minΔ={dx.min():.3g}, medΔ={np.median(dx):.3g}, maxΔ={dx.max():.3g} -> {uniform}")

def bin_average_irregular(x_sorted: np.ndarray, y_sorted: np.ndarray,
                          edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = 0.5*(edges[:-1] + edges[1:])
    out = np.empty(edges.size-1, float)
    for i, (a,b) in enumerate(zip(edges[:-1], edges[1:])):
        l = np.searchsorted(x_sorted, a, side="left")
        r = np.searchsorted(x_sorted, b, side="right")
        xseg = np.r_[a, x_sorted[l:r], b]
        yseg = np.r_[np.interp(a, x_sorted, y_sorted),
                     y_sorted[l:r],
                     np.interp(b, x_sorted, y_sorted)]
        area = np.trapz(yseg, xseg)
        out[i] = area/max(b-a, 1e-30)
    return centers, out

# ========================= ESPECIES POR DEFECTO =========================
def default_species() -> List[Species]:
    base = "data/isopo/"
    return [
        Species('H2O', 1, 1, base+'q1.txt',   18.0153,  1.876e+04/1e6),
        Species('CO2', 2, 1, base+'q7.txt',   44.0095,  330/1e6),
        Species('O3',  3, 1, base+'q16.txt',  47.9982,  0.03017/1e6),
        Species('N2O', 4, 1, base+'q21.txt',  44.0128,  0.32/1e6),
        Species('CO',  5, 1, base+'q26.txt',  28.0101,  0.15/1e6),
        Species('CH4', 6, 1, base+'q32.txt',  16.0313,  1.7/1e6),
        Species('O2',  7, 1, base+'q36.txt',  31.9988,  0.20946),
    ]

# ========================= FORWARD PRINCIPAL =========================
def run_simulation(
    parfile: str,
    species: List[Species],
    nu_min: float = 625.0,
    nu_max: float = 10000.0,
    dnu: float = 0.01,
    tileW: float = 20.0,
    guard: float = 5.0,
    temp_K: float = 296.0,
    L_m: float = 1.0,
    P_atm_total: float = 1.0,
    delta_um: float = 0.020,               # resolución objetivo (bins de salida)
    lsf: Optional[Dict[str, Any]] = None,  # {"kind": "gaussian", "W_um": 0.02}  (en λ)
    save_csv: bool = False,
    outdir: str = "out",
    make_plots: bool = True,
) -> Dict[str, Any]:

    if not os.path.isfile(parfile):
        raise FileNotFoundError(f"Missing HITRAN .par: {parfile}")
    for sp in species:
        if not os.path.isfile(sp.qfile):
            raise FileNotFoundError(f"Missing q-file for {sp.name}: {sp.qfile}")

    # Q(T)
    for sp in species:
        sp.Qref, sp.QT = load_Q_vals(sp.qfile, TREF, temp_K)

    # HITRAN y máscaras por especie
    H = read_hitran_par_minimal(parfile)
    for sp in species:
        sp.idx_all = (H['mol']==sp.mol) & (H['iso']==sp.iso)

    # -------- LBL en ν (uniforme) con tiling ----------
    edges_tiles = np.arange(nu_min, nu_max+1e-9, tileW)
    nu_all_parts, T_prod_all_parts = [], []
    T_each_acc = [[] for _ in species]

    describe_spacing(np.arange(nu_min, nu_max+dnu, dnu), "ν (objetivo)", tol=0.02)

    for a in edges_tiles:
        b = min(a+tileW, nu_max)
        a_ext, b_ext = max(nu_min, a-guard), min(nu_max, b+guard)
        nu_ext = np.arange(a_ext, b_ext+1e-12, dnu)

        T_ext_each = np.ones((len(species), nu_ext.size), float)
        for k, sp in enumerate(species):
            idx_tile = sp.idx_all & (H['nu0']>=a_ext) & (H['nu0']<=b_ext)
            if np.any(idx_tile):
                T_ext_each[k,:] = transmittance_for_gas_tile(
                    nu_ext, H, sp, temp_K, P_atm_total, L_m, idx_tile
                )
        T_ext_prod = np.prod(T_ext_each, axis=0)

        keep = (nu_ext>=a) & (nu_ext<=b)
        nu_all_parts.append(nu_ext[keep])
        T_prod_all_parts.append(T_ext_prod[keep])
        for k in range(len(species)):
            T_each_acc[k].append(T_ext_each[k, keep])

    nu_all = np.concatenate(nu_all_parts)                 # cm^-1 (ascendente)
    T_prod = np.concatenate(T_prod_all_parts)
    T_each = [np.concatenate(T_each_acc[k]) for k in range(len(species))]

    # -------- ν -> λ (no uniforme) ----------
    lambda_um = 1e4/nu_all
    ord_idx    = np.argsort(lambda_um)
    lam_raw    = lambda_um[ord_idx]
    T_total_raw= T_prod[ord_idx]
    T_each_raw = [T_each[k][ord_idx] for k in range(len(species))]
    describe_spacing(lam_raw, "λ (raw)", tol=0.02)

    # -------- Re-muestreo uniforme fino en λ ----------
    oversample = max(8, int(40))  # ~40× como SpectralCalc
    dlam_fine  = delta_um/oversample
    lam_min = math.ceil(lam_raw.min()/dlam_fine)*dlam_fine
    lam_max = math.floor(lam_raw.max()/dlam_fine)*dlam_fine
    lam_fine = np.arange(lam_min, lam_max+dlam_fine/2, dlam_fine)

    T_total_fine = np.interp(lam_fine, lam_raw, T_total_raw)
    T_each_fine  = [np.interp(lam_fine, lam_raw, arr) for arr in T_each_raw]

    # -------- LSF en λ (si se pide) ----------
    if lsf is None:
        lsf = {"kind":"gaussian", "W_um": delta_um}
    kind = (lsf.get("kind","gaussian") or "gaussian").lower()

    # Si te pasaron W_cm1 por costumbre, avisa y aproxímalo a W_um ~ Δλ
    W_um = float(lsf.get("W_um", delta_um))

    k_lsf = build_lsf_lambda(kind=kind, W_um=W_um, dlam=dlam_fine)

    T_total_conv = np.convolve(T_total_fine, k_lsf, mode="same")

    # -------- Downsampling a Δλ=delta_um (promedio trapecial) ----------
    lam_min_ds = math.ceil(lam_fine.min()/delta_um)*delta_um
    lam_max_ds = math.floor(lam_fine.max()/delta_um)*delta_um
    edges = np.arange(lam_min_ds, lam_max_ds+1e-12, delta_um)

    lambda_centers, T_total_ds = bin_average_irregular(lam_fine, T_total_conv, edges)
    T_each_ds = []
    # Nota: para referencia por-gas (sin LSF)
    for arr in T_each_fine:
        _, yy = bin_average_irregular(lam_fine, arr, edges)
        T_each_ds.append(yy)

    # -------- Atenuación (dB/m) ----------
    invL = 1.0/max(L_m, 1e-300)
    A_total_dbm = -(10.0*invL)*np.log10(np.clip(T_total_ds, 1e-300, 1.0))
    A_each_dbm  = [-(10.0*invL)*np.log10(np.clip(arr,       1e-300, 1.0)) for arr in T_each_ds]

    # -------- Plots opcionales ----------
    if make_plots:
        os.makedirs(outdir, exist_ok=True)
        plt.rcParams.update({"font.size":15})
        # Total
        fig, ax = plt.subplots(figsize=(13,6))
        ymin,ymax = 1e-5, 1e3
        ax.semilogy(lambda_centers, np.clip(A_total_dbm, ymin, None),
                    lw=2.2, color="#1f77b4", label="Total (LSF en λ)")
        ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Attenuation (dB/m)")
        ax.set_title(f"Atmospheric attenuation | LSF={kind}, W={W_um:.3f} µm")
        ax.set_ylim(ymin, ymax); ax.grid(True, which="both", alpha=0.35)
        ax.legend()
        fig.savefig(os.path.join(outdir, "attenuation_total_lsf_lambda.png"),
                    dpi=300, bbox_inches="tight"); plt.close(fig)

    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame({
            "lambda_um": lambda_centers,
            "T_total": T_total_ds,
            "A_total_dbm": A_total_dbm
        }).to_csv(os.path.join(outdir, "attenuation_total_lsf_lambda.csv"), index=False)

    return dict(
        lambda_centers=lambda_centers,
        T_total=T_total_ds, T_each=T_each_ds,
        A_total_dbm=A_total_dbm, A_each_dbm=A_each_dbm,
        species=species,
        meta=dict(delta_um=delta_um, dlam_fine=dlam_fine, lsf_used=dict(kind=kind, W_um=W_um))
    )

# ========================= COMPARADOR CON OBS =========================
def compare_with_observable_cosine(
    lam_sim_um: np.ndarray, A_sim_dbm: np.ndarray,
    obs_path: str, L_m_obs: Optional[float]=1.0, has_header: bool=False
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Lee (λ, τ, A_dB_tot) y devuelve cosine(sim atten, obs atten) tras igualar λ."""
    df = pd.read_csv(obs_path, delim_whitespace=True,
                     header=None if not has_header else "infer", comment="#")
    lam_obs = df.iloc[:,0].to_numpy(float)
    tau_obs = df.iloc[:,1].to_numpy(float)
    AdB_obs = df.iloc[:,2].to_numpy(float)
    if L_m_obs is None:
        x = -10.0*np.log10(np.clip(tau_obs, 1e-300, 1.0))
        s = np.sum(x*AdB_obs)/max(np.sum(x*x), 1e-30)
        L_eff = float(s)
    else:
        L_eff = float(L_m_obs)
    alpha_obs = AdB_obs/max(L_eff, 1e-12)

    # Interpola la simulación al grid del observable
    A_sim_interp = np.interp(lam_obs, lam_sim_um, A_sim_dbm)
    # Cosine similarity (sobre valores no negativos)
    v1 = np.clip(A_sim_interp, 1e-12, None)
    v2 = np.clip(alpha_obs,    1e-12, None)
    cos = float(np.dot(v1, v2)/ (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-30))
    return cos, lam_obs, A_sim_interp, alpha_obs

def plot_overlay(lam_um, a_obs, a_sim, title, outpath=None):
    plt.figure(figsize=(12.5,5))
    plt.semilogy(lam_um, np.clip(a_obs, 1e-6, None), lw=1.8, label="Obs (dB/m)")
    plt.semilogy(lam_um, np.clip(a_sim, 1e-6, None), lw=1.6, label="Sim + LSF (dB/m)")
    plt.xlabel("Wavelength (µm)"); plt.ylabel("Attenuation (dB/m)")
    plt.title(title); plt.grid(True, which="both", alpha=0.3); plt.legend()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
