# final_forward_attenuation_lsf.py
import os, math
import numpy as np
import pandas as pd
from scipy.special import wofz
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# ========================= CONSTANTS =========================
TREF = 296.0
K_B_erg = 1.380649e-16
C_M_S   = 299792458.0
C_CM_S  = C_M_S * 100.0
C2      = 1.43880285
N_A     = 6.02214086e23
N_L     = 2.47937196e19  # molec/cm^3 at 1 atm and TREF

# ========================= DATACLASS =========================
@dataclass
class Isotopologue:
    iso: str
    qfile: str
    Wg: float
    Pmol: float | None = None  # optional override of VMR for this isotopologue

@dataclass
class Species:
    name: str
    mol: int
    iso: str
    qfile: str
    Wg: float
    Pmol: float  # VMR (mole fraction) chi
    extra_isotopologues: List[Isotopologue] = field(default_factory=list)
    Qref: float = field(init=False, default=np.nan)
    QT: float   = field(init=False, default=np.nan)
    idx_all: np.ndarray | None = field(init=False, default=None)

# ========================= HELPERS =========================
def convert_atm(P, u):
    if u == "gcms2":
        return P/1013.25 * 1e-3
    if u == "mbar":
        return P/1013.25
    raise ValueError(f"Unknown pressure unit: {u}")

def convert_vmr(ppm):
    return ppm * 1e-6

def load_Q_vals(qfile: str, Tref: float, T: float) -> Tuple[float, float]:
    if not os.path.isfile(qfile):
        raise FileNotFoundError(f"Missing Q(T) file: {qfile}")
    try:
        df = pd.read_csv(qfile, header=None, delim_whitespace=True, comment="#")
        if df.shape[1] < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(qfile, header=None, sep=r"[\s,;]+", engine="python", comment="#")
    Tcol = df.iloc[:, 0].astype(float).to_numpy()
    Qcol = df.iloc[:, 1].astype(float).to_numpy()
    Qref = float(np.interp(Tref, Tcol, Qcol))
    QT   = float(np.interp(T,    Tcol, Qcol))
    return Qref, QT

def read_hitran_par_minimal(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find {path}")
    widths = [2,1,12,10,10,5,5,10,4,8, 15,15,15,15,6,12,1,7,7]
    use_up_to = 10
    cum = np.cumsum([0] + widths)
    sl = [(cum[i], cum[i+1]) for i in range(use_up_to)]

    mol, iso, nu0, Sref, A, g_air, g_self, Elow, n_air, shift = ([] for _ in range(10))
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s = [line[a:b] for a, b in sl]
                mol.append(int(s[0])); iso.append(s[1].strip())
                nu0.append(float(s[2])); Sref.append(float(s[3]))
                A.append(float(s[4])); g_air.append(float(s[5]))
                g_self.append(float(s[6])); Elow.append(float(s[7]))
                n_air.append(float(s[8])); shift.append(float(s[9]))
            except Exception:
                continue

    return dict(
        mol=np.asarray(mol, dtype=np.int32),
        iso=np.asarray(iso, dtype="<U10"),
        nu0=np.asarray(nu0, dtype=np.float64),
        Sref=np.asarray(Sref, dtype=np.float64),
        A=np.asarray(A, dtype=np.float64),
        g_air=np.asarray(g_air, dtype=np.float64),
        g_self=np.asarray(g_self, dtype=np.float64),
        Elow=np.asarray(Elow, dtype=np.float64),
        n_air=np.asarray(n_air, dtype=np.float64),
        shift=np.asarray(shift, dtype=np.float64),
    )

def voigt_profile(nu: np.ndarray, nu0_shifted: np.ndarray, alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    s2 = np.sqrt(np.log(2.0))
    x = s2 * (nu[:, None] - nu0_shifted[None, :]) / alpha[None, :]
    y = s2 * (gamma[None, :] / alpha[None, :])
    z = x + 1j * y
    w = wofz(z)
    return (s2/np.sqrt(np.pi)) * (np.real(w)/alpha[None, :])

def transmittance_for_gas_tile(
    nu_vec: np.ndarray,
    H: Dict[str, np.ndarray],
    sp: Species,
    Tgas: float,
    P_atm_total: float,
    Lm: float,
    mask_lines: np.ndarray
) -> np.ndarray:
    if not np.any(mask_lines):
        return np.ones_like(nu_vec)

    nu0   = H["nu0"][mask_lines]
    Sref  = H["Sref"][mask_lines]
    g_air = H["g_air"][mask_lines]
    g_self= H["g_self"][mask_lines]
    Elow  = H["Elow"][mask_lines]
    n_air = H["n_air"][mask_lines]
    shift = H["shift"][mask_lines]  # cm^-1/atm (air shift)

    chi = float(sp.Pmol)
    Pself = chi * P_atm_total
    Pair  = max(P_atm_total - Pself, 0.0)

    # pressure shift should use air/foreign pressure
    nu0s = nu0 + shift * Pair

    # S(T)
    S_T = (Sref * (sp.Qref / sp.QT) *
           np.exp(-C2 * Elow / Tgas) / np.exp(-C2 * Elow / TREF) *
           (1.0 - np.exp(-C2 * nu0 / Tgas)) / (1.0 - np.exp(-C2 * nu0 / TREF)))

    # column density (#/cm^2)
    Lcm = Lm * 100.0
    col = (TREF / Tgas) * N_L * P_atm_total * chi * Lcm

    # widths
    alpha = nu0 / C_CM_S * np.sqrt(2.0 * N_A * K_B_erg * Tgas * np.log(2.0) / sp.Wg)
    gamma = ((TREF / Tgas) ** n_air) * (g_air * Pair + g_self * Pself)

    fV = voigt_profile(nu_vec, nu0s, alpha, gamma)
    tau = fV @ (S_T * col)
    return np.exp(-tau)

def _prepare_forward_variants(species: List[Species], use_all: bool) -> Tuple[List[Species], List[int]]:
    variants: List[Species] = []
    parent_map: List[int] = []
    for parent_idx, sp in enumerate(species):
        variant_defs = [Isotopologue(sp.iso, sp.qfile, sp.Wg, sp.Pmol)]
        if use_all:
            variant_defs.extend(sp.extra_isotopologues)
        for v in variant_defs:
            variants.append(Species(
                name=sp.name, mol=sp.mol,
                iso=v.iso, qfile=v.qfile, Wg=v.Wg,
                Pmol=float(v.Pmol) if v.Pmol is not None else float(sp.Pmol),
            ))
            parent_map.append(parent_idx)
    return variants, parent_map

def _isotopologue_bank(base: str) -> Dict[str, Dict[str, Any]]:
    return {
        "H2O": {"mol": 1, "Pmol": 1.876e+04/1e6, "variants": [
            Isotopologue("1", base+"H2O/q1.txt",   18.010565),
            Isotopologue("2", base+"H2O/q2.txt",   20.014811),
            Isotopologue("3", base+"H2O/q3.txt",   19.014780),
            Isotopologue("4", base+"H2O/q4.txt",   19.016740),
            Isotopologue("5", base+"H2O/q5.txt",   21.020985),
            Isotopologue("6", base+"H2O/q6.txt",   20.020956),
            Isotopologue("7", base+"H2O/q129.txt", 20.022915),
        ]},
        "CO2": {"mol": 2, "Pmol": 330/1e6, "variants": [
            Isotopologue("1", base+"CO2/q7.txt",   43.989830),
            Isotopologue("2", base+"CO2/q8.txt",   44.993185),
            Isotopologue("3", base+"CO2/q9.txt",   45.994076),
            Isotopologue("4", base+"CO2/q10.txt",  44.994045),
            Isotopologue("5", base+"CO2/q11.txt",  46.997431),
            Isotopologue("6", base+"CO2/q12.txt",  45.997400),
            Isotopologue("7", base+"CO2/q13.txt",  47.998320),
            Isotopologue("8", base+"CO2/q14.txt",  46.998291),
            Isotopologue("9", base+"CO2/q15.txt",  45.998262),
            Isotopologue("10",base+"CO2/q120.txt", 49.001675),
            Isotopologue("A", base+"CO2/q121.txt", 48.001646),
            Isotopologue("B", base+"CO2/q122.txt", 47.001618),
        ]},
        "O3": {"mol": 3, "Pmol": 0.03017/1e6, "variants": [
            Isotopologue("1", base+"O3/q16.txt", 47.984745),
            Isotopologue("2", base+"O3/q17.txt", 49.988991),
            Isotopologue("3", base+"O3/q18.txt", 49.988991),
            Isotopologue("4", base+"O3/q19.txt", 48.988960),
            Isotopologue("5", base+"O3/q20.txt", 48.988960),
        ]},
        "N2O": {"mol": 4, "Pmol": 0.32/1e6, "variants": [
            Isotopologue("1", base+"N2O/q21.txt", 44.001062),
            Isotopologue("2", base+"N2O/q22.txt", 44.998096),
            Isotopologue("3", base+"N2O/q23.txt", 44.998096),
            Isotopologue("4", base+"N2O/q24.txt", 46.005308),
            Isotopologue("5", base+"N2O/q25.txt", 45.005278),
        ]},
        "CO": {"mol": 5, "Pmol": 0.15/1e6, "variants": [
            Isotopologue("1", base+"CO/q26.txt", 27.994915),
            Isotopologue("2", base+"CO/q27.txt", 28.998270),
            Isotopologue("3", base+"CO/q28.txt", 29.999161),
            Isotopologue("4", base+"CO/q29.txt", 28.999130),
            Isotopologue("5", base+"CO/q30.txt", 31.002516),
            Isotopologue("6", base+"CO/q31.txt", 30.002485),
        ]},
        "CH4": {"mol": 6, "Pmol": 1.7/1e6, "variants": [
            Isotopologue("1", base+"CH4/q32.txt", 16.031300),
            Isotopologue("2", base+"CH4/q33.txt", 17.034655),
            Isotopologue("3", base+"CH4/q34.txt", 17.037475),
            Isotopologue("4", base+"CH4/q35.txt", 18.040830),
        ]},
        "O2": {"mol": 7, "Pmol": 0.20946, "variants": [
            Isotopologue("1", base+"O2/q36.txt", 31.989830),
            Isotopologue("2", base+"O2/q37.txt", 33.994076),
            Isotopologue("3", base+"O2/q38.txt", 32.994045),
        ]},
    }

def default_species(
    tips_base: str = "/home/jleon13/Documents/AFOSR project/Py4catsForward/forwardOscar/Stand-off/TIPS/"
) -> List[Species]:
    bank = _isotopologue_bank(tips_base)
    order = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2"]
    out: List[Species] = []
    for name in order:
        entry = bank[name]
        main, *others = entry["variants"]
        out.append(Species(
            name=name, mol=entry["mol"],
            iso=main.iso, qfile=main.qfile, Wg=main.Wg,
            Pmol=float(entry["Pmol"]),
            extra_isotopologues=list(others),
        ))
    return out

# ========================= LSF in wavenumber (nu) =========================
def _gauss_kernel_nu(FWHM_cm1: float, dnu: float, truncate_sigma: float = 4.0) -> np.ndarray:
    sigma = FWHM_cm1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(2, int(truncate_sigma * sigma / max(dnu, 1e-30)))
    x = np.arange(-half, half + 1) * dnu
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k

def build_lsf_nu(kind: str, W_cm1: float, dnu: float) -> np.ndarray:
    kind = (kind or "gaussian").lower()
    if kind == "gaussian":
        return _gauss_kernel_nu(FWHM_cm1=W_cm1, dnu=dnu, truncate_sigma=4.0)
    raise ValueError(f"Unknown LSF kind: {kind}")

# ========================= Bin-average in lambda (trapz) =========================
def bin_average_irregular(x_sorted: np.ndarray, y_sorted: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.empty(edges.size - 1, float)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        l = np.searchsorted(x_sorted, a, side="left")
        r = np.searchsorted(x_sorted, b, side="right")
        xseg = np.r_[a, x_sorted[l:r], b]
        yseg = np.r_[np.interp(a, x_sorted, y_sorted),
                     y_sorted[l:r],
                     np.interp(b, x_sorted, y_sorted)]
        out[i] = np.trapz(yseg, xseg) / max(b - a, 1e-30)
    return centers, out

# ========================= MAIN =========================
def run_simulation(
    species: List[Species],
    parfile: str,
    nu_min: float,
    nu_max: float,
    dnu: float = 0.01,
    tileW: float = 20.0,
    guard: float = 10.0,
    temp_K: float = 296.0,
    L_m: float = 1.0,
    pres: float = 1.0,                 # total pressure [atm]
    delta_um: float = 0.020,           # output sampling step in lambda
    lsf: Optional[Dict[str, Any]] = None,   # {"kind":"gaussian","W_cm1":2.0,"domain":"T"|"tau"}
    use_all_isotopologues: bool = False,
    species_to_use: Optional[List[str]] = None,
    transmission_npy_path: Optional[str] = None,  # if set, saves [lambda_centers; T_total] to this path
) -> Dict[str, Any]:

    if not os.path.isfile(parfile):
        raise FileNotFoundError(f"Missing HITRAN .par: {parfile}")

    if species_to_use is not None:
        S = {s.upper() for s in species_to_use}
        species = [sp for sp in species if sp.name.upper() in S]
        if not species:
            raise ValueError(f"No species matched: {species_to_use}")

    forward_variants, variant_to_parent = _prepare_forward_variants(species, use_all_isotopologues)

    for var in forward_variants:
        if not os.path.isfile(var.qfile):
            raise FileNotFoundError(f"Missing q-file for {var.name} iso {var.iso}: {var.qfile}")
        var.Qref, var.QT = load_Q_vals(var.qfile, TREF, temp_K)

    H = read_hitran_par_minimal(parfile)
    for var in forward_variants:
        var.idx_all = (H["mol"] == var.mol) & (H["iso"] == var.iso)

    # --- tiling in nu ---
    edges_tiles = np.arange(nu_min, nu_max + 1e-9, tileW)
    nu_all_parts = []
    T_each_species_parts = [[] for _ in range(len(species))]

    for a in edges_tiles:
        b = min(a + tileW, nu_max)
        a_ext, b_ext = max(nu_min, a - guard), min(nu_max, b + guard)
        nu_ext = np.arange(a_ext, b_ext + 1e-12, dnu)

        T_ext_each_variants = np.ones((len(forward_variants), nu_ext.size), dtype=np.float64)
        for k, var in enumerate(forward_variants):
            idx_tile = var.idx_all & (H["nu0"] >= a_ext) & (H["nu0"] <= b_ext)
            if np.any(idx_tile):
                T_ext_each_variants[k, :] = transmittance_for_gas_tile(
                    nu_ext, H, var, temp_K, pres, L_m, idx_tile
                )

        # collapse isotopologues -> per-species
        T_species_ext = np.ones((len(species), nu_ext.size), dtype=np.float64)
        for vidx, parent_idx in enumerate(variant_to_parent):
            T_species_ext[parent_idx] *= T_ext_each_variants[vidx, :]

        keep = (nu_ext >= a) & (nu_ext <= b)
        nu_all_parts.append(nu_ext[keep])
        for sidx in range(len(species)):
            T_each_species_parts[sidx].append(T_species_ext[sidx, keep])

    nu_all = np.concatenate(nu_all_parts)  # increasing
    T_each_species = [np.concatenate(T_each_species_parts[i]) for i in range(len(species))]

    # TOTAL (physics) BEFORE any resample: product across species
    T_total_nu = np.ones_like(nu_all)
    for arr in T_each_species:
        T_total_nu *= arr

    # --- LSF in nu (recommended) ---
    lsf_used = None
    if lsf is not None:
        kind = (lsf.get("kind", "gaussian") or "gaussian").lower()
        W_cm1 = float(lsf.get("W_cm1", 2.0))
        domain = (lsf.get("domain", "T") or "T").lower()  # "t" or "tau"
        k = build_lsf_nu(kind=kind, W_cm1=W_cm1, dnu=dnu)

        if domain == "tau":
            tau = -np.log(np.clip(T_total_nu, 1e-300, 1.0))
            tau_c = np.convolve(tau, k, mode="same")
            T_total_nu = np.exp(-tau_c)
        else:
            T_total_nu = np.convolve(T_total_nu, k, mode="same")

        lsf_used = dict(kind=kind, W_cm1=W_cm1, domain=domain, dnu=dnu, kernel_len=len(k))

    # --- nu -> lambda and bin-average in lambda ---
    lam_raw = 1e4 / nu_all
    ord_idx = np.argsort(lam_raw)
    lam_raw = lam_raw[ord_idx]
    T_total_raw = T_total_nu[ord_idx]

    lam_min = math.ceil(lam_raw.min() / delta_um) * delta_um
    lam_max = math.floor(lam_raw.max() / delta_um) * delta_um
    edges = np.arange(lam_min, lam_max + 1e-12, delta_um)

    lambda_centers, T_total_ds = bin_average_irregular(lam_raw, T_total_raw, edges)

    # per-gas outputs on same lambda grid (debug only; DO NOT use to rebuild total)
    T_each_ds = []
    for arr in T_each_species:
        arr_lam = arr[ord_idx]
        _, arr_ds = bin_average_irregular(lam_raw, arr_lam, edges)
        T_each_ds.append(arr_ds)

    invL = 1.0 / max(L_m, 1e-300)
    A_total_dbm = -(10.0 * invL) * np.log10(np.clip(T_total_ds, 1e-300, 1.0))

    # --- Save npy if requested ---
    saved_path = None
    if transmission_npy_path is not None:
        transmission_npy_path = os.path.abspath(transmission_npy_path)
        os.makedirs(os.path.dirname(transmission_npy_path), exist_ok=True)
        np.save(transmission_npy_path, np.vstack([lambda_centers, T_total_ds]))
        saved_path = transmission_npy_path

    return dict(
        lambda_centers=lambda_centers,
        T_total=T_total_ds,
        T_each=T_each_ds,
        A_total_dbm=A_total_dbm,
        species=species,
        transmission_npy_path=saved_path,
        meta=dict(pres_atm_total=pres, delta_um=delta_um, dnu=dnu, tileW=tileW, guard=guard, lsf=lsf_used),
    )
