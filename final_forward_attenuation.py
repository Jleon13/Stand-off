# ========================= INITIAL SETUP =========================
# Librerías necesarias para el cálculo espectral y el manejo de datos
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.special import wofz
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# ========================= CONSTANTS =========================
# Constantes físicas base sobre las cuales se escalan intensidades y anchos
TREF = 296.0 #Temp ref
K_B_erg = 1.380649e-16 # Boltzmann constant in erg/K
C_M_S   = 299792458.0 # Speed of light in m/s
C_CM_S  = C_M_S * 100.0 # Speed of light in cm/s
C2      = 1.43880285 #h*c/kB in cm·K
N_A     = 6.02214086e23 # Avogadro's number
N_L     = 2.47937196e19 #Density at 1 atm and TREF in molec/cm3, P0/(kB*TREF)

# ========================= DATACLASS =========================
# Describen variantes de isotopólogos y especies completas para el forward
@dataclass
class Isotopologue: # Minimal configuration per isotopologue
    iso: str
    qfile: str
    Wg: float
    Pmol: float | None = None

@dataclass
class Species: # Gas species parameters
    name: str
    mol: int
    iso: str
    qfile: str
    Wg: float
    Pmol: float
    extra_isotopologues: List[Isotopologue] = field(default_factory=list)
    Qref: float = field(init=False, default=np.nan)
    QT: float = field(init=False, default=np.nan)
    idx_all: np.ndarray | None = field(init=False, default=None)

def _prepare_forward_variants(species: List[Species], use_all: bool) -> Tuple[List[Species], List[int]]:
    """Expande cada especie en variantes individuales y recuerda su origen."""
    variants: List[Species] = []
    parent_map: List[int] = []
    for parent_idx, sp in enumerate(species):
        # Comenzamos con el isotopólogo base (siempre presente)
        variant_defs = [Isotopologue(sp.iso, sp.qfile, sp.Wg, sp.Pmol)]
        # Si se pide, añadimos las variantes configuradas adicionales
        if use_all:
            variant_defs.extend(sp.extra_isotopologues)
        for variant in variant_defs:
            variant_sp = Species(
                name=sp.name,
                mol=sp.mol,
                iso=variant.iso,
                qfile=variant.qfile,
                Wg=variant.Wg,
                Pmol=variant.Pmol if variant.Pmol is not None else sp.Pmol,
            )
            variants.append(variant_sp)  # Guardamos la copia aislada
            parent_map.append(parent_idx)  # y su índice original
    return variants, parent_map

def _isotopologue_bank(base: str = "TIPS/") -> Dict[str, Dict[str, Any]]:
    """Construye un catálogo (mol, Pmol, variantes) para cada especie."""
    return {
        "H2O": {
            "mol": 1,
            "Pmol": 1.876e+04 / 1e6,
            "variants": [
                Isotopologue(iso="1", qfile=base + "H2O/q1.txt", Wg=18.010565),
                Isotopologue(iso="2", qfile=base + "H2O/q2.txt", Wg=20.014811),
                Isotopologue(iso="3", qfile=base + "H2O/q3.txt", Wg=19.014780),
                Isotopologue(iso="4", qfile=base + "H2O/q4.txt", Wg=19.016740),
                Isotopologue(iso="5", qfile=base + "H2O/q5.txt", Wg=21.020985),
                Isotopologue(iso="6", qfile=base + "H2O/q6.txt", Wg=20.020956),
                Isotopologue(iso="7", qfile=base + "H2O/q129.txt", Wg=20.022915),
            ],
        },
        "CO2": {
            "mol": 2,
            "Pmol": 330 / 1e6,
            "variants": [
                Isotopologue(iso="1", qfile=base + "CO2/q7.txt", Wg=43.989830),
                Isotopologue(iso="2", qfile=base + "CO2/q8.txt", Wg=44.993185),
                Isotopologue(iso="3", qfile=base + "CO2/q9.txt", Wg=45.994076),
                Isotopologue(iso="4", qfile=base + "CO2/q10.txt", Wg=44.994045),
                Isotopologue(iso="5", qfile=base + "CO2/q11.txt", Wg=46.997431),
                Isotopologue(iso="6", qfile=base + "CO2/q12.txt", Wg=45.997400),
                Isotopologue(iso="7", qfile=base + "CO2/q13.txt", Wg=47.998320),
                Isotopologue(iso="8", qfile=base + "CO2/q14.txt", Wg=46.998291),
                Isotopologue(iso="9", qfile=base + "CO2/q15.txt", Wg=45.998262),
                Isotopologue(iso="10", qfile=base + "CO2/q120.txt", Wg=49.001675),
                Isotopologue(iso="A", qfile=base + "CO2/q121.txt", Wg=48.001646),
                Isotopologue(iso="B", qfile=base + "CO2/q122.txt", Wg=47.001618),
            ],
        },
        "O3": {
            "mol": 3,
            "Pmol": 0.03017 / 1e6,
            "variants": [
                Isotopologue(iso="1", qfile=base + "O3/q16.txt", Wg=47.984745),
                Isotopologue(iso="2", qfile=base + "O3/q17.txt", Wg=49.988991),
                Isotopologue(iso="3", qfile=base + "O3/q18.txt", Wg=49.988991),
                Isotopologue(iso="4", qfile=base + "O3/q19.txt", Wg=48.988960),
                Isotopologue(iso="5", qfile=base + "O3/q20.txt", Wg=48.988960),
            ],
        },
        "N2O": {
            "mol": 4,
            "Pmol": 0.32 / 1e6,
            "variants": [
                Isotopologue(iso="1", qfile=base + "N2O/q21.txt", Wg=44.001062),
                Isotopologue(iso="2", qfile=base + "N2O/q22.txt", Wg=44.998096),
                Isotopologue(iso="3", qfile=base + "N2O/q23.txt", Wg=44.998096),
                Isotopologue(iso="4", qfile=base + "N2O/q24.txt", Wg=46.005308),
                Isotopologue(iso="5", qfile=base + "N2O/q25.txt", Wg=45.005278),
            ],
        },
        "CO": {
            "mol": 5,
            "Pmol": 0.15 / 1e6,
            "variants": [
                Isotopologue(iso="1", qfile=base + "CO/q26.txt", Wg=27.994915),
                Isotopologue(iso="2", qfile=base + "CO/q27.txt", Wg=28.998270),
                Isotopologue(iso="3", qfile=base + "CO/q28.txt", Wg=29.999161),
                Isotopologue(iso="4", qfile=base + "CO/q29.txt", Wg=28.999130),
                Isotopologue(iso="5", qfile=base + "CO/q30.txt", Wg=31.002516),
                Isotopologue(iso="6", qfile=base + "CO/q31.txt", Wg=30.002485),
            ],
        },
        "CH4": {
            "mol": 6,
            "Pmol": 1.7 / 1e6,
            "variants": [
                Isotopologue(iso="1", qfile=base + "CH4/q32.txt", Wg=16.031300),
                Isotopologue(iso="2", qfile=base + "CH4/q33.txt", Wg=17.034655),
                Isotopologue(iso="3", qfile=base + "CH4/q34.txt", Wg=17.037475),
                Isotopologue(iso="4", qfile=base + "CH4/q35.txt", Wg=18.040830),
            ],
        },
        "O2": {
            "mol": 7,
            "Pmol": 0.20946,
            "variants": [
                Isotopologue(iso="1", qfile=base + "O2/q36.txt", Wg=31.989830),
                Isotopologue(iso="2", qfile=base + "O2/q37.txt", Wg=33.994076),
                Isotopologue(iso="3", qfile=base + "O2/q38.txt", Wg=32.994045),
            ],
        },
    }

# ========================= FUNCTIONS =========================
# ========================= FUNCTIONS =========================
def convert_atm(P, u):
    """Convierte presión entre unidades habituales y atmósferas."""
    if u == "gcms2":
        return P/1013.25 * 10**-3  # pasa de g/cm²/s² a atm
    elif u == "mbar":
        return P/1013.25  # de milibares a atm

def convert_vmr(ppm):
    """Pasa de partes por millón a fracción molar."""
    return ppm * 1e-6

def txt_to_npy(txt_path, out_path):
    waves = []
    trans = []

    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                w, t = map(float, line.split())
                waves.append(w)
                trans.append(t)
            except:
                continue
            
    wave = np.array(waves)
    tr   = np.array(trans)
    np.save(out_path, [wave, tr])
    print(f"Guardado como arreglo numpy en: {out_path}.npy")  # aviso al usuario


def plot_npy(wave, tr, name):
    # Dibuja la señal guardada en el numpy con un estilo limpio
    plt.figure(figsize=(16, 7))
    plt.plot(wave, tr, lw=2.5)

    plt.gca().ticklabel_format(useOffset=False)
    e = 0.0001
    plt.ylim(tr.min()-e, tr.max()+e)

    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Transmittance")
    plt.title("Spectral Transmittance of "+ name)
    plt.grid(True)
    plt.show()
    

def plot_histogram(wave, name, bins=60, range=None, ax=None):
    # Histograma de densidad espectral para chequear distribución de puntos
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 7))
    else:
        fig = ax.figure
    ax.hist(wave, bins=bins, range=range, color="#1ae981", edgecolor='white')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram wavelengths of ' + name)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()



def downsample(simulated_dir, gt_dir, output_dir):
    lam_tgt = simulated_dir[0]  
    lam_src = gt_dir[0]
    tau_src = gt_dir[1]
    edges = np.empty(lam_tgt.size + 1)
    edges[1:-1] = 0.5 * (lam_tgt[:-1] + lam_tgt[1:])
    edges[0]    = lam_tgt[0]  - 0.5 * (lam_tgt[1] - lam_tgt[0])
    edges[-1]   = lam_tgt[-1] + 0.5 * (lam_tgt[-1] - lam_tgt[-2])
    
    # Limitar los bordes al rango de datos disponible en la fuente
    edges[0] = max(edges[0], lam_src[0])
    edges[-1] = min(edges[-1], lam_src[-1])
    
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (tau_src[:-1] + tau_src[1:]) * (lam_src[1:] - lam_src[:-1]))])
    def cum_at(x):
        return np.interp(x, lam_src, cum)

    area = cum_at(edges[1:]) - cum_at(edges[:-1])
    tau_ds = area / (edges[1:] - edges[:-1])
    valid = (edges[:-1] >= lam_src[0]) & (edges[1:] <= lam_src[-1])
    tau_ds[~valid] = np.nan
    out = np.vstack([lam_tgt, tau_ds])
    np.save(output_dir, out)


def load_Q_vals(qfile: str, Tref: float, T: float) -> Tuple[float, float]:
    """Carga el archivo q*.txt y calcula Q(Tref) y Q(T) por interpolación."""
    if not os.path.isfile(qfile):
        raise FileNotFoundError(f"Missing Q(T) file: {qfile}")
    try:
        df = pd.read_csv(qfile, header=None, delim_whitespace=True, comment='#')
        if df.shape[1] < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(qfile, header=None, sep=r"[\s,;]+", engine="python", comment='#')
    # Se asume que la primera columna es temperatura, la segunda Q(T)
    Tcol = df.iloc[:, 0].astype(float).to_numpy()
    Qcol = df.iloc[:, 1].astype(float).to_numpy()
    Qref = np.interp(Tref, Tcol, Qcol)
    QT   = np.interp(T,    Tcol, Qcol)
    return float(Qref), float(QT) #TIPS

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
                mol.append(int(s[0])); iso.append(s[1].strip())
                nu0.append(float(s[2])); Sref.append(float(s[3]))
                A.append(float(s[4])); g_air.append(float(s[5]))
                g_self.append(float(s[6])); Elow.append(float(s[7]))
                n_air.append(float(s[8])); shift.append(float(s[9]))
            except Exception:
                continue

    return dict(
        mol=np.asarray(mol, dtype=np.int32), iso=np.asarray(iso, dtype='<U10'),
        nu0=np.asarray(nu0, dtype=np.float64), Sref=np.asarray(Sref, dtype=np.float64),
        A=np.asarray(A, dtype=np.float64), g_air=np.asarray(g_air, dtype=np.float64),
        g_self=np.asarray(g_self, dtype=np.float64), Elow=np.asarray(Elow, dtype=np.float64),
        n_air=np.asarray(n_air, dtype=np.float64), shift=np.asarray(shift, dtype=np.float64)
    )

def count_lines_by_index(parfile: str, index: int) -> int:
    """Return how many HITRAN lines start with the given molecule index in column 1."""
    H = read_hitran_par_minimal(parfile)
    if 'mol' not in H:
        raise KeyError("HITRAN data missing 'mol' column")
    return int(np.count_nonzero(H['mol'] == index))

def voigt_profile(nu: np.ndarray, nu0_shifted: np.ndarray, alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Return normalized Voigt profile f_V(nu)."""
    s2 = np.sqrt(np.log(2.0))
    x = s2 * (nu[:, None] - nu0_shifted[None, :]) / alpha[None, :]
    y = s2 * (gamma[None, :] / alpha[None, :])
    z = x + 1j * y
    w = wofz(z)
    fV = s2 / np.sqrt(np.pi) / alpha[None, :] * np.real(w)
    return fV #Fvoigt normalized, is line shape function

def transmittance_for_gas_tile(nu_vec: np.ndarray, H: Dict[str, np.ndarray], sp: Species,
                               Tgas: float, pres: float, Lm: float, mask_lines: np.ndarray) -> np.ndarray:
    """Compute T(nu) for a single gas over a spectral tile."""
    if not np.any(mask_lines):
        return np.ones_like(nu_vec)

    nu0, Sref, g_air, g_self, Elow, n_air, shift = (
        H['nu0'][mask_lines], H['Sref'][mask_lines], H['g_air'][mask_lines],
        H['g_self'][mask_lines], H['Elow'][mask_lines], H['n_air'][mask_lines],
        H['shift'][mask_lines]
    )

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
    """Return the configured `Species` objects; each carries the top isotopologue plus a list of alternatives."""
    bank = _isotopologue_bank()
    order = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2']
    species_list: List[Species] = []
    for name in order:
        entry = bank[name]
        variants = entry['variants']
        if not variants:
            continue
        main, *others = variants
        species_list.append(Species(
            name=name,
            mol=entry['mol'],
            iso=main.iso,
            qfile=main.qfile,
            Wg=main.Wg,
            Pmol=entry['Pmol'],
            extra_isotopologues=list(others),
        ))
    return species_list

# ========================= MAIN FUNCTION =========================
def run_simulation(
    species: List[Species],
    parfile: str = 'PARS/ALL.par',
    nu_min: float = 666.67,
    nu_max: float = 10000.0,
    dnu: float = 0.01,
    tileW: float = 20.0,
    guard: float = 5.0,
    temp_K: float = 296.0,
    L_m: float = 1.0,
    pres: float = 1.0,
    delta_um: float = 0.020,
    save_csv: bool = False,
    outdir: str = 'OUT',
    make_plots: bool = True,
    att: bool = True,
    use_all_isotopologues: bool = False,
    simulated_dir: str = 'SIMULATED',
    transmission_npy_name: str | None = None,
    species_to_use: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Compute transmittances/attenuations and return sampled results.
    Set ``att`` to False to keep the raw transmittance signatures and skip attenuation plotting.
    ``simulated_dir`` and ``transmission_npy_name`` control the optional export of transmittance data as a stacked
    numpy array saved inside the provided directory.
    When ``use_all_isotopologues`` is True each species will expand to its configured isotopologues (and that
    configuration must provide iso-specific q-files and broadening coefficients) before running the forward model.
    ``species_to_use`` is an optional list of species names (e.g., ['H2O', 'CO2']) to filter which species
    from the input list will actually be simulated. If None, all species are used.
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError(f"Missing HITRAN .par: {parfile}")
    
    # Filter species based on species_to_use parameter
    if species_to_use is not None:
        species_to_use_upper = [s.upper() for s in species_to_use]
        species = [sp for sp in species if sp.name.upper() in species_to_use_upper]
        if not species:
            raise ValueError(f"No species found matching: {species_to_use}")
        print(f"Using species: {[sp.name for sp in species]}")
    
    forward_variants, variant_to_parent = _prepare_forward_variants(species, use_all_isotopologues)
    for var in forward_variants:
        print(f"DEBUG: {var.name} iso={var.iso} qfile={var.qfile} exists={os.path.isfile(var.qfile)}")
        if not os.path.isfile(var.qfile):
            raise FileNotFoundError(f"Missing q-file for {var.name} iso {var.iso}: {var.qfile}")

    # Load Q(T)
    for var in forward_variants:
        Qref, QT = load_Q_vals(var.qfile, TREF, temp_K)
        var.Qref, var.QT = Qref, QT

    # Read HITRAN .par
    H = read_hitran_par_minimal(parfile)

    # Line indices per species
    for var in forward_variants:
        var.idx_all = (H['mol'] == var.mol) & (H['iso'] == var.iso)

    # Spectral tiling
    edges_tiles = np.arange(nu_min, nu_max + 1e-9, tileW)
    nu_all_parts, T_prod_all_parts, T_sum_all_parts = [], [], []
    T_each_acc = [[] for _ in species]

    for a in edges_tiles:
        b = min(a + tileW, nu_max)
        a_ext, b_ext = max(nu_min, a - guard), min(nu_max, b + guard)
        nu_ext = np.arange(a_ext, b_ext + 1e-12, dnu)

        T_ext_each_variants = np.ones((len(forward_variants), nu_ext.size), dtype=np.float64)
        for k, var in enumerate(forward_variants):
            idx_tile = var.idx_all & (H['nu0'] >= a_ext) & (H['nu0'] <= b_ext)
            if np.any(idx_tile):
                T_ext_each_variants[k, :] = transmittance_for_gas_tile(nu_ext, H, var, temp_K, pres, L_m, idx_tile)

        T_ext_prod = np.prod(T_ext_each_variants, axis=0)
        T_species_ext = np.ones((len(species), nu_ext.size), dtype=np.float64)
        for variant_idx, parent_idx in enumerate(variant_to_parent):
            T_species_ext[parent_idx] *= T_ext_each_variants[variant_idx, :]
        T_ext_sum = np.sum(T_species_ext, axis=0)

        keep = (nu_ext >= a) & (nu_ext <= b)
        nu_all_parts.append(nu_ext[keep])
        T_prod_all_parts.append(T_ext_prod[keep])
        T_sum_all_parts.append(T_ext_sum[keep])
        for k in range(len(species)):
            T_each_acc[k].append(T_species_ext[k, keep])

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

    # Attenuation in dB/m (optional)
    if att:
        invL = 1.0 / max(L_m, 1e-300)
        A_dbm_lam_each = [-(10.0 * invL) * np.log10(np.clip(arr, 1e-300, 1.0)) for arr in T_each_samp]
        A_dbm_lam_sum = np.sum(np.stack(A_dbm_lam_each, axis=0), axis=0)
    else:
        A_dbm_lam_each = None
        A_dbm_lam_sum = None

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

        def _log_axis_limits(arr: np.ndarray) -> Tuple[float, float]:
            arr = np.asarray(arr)
            mask = (arr > 0) & np.isfinite(arr)
            if not np.any(mask):
                return 1e-15, 1e4
            min_val = np.min(arr[mask])
            max_val = np.max(arr[mask])
            lower = max(1e-15, min_val * 0.9)
            upper = max_val * 10.0
            if lower >= upper:
                upper = lower * 10
            return lower, upper

        def _lin_axis_limits(arr: np.ndarray) -> Tuple[float, float]:
            arr = np.asarray(arr)
            mask = np.isfinite(arr)
            if not np.any(mask):
                return 0.0, 1.0
            min_val = np.min(arr[mask])
            max_val = np.max(arr[mask])
            padding = max(1e-3, 0.05 * max(1.0, max_val - min_val))
            lower = min_val - padding
            upper = max_val + padding
            return lower, upper

        def save_fig(fig, basename: str):
            for ext in EXPORT_FORMATS:
                dpi = DPI_EXPORT if ext.lower() in ("png", "jpg", "jpeg", "tif", "tiff") else None
                fig.savefig(
                    os.path.join(outdir, f"{basename}.{ext}"),
                    dpi=dpi, bbox_inches="tight", pad_inches=0.05
                )

        if att:
            # ---------- Figure 1: total attenuation ----------
            A_sum_plot = _positivize(A_dbm_lam_sum)
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.semilogy(lambda_centers, A_sum_plot, lw=2.5, color="#1f77b4", label="Total")
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Attenuation (dB/m)")
            ax.set_title("Combined atmospheric attenuation")
            ax.set_ylim(*_log_axis_limits(A_sum_plot))
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
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
                ax.semilogy(lambda_centers, arr, lw=2, label=spc.name, color=color)
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
            ax.set_ylim(*_log_axis_limits(np.concatenate(A_each_plot)))
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.grid(True, which='major', axis='both', color='#bbb', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.grid(True, which='minor', axis='both', color='#eee', linestyle=':', linewidth=0.5, alpha=0.3)
            ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, ncol=2)
            plt.tight_layout()
            save_fig(fig, "attenuation_by_gas")
            plt.show()
            plt.close(fig)
        else:
            # ---------- Figure 1: total transmittance ----------
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(lambda_centers, T_prod_samp, lw=2.5, color="#1f77b4", label="Total")
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Transmittance")
            ax.set_title("Combined atmospheric transmittance")
            ax.set_ylim(*_lin_axis_limits(T_prod_samp))
            ax.grid(True, which='both', color='#bbb', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True)
            plt.tight_layout()
            save_fig(fig, "transmittance_total")
            plt.show()
            plt.close(fig)

            # ---------- Figure 2: transmittance by gas ----------
            fig, ax = plt.subplots(figsize=FIGSIZE)
            colors = plt.cm.Set2(np.linspace(0, 1, len(species)))
            for arr, spc, color in zip(T_each_samp, species, colors):
                ax.plot(lambda_centers, arr, lw=2, label=spc.name, color=color)
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Transmittance")
            ax.set_title("Spectral transmittance by gas")
            ax.set_ylim(*_lin_axis_limits(np.concatenate(T_each_samp)))
            ax.grid(True, which='both', color='#bbb', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True, ncol=2)
            plt.tight_layout()
            save_fig(fig, "transmittance_by_gas")
            plt.show()
            plt.close(fig)

    # ===================== CSV EXPORT (optional) =====================
    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        # Total
        df_total = {
            "lambda_um": lambda_centers,
            "T_total": T_prod_samp,
        }
        if att:
            df_total["A_total_dbm"] = A_dbm_lam_sum
        pd.DataFrame(df_total).to_csv(
            os.path.join(outdir, "transmission_total.csv" if not att else "attenuation_total.csv"),
            index=False
        )

        # By gas (attenuation and/or transmittance)
        cols = {"lambda_um": lambda_centers}
        for sp, T_arr in zip(species, T_each_samp):
            cols[f"T_{sp.name}"] = T_arr
        if att:
            for sp, A_arr in zip(species, A_dbm_lam_each):
                cols[f"A_{sp.name}_dbm"] = A_arr
        pd.DataFrame(cols).to_csv(
            os.path.join(outdir, "transmission_by_gas.csv" if not att else "attenuation_by_gas.csv"),
            index=False
        )

    result = dict(
        lambda_centers=lambda_centers,
        T_prod_samp=T_prod_samp,
        T_each_samp=T_each_samp,
        species=species,
        att=att,
    )
    if att:
        result["A_dbm_lam_sum"] = A_dbm_lam_sum
        result["A_dbm_lam_each"] = A_dbm_lam_each
    else:
        result["A_dbm_lam_sum"] = None
        result["A_dbm_lam_each"] = None
    if transmission_npy_name:
        sim_root = simulated_dir or 'SIMULATED'
        sim_root = os.path.abspath(sim_root) if os.path.isabs(sim_root) else os.path.abspath(os.path.join(os.getcwd(), sim_root))
        os.makedirs(sim_root, exist_ok=True)
        if not transmission_npy_name.lower().endswith('.npy'):
            transmission_npy_name = f"{transmission_npy_name}.npy"
        transmission_path = os.path.join(sim_root, transmission_npy_name)
        stacked = np.vstack([lambda_centers, T_prod_samp])
        np.save(transmission_path, stacked)
        result["transmission_npy_path"] = transmission_path
    else:
        result["transmission_npy_path"] = None
    return result
