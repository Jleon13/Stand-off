import numpy as np

def inspect_grid(lam_um, name=""):
    lam_um = np.asarray(lam_um).copy()
    # asegurar orden ascendente
    if lam_um[0] > lam_um[-1]:
        lam_um = lam_um[::-1]

    dlam = np.diff(lam_um)
    nu = 1e4 / lam_um
    # nu decrece si lam crece -> usar abs
    dnu = np.abs(np.diff(nu))

    def stats(x):
        return dict(min=float(np.min(x)), p50=float(np.median(x)), max=float(np.max(x)),
                    mean=float(np.mean(x)), cv=float(np.std(x)/np.mean(x)))

    print(f"\n=== {name} ===")
    print("N:", lam_um.size)
    print("dlam stats:", stats(dlam))
    print("dnu  stats:", stats(dnu))
    print("Interpretación: si CV(dnu) << CV(dlam), tu grilla es 'más constante' en cm^-1.")

# cargar npy tipo [2,N]
gt = np.load("/mnt/data/Standard_All_4-10um_1m.npy")
sim = np.load("/mnt/data/Simulated_All_4-10_1m.npy")

inspect_grid(gt[0], "GT SpectralCalc")
inspect_grid(sim[0], "Simulado")
