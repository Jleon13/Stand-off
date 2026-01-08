from final_forward_attenuation_lsf import *

Ptotal = convert_atm(1013, "mbar")
T = 288.2
L = 1.0

vmr = {
    "H2O": 1.876e4/1e6,
    "CO2": 0.00033,
    "O3":  0.03017/1e6,
    "N2O": 0.32/1e6,
    "CO":  0.15/1e6,
    "CH4": 1.7/1e6,
    "O2":  0.20946,
}

sp = default_species()
for s in sp:
    s.Pmol = float(vmr[s.name])

res = run_simulation(
    species=sp,
    parfile="/home/jleon13/Documents/AFOSR project/Py4catsForward/Py4cats_Tinkering/680HIT87b.par",
    nu_min=1000, nu_max=2500,
    dnu=0.01, tileW=20.0, guard=10.0,
    temp_K=T, L_m=L, pres=Ptotal,
    delta_um=0.020,
    # LSF en nu (cm^-1). Prueba W_cm1 ~ 1–5 y domain="T" vs "tau"
    lsf={"kind": "gaussian", "W_cm1": 2.0, "domain": "T"},
    use_all_isotopologues=True,
    transmission_npy_path="OUT/Simulated_All_4-10_1m.npy",
)

print("Keys:", res.keys())
print("Saved:", res["transmission_npy_path"])

lam = res["lambda_centers"]
Ttot = res["T_total"]
