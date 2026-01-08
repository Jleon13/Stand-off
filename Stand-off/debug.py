# run_case.py
from final_forward_attenuation import *

def main():
    Ptotal = convert_atm(1013, "mbar")
    T = 288.2

    sp = default_species()
    for s in sp:
        if s.name == "N2O":
            s.Pmol = 3.1992104613866274e-07
        else:
            s.Pmol = 0.0

    res = run_simulation(
        species=sp,
        parfile="/home/jleon13/Documents/AFOSR project/Py4catsForward/Py4cats_Tinkering/680HIT87b.par",
        nu_min=1000, nu_max=2500, dnu=0.01,
        tileW=20.0, guard=5.0,
        temp_K=T, L_m=1.0, pres=Ptotal,
        delta_um=0.020,
        save_csv=True, outdir="OUT", make_plots=False,
        att=False,
        transmission_npy_name="Simulated_N2O_4-10_1m.npy",
        use_all_isotopologues=True,
        species_to_use=["N2O"],
    )

    # solo para que tengas variables “vivas” en el debugger
    lambda_um = res["lambda_centers"]
    Tprod = res["T_prod_samp"]
    Tn2o  = res["T_each_samp"][0]
    print(lambda_um.shape, Tprod.min(), Tprod.max(), Tn2o.min(), Tn2o.max())

if __name__ == "__main__":
    main()
