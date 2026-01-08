#!/usr/bin/env python3
import argparse
import numpy as np

def load_gt(path: str):
    gt = np.load(path, allow_pickle=True, mmap_mode="r")  # good for big files

    # Accept both formats:
    #  (N,2): columns [lambda_um, tau]
    #  (2,N): rows [lambda_um, tau]
    if gt.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={gt.shape}")

    if gt.shape[1] == 2:
        lam = gt[:, 0]
        tau = gt[:, 1]
        orientation = "Nx2"
    elif gt.shape[0] == 2:
        lam = gt[0, :]
        tau = gt[1, :]
        orientation = "2xN"
    else:
        raise ValueError(f"Expected (N,2) or (2,N), got shape={gt.shape}")

    return lam, tau, orientation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Input .npy GT file")
    ap.add_argument("--outfile", required=True, help="Output .npy cropped file")
    ap.add_argument("--lam-min", type=float, default=4.0)
    ap.add_argument("--lam-max", type=float, default=10.0)
    args = ap.parse_args()

    lam, tau, orientation = load_gt(args.infile)

    # Crop
    mask = (lam >= args.lam_min) & (lam <= args.lam_max)
    lam_c = np.asarray(lam[mask])
    tau_c = np.asarray(tau[mask])

    # Ensure sorted by wavelength (just in case)
    idx = np.argsort(lam_c)
    lam_c = lam_c[idx]
    tau_c = tau_c[idx]

    # Save keeping the common (N,2) format
    out = np.column_stack([lam_c, tau_c]).astype(np.float64)
    np.save(args.outfile, out)

    print(f"Loaded:  {args.infile}  (orientation={orientation}, total={lam.size})")
    print(f"Cropped: [{args.lam_min}, {args.lam_max}] µm  ->  kept={lam_c.size}")
    print(f"Saved:   {args.outfile}  shape={out.shape}")
    print(f"Range:   {lam_c.min():.6f} .. {lam_c.max():.6f} µm")

if __name__ == "__main__":
    main()
