"""SOURCE of the DESC Delta_res data -- generates the *_desc_bump_nlam_20_delta_omega_0.002.npz
files that compute_delta_res.py reduces.

This is the upstream step: DESC equilibrium (.h5) -> TrappedResonance objective ->
npz. `compute_delta_res.py` then reduces that npz to the plotted curve.

Requires the DESC fork that provides the TrappedResonance objective -- it is NOT
in upstream DESC. See PROVENANCE.md next to this file. In short:

    repo:   https://github.com/jlabbate15/DESC_TrappedRes
    COMMIT: 4b77719dfa44ef26df29ba80b9c620ecb19faaf4   (2026-07-06)
    branch containing it: ejp-merge-master
    file:   desc/objectives/_trapped_resonance.py

Check out the COMMIT, not the branch: as of 2026-08-29 origin/ejp-merge-master
is 70 commits AHEAD of the pin, so the branch tip will NOT reproduce this data.

    git clone https://github.com/jlabbate15/DESC_TrappedRes
    cd DESC_TrappedRes && git checkout 4b77719dfa44ef26df29ba80b9c620ecb19faaf4

Point --desc-path at that checkout. This script verifies the checkout's HEAD and
refuses to run on a mismatch unless --allow-version-mismatch is given. Written for GPU (the original runs
used set_device("gpu")); --device cpu works but is slow.

Objective parameters below are those of the original March 2026 run. Verified:
re-running this reproduces the packaged npz to 1e-7 relative -- exact agreement
on the beta scan at Bcrit=6.24, max abs difference 8.9e-7 on the beta=0 Bcrit
curve (GPU floating-point noise, not a real difference).

Usage:
    python generate_delta_res.py --desc-path /path/to/DESC_TrappedRes [--device cpu]
"""
import argparse
import glob
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--desc-path", required=True,
                    help="checkout of github.com/jlabbate15/DESC_TrappedRes")
parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
parser.add_argument("--outdir", default=None, help="default: ../desc_data next to this script")
parser.add_argument("--allow-version-mismatch", action="store_true",
                    help="proceed even if the DESC checkout is not at the pinned commit")
args = parser.parse_args()

# --- DESC version pin -------------------------------------------------------
DESC_COMMIT = "4b77719dfa44ef26df29ba80b9c620ecb19faaf4"
DESC_REPO = "https://github.com/jlabbate15/DESC_TrappedRes"
DESC_BRANCH = "ejp-merge-master"


def _check_desc_commit(path):
    """Verify the DESC checkout is at the pinned commit. See PROVENANCE.md."""
    import subprocess
    try:
        head = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        print(f"WARNING: {path} is not a git checkout; cannot verify the DESC "
              f"version.\n         Expected commit {DESC_COMMIT} "
              f"({DESC_REPO}, on branch {DESC_BRANCH}).", flush=True)
        return
    if head == DESC_COMMIT:
        print(f"[check] DESC at pinned commit {head[:12]} (branch {DESC_BRANCH}) OK",
              flush=True)
        return
    msg = (f"DESC VERSION MISMATCH\n"
           f"  found  : {head}\n"
           f"  pinned : {DESC_COMMIT}\n"
           f"  repo   : {DESC_REPO}  (commit is on branch {DESC_BRANCH})\n"
           f"  Check out the COMMIT, not the branch -- the branch tip has moved\n"
           f"  well past the pin and will not reproduce the packaged data.\n"
           f"    git -C {path} checkout {DESC_COMMIT}\n"
           f"  Override with --allow-version-mismatch if you know what you are doing.")
    if not args.allow_version_mismatch:
        raise SystemExit(msg)
    print("WARNING: " + msg, flush=True)


_check_desc_commit(args.desc_path)

sys.path.insert(0, args.desc_path)

import numpy as np
from desc import set_device

set_device(args.device)

from desc.equilibrium import Equilibrium
from desc.objectives import TrappedResonance

HERE = os.path.dirname(os.path.abspath(__file__))
EQ_DIR = os.path.join(HERE, "equilibria")
OUTDIR = args.outdir or os.path.join(HERE, os.pardir, "desc_data")

# Pitch grid: 20 points. This is the Bcrit axis of the orange curve.
Bcrit = np.linspace(4.5, 7.8, 20)


def beta_tag(path):
    """wout_..._desc.h5 -> '0';  wout_..._desc_0.004.h5 -> '0.004'."""
    stem = os.path.basename(path)[: -len(".h5")]
    tail = stem.rsplit("_desc", 1)[1]
    return tail.lstrip("_") or "0"


eq_paths = sorted(glob.glob(os.path.join(EQ_DIR, "*.h5")))
if not eq_paths:
    raise SystemExit(f"no .h5 equilibria found in {EQ_DIR}")

for eq_path in eq_paths:
    tag = beta_tag(eq_path)
    print(f"=== beta={tag} ===", flush=True)
    eq = Equilibrium.load(eq_path)

    obj = TrappedResonance(
        eq,
        num_rho=60,
        num_eta=20,
        M=0,
        N=1,
        pitch_invs=Bcrit,
        num_well=1,
        num_transit=4,
        knots_per_transit=50,
        num_quad=48,
        KE_frac=1,
        weight_method="bump",
        Delta_Omega=0.002,
        p_max=0,
        q_max=5,
        fill_value=11,
    )
    obj.build()
    val = obj.compute(eq.params_dict)

    out = os.path.join(OUTDIR, f"v3_beta_{tag}_desc_bump_nlam_20_delta_omega_0.002.npz")
    np.savez(out, val=val)
    print(f"saved {out}", flush=True)

print("DONE", flush=True)
