#!/usr/bin/env python3
"""
Wrapper script to run fusion_distribution_classification.py for all wout*.nc files
in the same directory.
"""
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WOUT_DIR = SCRIPT_DIR  # wout files and fusion_distribution_classification.py are in the same directory
STATS_SCRIPT = SCRIPT_DIR / "fusion_distribution_classification.py"
N_TASKS = os.environ.get("SCAN_NTASKS") or os.environ.get("SLURM_NTASKS", "128")


def main():
    wout_files = sorted(WOUT_DIR.glob("wout*.nc"))
    if not wout_files:
        print(f"No wout*.nc files found in {WOUT_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wout_files)} wout files")
    for wout_path in wout_files:
        print(f"\n--- Running fusion_distribution_classification.py for {wout_path.name} ---")
        launch_cmd = [
            "srun",
            "--ntasks",
            N_TASKS,
            "--cpus-per-task",
            "1",
            sys.executable,
            "-u",
            str(STATS_SCRIPT),
            str(wout_path.name),
        ]
        result = subprocess.run(
            launch_cmd,
            cwd=str(WOUT_DIR),
        )
        if result.returncode != 0:
            print(f"WARNING: {wout_path.name} failed with exit code {result.returncode}", file=sys.stderr)
        else:
            print(f"Completed {wout_path.name}")

    print("\n--- All runs finished ---")


if __name__ == "__main__":
    main()
