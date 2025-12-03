#!/usr/bin/env python3
"""
EP coevolution driver: alternates
  1) evolving defenders,
  2) evolving attackers,
for N_ROUNDS rounds.
"""
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import subprocess
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent  # EA-in-Cyber
MAIN_METHODS_DIR = ROOT_DIR / "Main_methods"

DEFENDER_EP_SCRIPT = MAIN_METHODS_DIR / "ep_evolve_defenders.py"
ATTACKER_EP_SCRIPT = MAIN_METHODS_DIR / "ep_evolve_attackers.py"

N_ROUNDS = 5  # <- change to however many coevolution cycles you want


def run_script(path: Path):
    print(f"\n===== Running: {path.name} =====")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(path.parent),
        check=False
    )
    if result.returncode != 0:
        print(f"WARNING: {path.name} exited with code {result.returncode}")


def main():
    print(f"Root dir: {ROOT_DIR}")
    print(f"Defender EP script: {DEFENDER_EP_SCRIPT}")
    print(f"Attacker EP script: {ATTACKER_EP_SCRIPT}")
    print(f"Starting EP coevolution for {N_ROUNDS} rounds...\n")

    for r in range(1, N_ROUNDS + 1):
        print("\n" + "#" * 70)
        print(f"### EP COEVOLUTION ROUND {r}/{N_ROUNDS}")
        print("#" * 70)

        # 1) Defenders adapt to current best attackers
        run_script(DEFENDER_EP_SCRIPT)

        # 2) Attackers adapt to updated defender pool
        run_script(ATTACKER_EP_SCRIPT)

    print("\nEP coevolution loop complete.")


if __name__ == "__main__":
    main()
