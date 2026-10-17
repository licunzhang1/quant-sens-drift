"""Multi-seed experiment orchestration.

Why it is needed
----------------
A single seed only gives a point estimate. Reporting a "mean +/- std" requires
several random seeds. At 10~16 layers the sampling variance of rho is large to
begin with, so a conclusion without error bars does not hold up.

Directory convention
--------------------
    results/seed{SEED}/baseline/{backbone}.pt
    results/seed{SEED}/sensitivity/{rank,sens}_{backbone}_{shift}.json|npz
    results/seed{SEED}/equal_error/arm_{backbone}_{shift}.json

`--migrate-base-seed` moves the old flat layout
(`results/baseline`, `results/sensitivity`, `results/equal_error`) into
`results/seed{N}/` wholesale, with no need to rerun anything.

Usage
-----
    # see what would run first
    python scripts/run_seeds.py --seeds 43,44 --dry-run

    # run it (Ctrl+C any time; rerun with --skip-existing to resume)
    python scripts/run_seeds.py --seeds 43,44 --skip-existing

    # move existing results into the unified layout
    python scripts/run_seeds.py --migrate-base-seed 42
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

FLAT = ("baseline", "sensitivity", "equal_error")


def seed_root(seed: int) -> pathlib.Path:
    return ROOT / "results" / f"seed{seed}"


def run(cmd: List[str], log: pathlib.Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {' '.join(cmd[2:])}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"    X exit={p.returncode}  log={log.relative_to(ROOT)}", flush=True)
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
        for ln in tail:
            print(f"      | {ln}")
    return p.returncode


def migrate_base_seed(seed: int) -> int:
    """Move the old flat results/ layout into results/seed{N}/."""
    dst = seed_root(seed)
    moved = 0
    for name in FLAT:
        src = ROOT / "results" / name
        if not src.is_dir():
            continue
        tgt = dst / name
        if tgt.exists():
            print(f"  skipping {name}/: target already exists {tgt.relative_to(ROOT)}")
            continue
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(tgt))
        print(f"  {src.relative_to(ROOT)}  ->  {tgt.relative_to(ROOT)}")
        moved += 1
    print(f"  migration done: {moved} directories")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="multi-seed experiment orchestration")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", default="43,44", help="comma-separated list of seeds")
    ap.add_argument("--backbones", default="cldnn,resnet1d")
    ap.add_argument("--shifts", default=None,
                    help="defaults to every shift in the config")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--migrate-base-seed", type=int, default=None,
                    help="move the flat results/ layout into results/seed{N}/ and exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.migrate_base_seed is not None:
        return migrate_base_seed(int(args.migrate_base_seed))

    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    backbones = [b for b in args.backbones.split(",") if b.strip()]
    shifts = ([s for s in args.shifts.split(",") if s.strip()] if args.shifts
              else list((cfg["data"].get("shifts") or {}).keys()))

    py = sys.executable
    n_train = len(seeds) * len(backbones)
    n_sens = len(seeds) * len(backbones) * len(shifts)
    print(f"[run_seeds] config={args.config}")
    print(f"  seeds     : {seeds}")
    print(f"  backbones : {backbones}")
    print(f"  shifts    : {shifts}")
    print(f"  {n_train} training runs, {n_sens} sensitivity runs, "
          f"{n_sens} equal-error arms")

    total_min = n_train * 8 + n_sens * 2.2
    print(f"  estimated time: about {total_min:.0f} min "
          "(CPU, assuming the measured 8 min per training run)")
    print()

    if args.dry_run:
        for seed in seeds:
            root = seed_root(seed)
            print(f"  --- seed {seed}  ->  {root.relative_to(ROOT)}")
            for b in backbones:
                ck = root / "baseline" / f"{b}.pt"
                print(f"      train {b:<10} {'done' if ck.exists() else 'to run'}")
            for b in backbones:
                for s in shifts:
                    rk = root / "sensitivity" / f"rank_{b}_{s}.json"
                    ar = root / "equal_error" / f"arm_{b}_{s}.json"
                    print(f"      {b:<10} x {s:<15} sens={'.' if not rk.exists() else 'OK'}"
                          f"  arm={'.' if not ar.exists() else 'OK'}")
        return 0

    logs = ROOT / "experiments" / "logs"
    rc_all = 0
    t0 = time.time()
    done_units = 0
    total_units = n_train + 2 * n_sens

    for seed in seeds:
        root = seed_root(seed)
        print(f"\n{'=' * 72}\n  seed {seed}  ->  {root.relative_to(ROOT)}\n{'=' * 72}", flush=True)
        base_dir = root / "baseline"
        sens_dir = root / "sensitivity"
        arm_dir = root / "equal_error"

        # ---------------------------------------------------------- training
        for b in backbones:
            ck = base_dir / f"{b}.pt"
            if args.skip_existing and ck.exists():
                print(f"  [train] {b}: already done, skipping", flush=True)
                done_units += 1
                continue
            print(f"  [train] {b}  (seed={seed})", flush=True)
            rc_all |= run([py, "scripts/train_baseline.py", "--config", args.config,
                           "--backbone", b, "--seed", str(seed),
                           "--outdir", str(base_dir.relative_to(ROOT))],
                          logs / f"train_{b}_s{seed}.log")
            done_units += 1
            el = time.time() - t0
            print(f"    progress {done_units}/{total_units}  "
                  f"elapsed {el/60:.1f} min", flush=True)

        # ---------------------------------------------------------- sensitivity + equal-error arm
        for b in backbones:
            for s in shifts:
                rk = sens_dir / f"rank_{b}_{s}.json"
                ar = arm_dir / f"arm_{b}_{s}.json"
                ck = base_dir / f"{b}.pt"

                if args.skip_existing and rk.exists():
                    print(f"  [sens] {b} x {s}: already done, skipping", flush=True)
                else:
                    print(f"  [sens] {b} x {s}", flush=True)
                    rc_all |= run([py, "scripts/run_sensitivity.py",
                                   "--config", args.config, "--backbone", b,
                                   "--shift", s, "--seed", str(seed),
                                   "--ckpt", str(ck.relative_to(ROOT)),
                                   "--outdir", str(sens_dir.relative_to(ROOT))],
                                  logs / f"sens_{b}_{s}_s{seed}.log")
                done_units += 1
                el = time.time() - t0
                print(f"    progress {done_units}/{total_units}  "
                      f"elapsed {el/60:.1f} min", flush=True)

                if args.skip_existing and ar.exists():
                    print(f"  [arm ] {b} x {s}: already done, skipping", flush=True)
                else:
                    print(f"  [arm ] {b} x {s}", flush=True)
                    rc_all |= run([py, "scripts/run_equal_error.py",
                                   "--config", args.config, "--backbone", b,
                                   "--shift", s, "--seed", str(seed),
                                   "--sens", str((sens_dir / f"sens_{b}_{s}.npz").relative_to(ROOT)),
                                   "--ckpt", str(ck.relative_to(ROOT)),
                                   "--outdir", str(arm_dir.relative_to(ROOT))],
                                  logs / f"arm_{b}_{s}_s{seed}.log")
                done_units += 1
                el = time.time() - t0
                print(f"    progress {done_units}/{total_units}  "
                      f"elapsed {el/60:.1f} min", flush=True)

    print(f"\ndone, total elapsed {(time.time() - t0)/60:.1f} min")
    if rc_all:
        print("X some sub-steps failed; see experiments/logs/")
    print("\nnext: python scripts/aggregate_seeds.py")
    return 0 if rc_all == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
