"""Run the full experiment matrix: backbone x distribution shift, one summary table.

    # see what would run (without executing anything)
    python scripts/run_matrix.py --dry-run

    # run every combination (backbones the configs know, shifts a subset of data.shifts)
    python scripts/run_matrix.py --backbones cldnn,resnet1d --shifts D1_snr,D2_cfo,D3_gain

    # also train any missing backbone checkpoints
    python scripts/run_matrix.py --train-missing

Artifacts
---------
results/matrix/matrix.csv        summary table (ready to paste into a report)
results/matrix/matrix.md         Markdown version
results/matrix/summary.json      full results

Why this script exists
----------------------
A single run has a very noisy rho (the 95% CI stays wide at n=10~16 layers). The
conclusions of this project **must rest on evidence accumulated over several
backbones x several shifts**, not on whether one particular rho is significant.
This script turns "accumulate evidence" into a single command.

(See docs/findings.md: the smoke test gives rho=+0.71, p=0.06, landing right in
  the "close to significant but not quite" band -- a single run is not enough to
  support a conclusion.)
"""

from __future__ import annotations

# --- console safety --------------------------------------------------------
# Chinese Windows consoles default to GBK. A character outside GBK raises
# UnicodeEncodeError on print and aborts the whole script. Setting
# errors="replace" degrades such a character to "?" instead of crashing.
import sys as _console_sys

for _stream in (_console_sys.stdout, _console_sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass
del _console_sys, _stream
# --- end console safety ----------------------------------------------------

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import discover_seed_roots, dump_json, find_result_file, load_config  # noqa: E402


def run(cmd: List[str], log: pathlib.Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {' '.join(cmd[1:])}")
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"    X exit code {proc.returncode}, log {log}")
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
        for line in tail:
            print(f"      | {line}")
    return proc.returncode


def collect_row(backbone: str, shift: str) -> Optional[Dict]:
    results_dir = ROOT / "results"
    rank_json = find_result_file(results_dir, "sensitivity",
                                 f"rank_{backbone}_{shift}.json")
    arm_json = find_result_file(results_dir, "equal_error",
                                f"arm_{backbone}_{shift}.json")
    if rank_json is None:
        return None
    rk = json.loads(rank_json.read_text(encoding="utf-8"))
    row: Dict[str, object] = {
        "backbone": backbone,
        "shift": shift,
        "n_layers": rk.get("n_layers"),
        "ref_bits": rk.get("reference_bits"),
        "acc_A": (rk.get("baseline") or {}).get("acc_A"),
        "acc_B": (rk.get("baseline") or {}).get("acc_B"),
        "loss_A": (rk.get("baseline") or {}).get("loss_A"),
        "loss_B": (rk.get("baseline") or {}).get("loss_B"),
    }
    rc = rk.get("rank_correlation", {})
    for key in ("loss", "acc"):
        d = rc.get(key) or {}
        row[f"rho_{key}"] = d.get("rho")
        row[f"p_{key}"] = d.get("p_value")
        row[f"ci_{key}"] = f"[{d.get('ci_low')}, {d.get('ci_high')}]"
    row["verdict_loss"] = (rc.get("loss") or {}).get("verdict", "")

    if arm_json is not None:
        arm = json.loads(arm_json.read_text(encoding="utf-8"))
        arms = arm.get("arms") or []
        if arms:
            pick = sorted(arms, key=lambda r: abs(r["budget_ratio"] - 0.6))[0]
            row["budget_ratio"] = pick["budget_ratio"]
            row["cost_of_ignorance"] = pick["cost_of_domain_ignorance"]
            row["rel_cost_of_ignorance"] = pick["relative_cost_of_ignorance"]
            row["gain_from_sensitivity"] = pick["gain_from_sensitivity"]
            row["ranking_flipped"] = pick["ranking_flipped"]
            row["verdict_arm"] = pick["verdict"]
    return row


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--backbones", default=None,
                    help="comma separated; defaults to cldnn,resnet1d")
    ap.add_argument("--shifts", default=None,
                    help="comma separated; defaults to every shift in the config")
    ap.add_argument("--train-missing", action="store_true",
                    help="train missing checkpoints first")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip combinations that already have results (for resuming; "
                         "safe to rerun this command after an interruption)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default="results/matrix")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backbones = (args.backbones.split(",") if args.backbones
                 else ["cldnn", "resnet1d"])
    all_shifts = list((cfg["data"].get("shifts") or {}).keys())
    shifts = (args.shifts.split(",") if args.shifts else all_shifts)

    py = sys.executable
    plan = [(b, s) for b in backbones for s in shifts]

    def sens_done(b: str, s: str) -> bool:
        return find_result_file(ROOT / "results", "sensitivity",
                                f"rank_{b}_{s}.json") is not None

    def arm_done(b: str, s: str) -> bool:
        return find_result_file(ROOT / "results", "equal_error",
                                f"arm_{b}_{s}.json") is not None

    print(f"[run_matrix] config={args.config}")
    print(f"  backbones : {backbones}")
    print(f"  shifts    : {shifts}")
    print(f"  combos    : {len(plan)}")
    if args.skip_existing:
        done = sum(1 for b, s in plan if sens_done(b, s) and arm_done(b, s))
        print(f"  --skip-existing: {done} complete, {len(plan) - done} to run")
    if args.dry_run:
        for b, s in plan:
            ck = run_root / "baseline" / f"{b}.pt"
            print(f"    {b:<10s} x {s:<14s}  "
                  f"ckpt {'present' if ck.exists() else 'missing'}  "
                  f"sens {'done' if sens_done(b, s) else 'to run'}  "
                  f"arm {'done' if arm_done(b, s) else 'to run'}")
        return 0

    results_dir = ROOT / "results"
    seed = int(cfg.get("seed", 42))
    run_root = results_dir / f"seed{seed}"
    print(f"  writing to {run_root.relative_to(ROOT)}")
    logs = ROOT / "experiments" / "logs"
    rc_all = 0

    # ---------------------------------------------------------------- training
    for b in backbones:
        ck = run_root / "baseline" / f"{b}.pt"
        if ck.exists():
            print(f"\n[train] {b}: checkpoint exists, skipping")
            continue
        if not args.train_missing:
            print(f"\n[train] {b}: missing checkpoint {ck}; "
                  "pass --train-missing to train it. Skipping.")
            rc_all = 1
            continue
        print(f"\n[train] {b}")
        rc_all |= run([py, "scripts/train_baseline.py", "--config", args.config,
                       "--backbone", b, "--outdir",
                       str((run_root / "baseline").relative_to(ROOT))],
                      logs / f"train_{b}.log")

    # ---------------------------------------------------------------- sensitivity
    print("\n[sensitivity]")
    for b, s in plan:
        if args.skip_existing and sens_done(b, s):
            print(f"    {b} x {s}: already done, skipping")
            continue
        rc = run([py, "scripts/run_sensitivity.py", "--config", args.config,
                  "--shift", s, "--backbone", b,
                  "--ckpt", str((run_root / "baseline" / f"{b}.pt").relative_to(ROOT)),
                  "--outdir", str((run_root / "sensitivity").relative_to(ROOT))],
                 logs / f"sens_{b}_{s}.log")
        rc_all |= rc

    # ---------------------------------------------------------------- equal error arm
    print("\n[equal_error]")
    for b, s in plan:
        if args.skip_existing and arm_done(b, s):
            print(f"    {b} x {s}: already done, skipping")
            continue
        rc = run([py, "scripts/run_equal_error.py", "--config", args.config,
                  "--shift", s, "--backbone", b,
                  "--sens", str((run_root / "sensitivity" / f"sens_{b}_{s}.npz").relative_to(ROOT)),
                  "--ckpt", str((run_root / "baseline" / f"{b}.pt").relative_to(ROOT)),
                  "--outdir", str((run_root / "equal_error").relative_to(ROOT))],
                 logs / f"arm_{b}_{s}.log")
        rc_all |= rc

    # ---------------------------------------------------------------- summary
    rows = [r for r in (collect_row(b, s) for b, s in plan) if r]
    if not rows:
        print("\nno results collected; check the logs above.")
        return 1

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    cols = ["backbone", "shift", "n_layers", "acc_A", "acc_B",
            "rho_loss", "p_loss", "ci_loss", "rho_acc", "p_acc",
            "cost_of_ignorance", "rel_cost_of_ignorance", "ranking_flipped"]

    import csv

    with (outdir / "matrix.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = ["# Experiment matrix summary", "",
             "> Auto-generated by `scripts/run_matrix.py`. rho is the Spearman rank"
             " correlation of the per-layer quantization sensitivity;",
             "> `cost_of_ignorance` = the extra predicted damage from allocating bit"
             " widths by source-domain sensitivity instead of target-domain"
             " sensitivity.", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")

    lines += ["", "## Conclusions", ""]
    for r in rows:
        lines.append(f"- **{r['backbone']} x {r['shift']}**: "
                     f"rho(loss)={fmt(r.get('rho_loss'))} "
                     f"(p={fmt(r.get('p_loss'))}); {r.get('verdict_loss', '')}")
        if r.get("verdict_arm"):
            lines.append(f"  - equal-error arm: {r['verdict_arm']}")

    (outdir / "matrix.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(outdir / "summary.json", {"config": args.config, "rows": rows})

    print("\n" + "=" * 78)
    print("  experiment matrix summary")
    print("=" * 78)
    print("  " + " | ".join(f"{c}" for c in
                           ["backbone", "shift", "n", "rho_loss", "p_loss",
                            "cost_ign", "rel_cost"]))
    for r in rows:
        print("  " + " | ".join([
            str(r["backbone"]), str(r["shift"]), str(r.get("n_layers", "-")),
            fmt(r.get("rho_loss")), fmt(r.get("p_loss")),
            fmt(r.get("cost_of_ignorance"), 6),
            fmt(r.get("rel_cost_of_ignorance"), 3),
        ]))
    print()
    print(f"  wrote {outdir / 'matrix.csv'}")
    print(f"  wrote {outdir / 'matrix.md'}")
    print(f"  wrote {outdir / 'summary.json'}")
    if rc_all:
        print("\n  [!] some sub-steps failed; see experiments/logs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
