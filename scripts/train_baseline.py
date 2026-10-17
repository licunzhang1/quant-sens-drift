"""W3-W4: train the baseline backbone and produce per-domain accuracy tables.

    python scripts/train_baseline.py --config configs/base.yaml
    python scripts/train_baseline.py --config configs/base.yaml --backbone resnet1d

Output
------
``results/baseline/<backbone>_<domain>.json``  per-domain acc/loss
``results/baseline/<backbone>.pt``             weights

Notes
-----
* **Holding out a validation set for early stopping is mandatory**, otherwise the
  model trains down to the flat minimum at loss~=0 and the quantization sensitivity
  degenerates into noise (see [finding 2] in the header of tests/smoke_pipeline.py).
* Early stopping here uses the Domain A validation set; **no target-domain data may
  be used for any model selection**, or the cross-domain conclusions do not hold.
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
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    build_model,
    count_params,
    evaluate,
    load_config,
    make_domain,
    pick_device,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import dump_json, seed_everything  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--backbone", default=None, help="override model.name from the config")
    ap.add_argument("--outdir", default="results/baseline")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed from the config (used by the multi-seed runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backbone:
        cfg["model"]["name"] = args.backbone
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    tcfg = cfg["train"]
    dcfg = cfg["data"]
    device = pick_device(tcfg.get("device", "auto"))
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    name = cfg["model"]["name"]
    print(f"[train_baseline] backbone={name}  device={device}  seed={seed}")

    # ---------------------------------------------------------------- data
    n_tr = int(dcfg["n_train"])
    n_val = int(dcfg.get("n_val", 2000))
    t0 = time.time()
    Xtr, ytr = make_domain(cfg, "a", n_tr, seed)
    Xval, yval = make_domain(cfg, "a", n_val, seed + 1)
    # test domains: the source domain plus every shifted domain
    domains = {"A": (Xval, yval)}
    for key in (dcfg.get("shifts") or {}):
        domains[key] = make_domain(cfg, key, int(dcfg.get("n_test", 2000)), seed + 2)
    print(f"  data generated in {time.time() - t0:.1f}s   "
          f"train={Xtr.shape}  domains={list(domains)}")

    # infer the class count from the actual labels and override the config
    n_classes = int(len(np.unique(ytr)))
    model = build_model(cfg, n_classes=n_classes).to(device)
    print(f"  model {name}  params {count_params(model):,}  n_classes={n_classes}")

    # ---------------------------------------------------------------- training
    lr = float(tcfg.get("lr", 1e-3))
    wd = float(tcfg.get("weight_decay", 1e-4))
    epochs = int(tcfg.get("epochs", 40))
    bs = int(tcfg.get("batch_size", 256))
    patience = int(tcfg.get("early_stop_patience", 8))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss()

    xtr = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    n = len(ytr_t)
    gen = torch.Generator().manual_seed(seed)

    best_val, best_state, bad = float("inf"), None, 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        run = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = xtr[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run += float(loss.item()) * len(idx)
        sched.step()
        tr_loss = run / n
        val_acc, val_loss = evaluate(model, Xval, yval, device)
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_acc": val_acc, "val_loss": val_loss})
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:>3}  train_loss={tr_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep} "
                      f"(validation loss did not improve for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = outdir / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "n_classes": n_classes, "backbone": name}, ckpt)

    # ---------------------------------------------------------------- per-domain eval
    results = {}
    print("\n  per-domain metrics (loss is the primary metric)")
    print("    {:<12s}{:>10s}{:>12s}".format("domain", "acc", "loss"))
    for tag, (X, y) in domains.items():
        acc, loss = evaluate(model, X, y, device)
        results[tag] = {"acc": acc, "loss": loss, "n": int(len(y))}
        print("    {:<12s}{:>10.4f}{:>12.4f}".format(tag, acc, loss))

    a_acc = results["A"]["acc"]
    print(f"\n  in-domain baseline acc = {a_acc:.4f}")
    for tag, r in results.items():
        if tag != "A":
            print(f"  {tag}: acc drop {a_acc - r['acc']:+.4f}  "
                  f"loss rise {r['loss'] - results['A']['loss']:+.4f}")

    # ---------------------------------------------------------------- guardrails
    warnings: list[str] = []
    if a_acc > 0.99:
        warnings.append(
            f"source-domain accuracy {a_acc:.4f} > 0.99 -- **saturated**."
            " Quantization then barely changes the output, the per-layer sensitivity"
            " degenerates to roughly all zeros, and rho becomes noise"
            " (measured: 7 of 10 layers are zero)."
            " Lower the source-domain SNR, use fewer symbols, or add more classes."
            " See findings 2/4 in docs/findings.md."
        )
    if results["A"]["loss"] < 0.01:
        warnings.append(
            f"source-domain loss {results['A']['loss']:.4f} < 0.01 "
            "-- this also means saturation.")
    if a_acc < 0.80:
        warnings.append(
            f"source-domain accuracy {a_acc:.4f} < 0.80 -- the model is underfitting,"
            " so the sensitivity will be dominated by optimization noise."
            " Raise the source-domain SNR or train longer.")
    for tag, r in results.items():
        if tag == "A":
            continue
        drop = a_acc - r["acc"]
        if drop > a_acc - 0.30:
            warnings.append(
                f"{tag}: target-domain acc {r['acc']:.4f} is near chance level,"
                " so the sensitivity ranking will be noise. Weaken this shift.")
        if abs(drop) < 0.02:
            warnings.append(
                f"{tag}: domain gap is only {drop:+.4f} -- this is a **negative"
                " control** (no effective shift). The pipeline should correctly"
                " report 'no effect'; do not treat it as a meaningful shift arm.")
    if warnings:
        print("\n  " + "!" * 68)
        for w in warnings:
            print(f"  [!] {w}")
        print("  " + "!" * 68)

    dump_json(outdir / f"{name}_metrics.json", {
        "backbone": name, "seed": seed, "n_params": count_params(model),
        "n_classes": n_classes, "domains": results, "history": history,
        "checkpoint": str(ckpt),
    })
    print(f"\n  wrote {ckpt}")
    print(f"  wrote {outdir / f'{name}_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
