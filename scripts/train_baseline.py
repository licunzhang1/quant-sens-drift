"""W3–W4：训练 baseline 骨干，产出各域精度表。

    python scripts/train_baseline.py --config configs/base.yaml
    python scripts/train_baseline.py --config configs/base.yaml --backbone resnet1d

产出
----
``results/baseline/<backbone>_<domain>.json``  各域 acc/loss
``results/baseline/<backbone>.pt``             权重

注意
----
* **必须留验证集做早停**，否则模型会训练到 loss≈0 的平坦极小值，
  量化敏感性随之退化成噪声（见 tests/smoke_pipeline.py 文件头【发现 2】）。
* 这里用 Domain A 的验证集做早停；**不能用目标域的数据做任何模型选择**，
  否则跨域结论不成立。
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
    ap.add_argument("--backbone", default=None, help="覆盖 config 里的 model.name")
    ap.add_argument("--outdir", default="results/baseline")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config 里的 seed（多种子实验用）")
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

    # ---------------------------------------------------------------- 数据
    n_tr = int(dcfg["n_train"])
    n_val = int(dcfg.get("n_val", 2000))
    t0 = time.time()
    Xtr, ytr = make_domain(cfg, "a", n_tr, seed)
    Xval, yval = make_domain(cfg, "a", n_val, seed + 1)
    # 测试域：源域 + 所有偏移域
    domains = {"A": (Xval, yval)}
    for key in (dcfg.get("shifts") or {}):
        domains[key] = make_domain(cfg, key, int(dcfg.get("n_test", 2000)), seed + 2)
    print(f"  数据生成完成 {time.time() - t0:.1f}s   "
          f"train={Xtr.shape}  域={list(domains)}")

    # 类别数按实际标签推断并覆盖配置
    n_classes = int(len(np.unique(ytr)))
    model = build_model(cfg, n_classes=n_classes).to(device)
    print(f"  模型 {name}  参数量 {count_params(model):,}  n_classes={n_classes}")

    # ---------------------------------------------------------------- 训练
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
                print(f"  早停于第 {ep} 轮（验证损失连续 {patience} 轮未改善）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = outdir / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "n_classes": n_classes, "backbone": name}, ckpt)

    # ---------------------------------------------------------------- 各域评估
    results = {}
    print("\n  各域指标（loss 为主口径）")
    print("    {:<12s}{:>10s}{:>12s}".format("domain", "acc", "loss"))
    for tag, (X, y) in domains.items():
        acc, loss = evaluate(model, X, y, device)
        results[tag] = {"acc": acc, "loss": loss, "n": int(len(y))}
        print("    {:<12s}{:>10.4f}{:>12.4f}".format(tag, acc, loss))

    a_acc = results["A"]["acc"]
    print(f"\n  同域基线 acc = {a_acc:.4f}")
    for tag, r in results.items():
        if tag != "A":
            print(f"  {tag}: acc 掉点 {a_acc - r['acc']:+.4f}  "
                  f"loss 上升 {r['loss'] - results['A']['loss']:+.4f}")

    # ---------------------------------------------------------------- 护栏
    warnings: list[str] = []
    if a_acc > 0.99:
        warnings.append(
            f"源域准确率 {a_acc:.4f} > 0.99 —— **饱和**。"
            "此时量化几乎不改变输出，逐层敏感性会退化成近似全 0，"
            "ρ 变成噪声（实测：10 层里 7 层为 0）。"
            "请降低源域 SNR、减少符号数，或增加类别数。见 docs/findings.md 发现 2/4。"
        )
    if results["A"]["loss"] < 0.01:
        warnings.append(
            f"源域 loss {results['A']['loss']:.4f} < 0.01 —— 同样意味着饱和。")
    if a_acc < 0.80:
        warnings.append(
            f"源域准确率 {a_acc:.4f} < 0.80 —— 模型欠拟合，"
            "敏感性会被优化不足的噪声主导。请提高源域 SNR 或延长训练。")
    for tag, r in results.items():
        if tag == "A":
            continue
        drop = a_acc - r["acc"]
        if drop > a_acc - 0.30:
            warnings.append(
                f"{tag}: 目标域 acc {r['acc']:.4f} 接近随机水平，"
                "敏感性排序会变成噪声。请减弱该偏移强度。")
        if abs(drop) < 0.02:
            warnings.append(
                f"{tag}: 域间隙仅 {drop:+.4f} —— 这是一个**阴性对照**（无有效偏移）。"
                "管线应当正确报告'无影响'；不要把它当作有意义的偏移臂。")
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
    print(f"\n  已写入 {ckpt}")
    print(f"  已写入 {outdir / f'{name}_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
