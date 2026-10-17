# quant-sens

量化敏感性的域依赖性 —— 信号分类模型中层间量化误差与分布偏移的可分离性研究。

## 问题

训练后量化（PTQ）的校准集选择与混合精度搜索，普遍依赖一个前提：

> 量化敏感性主要由架构与权值分布决定，因此一个代表性校准集
> （甚至随机数据）就足以确定逐层比特分配。

这个前提在图像模型上大体成立，但在**信号模型 + 分布偏移**下没有被系统检验过。

本项目定义

```
ρ = Spearman( 源域逐层量化敏感性排序, 目标域逐层量化敏感性排序 )
```

并检验该前提在跨域部署场景下是否成立。

## 主要结果

两个骨干、五个目标域（含一个阴性对照）、**三个随机种子**（42 / 43 / 44）。
源域 SNR (-2,6) dB，n_symbols=128、sps=4、4 类调制。表中为跨种子的均值 ± 标准差。

| 骨干 | 层数 | 目标域 | ρ(loss) 均值±标准差 | 显著种子 | Fisher 合并 p | 域无知代价¹ 均值±标准差 | 为正的种子 |
|---|---|---|---|---|---|---|---|
| CLDNN | 10 | D1 SNR (-8,0) | +0.778 ± 0.094 | 3/3 | 0.0001 | 0.0069 ± 0.0070 | 2/3 |
| CLDNN | 10 | D2 CFO | -0.576 ± 0.219 | 1/3 | 0.0103 | 0.1240 ± 0.0488 | 3/3 |
| CLDNN | 10 | D3 增益（对照） | +0.802 ± 0.273 | 2/3 | 0.0000 | **0.0039 ± 0.0057** | 2/3 |
| CLDNN | 10 | D4 相位噪声 | -0.200 ± 0.468 | 1/3 | 0.2085 | 0.0861 ± 0.0821 | 3/3 |
| CLDNN | 10 | D5 IQ 失衡 | -0.305 ± 0.432 | 1/3 | 0.1031 | 0.0417 ± 0.0317 | 3/3 |
| ResNet1D | 16 | D1 SNR (-8,0) | +0.311 ± 0.349 | 1/3 | 0.0376 | 0.0313 ± 0.0123 | 3/3 |
| ResNet1D | 16 | D2 CFO | -0.314 ± 0.316 | 1/3 | 0.0479 | **0.6629 ± 0.2092** | 3/3 |
| ResNet1D | 16 | D3 增益（对照） | +0.590 ± 0.234 | 2/3 | 0.0001 | **0.0015 ± 0.0012** | 3/3 |
| ResNet1D | 16 | D4 相位噪声 | -0.312 ± 0.347 | 1/3 | 0.0384 | 0.3785 ± 0.1433 | 3/3 |
| ResNet1D | 16 | D5 IQ 失衡 | +0.048 ± 0.580 | 2/3 | 0.0156 | 0.0109 ± 0.0126 | 3/3 |

¹ 域无知代价：在固定总重构误差预算下，「按源域敏感性分配位宽」相对
「按目标域敏感性分配位宽」在目标域上多付出的预测损伤。前者是现实中可实现的，
后者不可实现，构成上界。

逐种子明细见 [results/multiseed/multiseed.md](results/multiseed/multiseed.md)。

**结论**

1. **不存在域无关的最优比特分配。** 30 个（骨干 × 偏移 × 种子）条件中有 **26 个**
   域无知代价为正。阴性对照的代价在两个骨干上都是最小的（0.0039 / 0.0015），
   真实偏移下最高达 **0.66 ± 0.21**。参照点正确归零、真实偏移下一致为正 ——
   这是本项目最稳的一条结论。
2. **ρ 不是可靠的结论载体，连符号都不稳定。** 10 个组合中只有 3 个做到 3/3 种子
   显著；更严重的是符号会随种子翻转（CLDNN × D5：单种子 +0.103，三种子
   **-0.305**；D4：单种子 +0.139，三种子 **-0.200**）。半数组合的标准差与
   均值同量级甚至更大（如 +0.048 ± 0.580）。此外 ρ 还强烈依赖**参考位宽**：
   同一对域在 2 bit 上 ρ=+0.867（p=0.002）而在 4 bit 上 ρ=+0.103（p=0.78）。
   因此 ρ 只能作为诊断量，结论应以 (骨干 × 偏移 × 参考位宽 × 种子) 上的
   分布给出。
3. **Fisher 合并能找回被单次实验掩盖的效应。** 例如 CLDNN × D2 CFO 单看只有
   1/3 种子显著（p 分别为 0.258 / …），合并后 p = 0.0103。

全部实验记录、失败案例与方法论细节见 [docs/findings.md](docs/findings.md)。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

CPU 即可。实测环境：Python 3.13.5、torch 2.14.1+cpu、scipy 1.18.1、scikit-learn 1.9.1。

## 运行

先跑不依赖 torch 的核心验证（几秒）：

```bash
python tests/run_all.py
```

再跑完整实验（两个骨干 × 五个目标域，纯 CPU 约 90 分钟）：

```bash
python scripts/run_matrix.py --config configs/base.yaml --skip-existing
```

`--skip-existing` 支持断点续跑：所有中间结果各自落盘，中断后重跑只补缺的部分。

另开一个终端查看进度（仅用标准库，不需要 numpy/torch）：

```bash
python scripts/watch_progress.py
```

### 多种子（给出误差棒）

单次运行的 ρ 方差很大，只有多种子才能给出"均值 ± 标准差"。

```bash
# 补跑额外种子（可断点续跑）；两个种子约 135 分钟（CPU）
python scripts/run_seeds.py --seeds 43,44 --skip-existing

# 跑的时候实时看进度（另开一个终端，纯标准库）
python scripts/watch_seeds.py --seeds 43,44

# 聚合：均值±标准差 + 跨种子 Fisher 合并 p
python scripts/aggregate_seeds.py
```

产物：`results/multiseed/multiseed.md`、`multiseed.csv`、`multiseed.json`。
结果与订正说明见 [docs/findings.md](docs/findings.md) 发现 10。

## 关键实现

- **逐层选择性量化**：`quantize_one_layer()` 临时替换单层权重（可选量化其激活），
  在 `finally` 中恢复，异常也不会污染模型。手写 fake-quant 而非使用
  `torch.quantization`，以精确控制 per-channel/per-tensor 与权重/激活的组合。
- **敏感性主口径为 loss**：准确率会饱和（源域 acc 达 1.0 时，把某层量化到 2 bit
  仍可能是 1.0），导致 Δacc ≡ 0、秩相关退化为 NaN。脚本同时输出两套口径并提示差异。
- **ρ 配置换检验与 bootstrap 区间**：层数为 10~16 时 Spearman ρ 的抽样方差很大，
  只报 ρ 不报 p 值不严谨。
- **框架无关的分析层**：`src/quant/core.py`、`src/analysis/`、`src/data/{synth,features}.py`
  仅依赖 numpy，可独立于 torch 验证。torch 相关模块通过惰性导入隔离。

## 目录结构

```
configs/       实验配置（难度区间已按实测标定）
src/
  data/        合成 IQ 生成、经典 AMC 累积量特征、torch Dataset
  models/      CLDNN、1D-ResNet，以及一个纯 numpy 参考实现
  quant/       量化数学（numpy，可独立测试）与 torch 侧 fake-quant
  analysis/    敏感性矩阵、秩相关、等重构误差臂、代理量
  offsets/     分布偏移的构造与管理
  viz/         报告图表
scripts/       训练、敏感性、等误差臂、矩阵、多种子、聚合、出图、进度查看
tests/         单元测试、端到端冒烟测试、三个诊断脚本
results/       实测结果（权重除外，见 .gitignore）
docs/          实验记录与方法论
```

## 数据

默认使用内置**合成 IQ 生成器**（`src/data/synth.py`）：零下载，且能精确控制
偏移的种类与强度 —— 这是必需的，因为公开数据集通常只提供单一固定信道条件，
无法支撑受控的偏移实验。

若要接入公开数据集（如 RML2018.01A），实现一个新的 `Dataset` 并让
`configs/base.yaml` 指向它即可，分析层无需改动。

## 复现全部结果

```bash
python tests/run_all.py                                        # 核心数学与管线
python scripts/train_baseline.py --backbone cldnn               # 训练（约 8 分钟）
python scripts/train_baseline.py --backbone resnet1d
python scripts/run_matrix.py --skip-existing                    # 敏感性 + 等误差臂 + 汇总
python scripts/rank_vs_bits.py                                  # ρ 的参考位宽依赖性
python scripts/make_figures.py --backbone cldnn --shift D1_snr  # 出图
```

权重文件（`results/baseline/*.pt`）不入库，用上面的 `train_baseline.py` 重新生成即可。
中间结果（`results/sensitivity/*.npz`）已入库，因此 `run_equal_error.py`
与 `rank_vs_bits.py` 可在不训练的情况下直接重跑。

## 许可

MIT，见 [LICENSE](LICENSE)。
