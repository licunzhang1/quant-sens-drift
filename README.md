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

两个骨干、五个目标域（含一个阴性对照），源域 SNR (-2,6) dB，
n_symbols=128、sps=4、4 类调制。

| 骨干 | 层数 | 目标域 | Δacc | ρ(loss) | p | 域无知代价^1 | 相对代价 |
|---|---|---|---|---|---|---|---|
| CLDNN | 10 | D1 SNR (-8,0) | 0.322 | +0.673 | 0.037 | 0.0066 | +14% |
| CLDNN | 10 | D2 CFO | 0.557 | -0.394 | 0.258 | 0.0678 | +163% |
| CLDNN | 10 | D3 增益（对照） | -0.003 | +1.000 | 0.0001 | **0.0000** | 0% |
| CLDNN | 10 | D4 相位噪声 | 0.425 | +0.139 | 0.705 | 0.0304 | +166% |
| CLDNN | 10 | D5 IQ 失衡 | 0.013 | +0.103 | 0.785 | 0.0190 | +93% |
| ResNet1D | 16 | D1 SNR (-8,0) | 0.246 | +0.329 | 0.213 | 0.0269 | +54% |
| ResNet1D | 16 | D2 CFO | 0.523 | -0.200 | 0.456 | 0.8031 | +122% |
| ResNet1D | 16 | D3 增益（对照） | -0.006 | +0.579 | 0.021 | **0.0027** | +50% |
| ResNet1D | 16 | D4 相位噪声 | 0.355 | -0.179 | 0.501 | 0.4725 | +120% |
| ResNet1D | 16 | D5 IQ 失衡 | 0.041 | +0.635 | 0.010 | 0.0016 | +6% |

^1 域无知代价：在固定总重构误差预算下，「按源域敏感性分配位宽」相对
「按目标域敏感性分配位宽」在目标域上多付出的预测损伤。前者是现实中可实现的，
后者不可实现，构成上界。

**结论**

1. **不存在域无关的最优比特分配。** 阴性对照（无有效域偏移）上两种分配完全等价
   （代价 0.0000 / 0.0027）；一旦出现真实域偏移，代价在两个结构不同的骨干上
   一致为正，相对损伤最高 166%。
2. **ρ 本身不是可靠的结论载体。** 10 个组合中 7 个不显著，D4 甚至跨骨干方向相反
   （+0.139 vs -0.179）。此外 ρ 还强烈依赖于**参考位宽**：同一对域在 2 bit 上
   ρ=+0.867（p=0.002）而在 4 bit 上 ρ=+0.103（p=0.78）。
   因此结论应以 (骨干 × 偏移 × 参考位宽) 空间上的分布给出，而不是单个数值。

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
scripts/       训练、敏感性、等误差臂、矩阵、出图、进度查看
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
