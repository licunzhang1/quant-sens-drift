# quant-sens

Quantization sensitivity and domain shift: how far can per-layer sensitivity
rankings be transferred across deployment domains?

## The question

Post-training quantization (PTQ) calibration and mixed-precision search both rest
on an assumption:

> Quantization sensitivity is determined mainly by the architecture and the
> weight distribution, so a single representative calibration set (even random
> data) is enough to decide per-layer bit allocation.

This holds reasonably well for vision models. It has not been systematically
tested for signal-classification models under distribution shift.

This project defines

```
rho = Spearman( per-layer quant sensitivity ranks in source domain,
                per-layer quant sensitivity ranks in target domain )
```

and tests whether the assumption survives cross-domain deployment.

## Main results

Two backbones, five target domains (one of which is a null-shift control), and
**three random seeds** (42, 43, 44). Source domain SNR (-2, 6) dB,
`n_symbols=128`, `sps=4`, four modulation classes. Values are mean +/- std over seeds.

| Backbone | Layers | Target domain | rho(loss) mean+/-std | Seeds sig. | Fisher combined p | Cost of domain ignorance^1 | Positive seeds |
|---|---|---|---|---|---|---|---|
| CLDNN | 10 | D1 SNR (-8,0) | +0.778 +/- 0.094 | 3/3 | 0.0001 | 0.0069 +/- 0.0070 | 2/3 |
| CLDNN | 10 | D2 CFO | -0.576 +/- 0.219 | 1/3 | 0.0103 | 0.1240 +/- 0.0488 | 3/3 |
| CLDNN | 10 | D3 gain (control) | +0.802 +/- 0.273 | 2/3 | 0.0000 | **0.0039 +/- 0.0057** | 2/3 |
| CLDNN | 10 | D4 phase noise | -0.200 +/- 0.468 | 1/3 | 0.2085 | 0.0861 +/- 0.0821 | 3/3 |
| CLDNN | 10 | D5 IQ imbalance | -0.305 +/- 0.432 | 1/3 | 0.1031 | 0.0417 +/- 0.0317 | 3/3 |
| ResNet1D | 16 | D1 SNR (-8,0) | +0.311 +/- 0.349 | 1/3 | 0.0376 | 0.0313 +/- 0.0123 | 3/3 |
| ResNet1D | 16 | D2 CFO | -0.314 +/- 0.316 | 1/3 | 0.0479 | **0.6629 +/- 0.2092** | 3/3 |
| ResNet1D | 16 | D3 gain (control) | +0.590 +/- 0.234 | 2/3 | 0.0001 | **0.0015 +/- 0.0012** | 3/3 |
| ResNet1D | 16 | D4 phase noise | -0.312 +/- 0.347 | 1/3 | 0.0384 | 0.3785 +/- 0.1433 | 3/3 |
| ResNet1D | 16 | D5 IQ imbalance | +0.048 +/- 0.580 | 2/3 | 0.0156 | 0.0109 +/- 0.0126 | 3/3 |

^1 Cost of domain ignorance: under a fixed total reconstruction-error budget,
the extra predicted damage on the target domain incurred by allocating
bit-widths from **source-domain** sensitivities instead of **target-domain**
sensitivities. The former is what you can actually do; the latter is
unattainable and serves as an upper bound.

Per-seed detail: [results/multiseed/multiseed.md](results/multiseed/multiseed.md).

**Conclusions**

1. **No domain-agnostic optimal bit allocation exists.** 26 of 30
   (backbone x shift x seed) conditions have a positive cost of domain
   ignorance. The null-shift control has the *smallest* cost on both backbones
   (0.0039 and 0.0015), while real shifts reach **0.66 +/- 0.21**. A reference
   point that correctly goes to zero, and a quantity that is consistently
   positive under real shift -- this is the most solid result in the project.
2. **rho is not a reliable carrier for the conclusion, and its sign is not even
   stable.** Only 3 of 10 combinations are significant on all three seeds, and
   the sign flips across seeds (CLDNN x D5: +0.103 on one seed, **-0.305** over
   three; D4: +0.139 on one seed, **-0.200** over three). Half the combinations
   have a standard deviation at or above their mean (e.g. +0.048 +/- 0.580).
   rho also depends strongly on the **reference bit-width**: the same domain
   pair gives rho = +0.867 (p = 0.002) at 2 bit and rho = +0.103 (p = 0.78) at
   4 bit. rho is therefore a diagnostic, not a conclusion; results should be
   reported as a distribution over
   (backbone x shift x reference bit-width x seed).
3. **Fisher's method recovers effects that a single run cannot resolve.**
   CLDNN x D2 CFO has only 1 of 3 seeds individually significant, but
   p = 0.0103 combined.

Full experimental record, failure cases and methodology notes:
[docs/findings.md](docs/findings.md).

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

CPU is sufficient. Tested with Python 3.13.5, torch 2.14.1+cpu, scipy 1.18.1,
scikit-learn 1.9.1.

## Running

Start with the core checks that do not need torch (a few seconds):

```bash
python tests/run_all.py
```

Then the full experiment (two backbones x five target domains, about 90 minutes
on CPU):

```bash
python scripts/run_matrix.py --config configs/base.yaml --skip-existing
```

`--skip-existing` makes runs resumable: every intermediate result is written to
its own file, so an interrupted run only redoes what is missing.

To watch progress from another terminal (standard library only, no numpy/torch):

```bash
python scripts/watch_seeds.py
```

### Multiple seeds (error bars)

A single run of rho has a large variance, so only multiple seeds give you a
mean +/- std.

```bash
# Add further seeds (resumable); about 135 minutes for two seeds on CPU
python scripts/run_seeds.py --seeds 43,44 --skip-existing

# Watch progress from another terminal (standard library only)
python scripts/watch_seeds.py --seeds 43,44

# Aggregate: mean +/- std plus Fisher combined p across seeds
python scripts/aggregate_seeds.py
```

Outputs: `results/multiseed/multiseed.md`, `multiseed.csv`, `multiseed.json`.
See finding 10 in [docs/findings.md](docs/findings.md) for the results and for
the corrections it makes to the single-seed numbers.

## Implementation notes

- **Selective per-layer quantization.** `quantize_one_layer()` temporarily
  replaces one layer's weights (and optionally quantizes its activations) and
  restores them in a `finally` block, so an exception cannot leave the model in
  a quantized state. Fake quantization is written by hand rather than taken from
  `torch.quantization`, because the experiment needs exact control over
  per-channel vs per-tensor and over the weight/activation combination.
- **Loss is the primary sensitivity metric, not accuracy.** Accuracy saturates:
  once source-domain accuracy reaches 1.0, quantizing a layer to 2 bit can still
  leave it at 1.0, so delta-accuracy is identically zero and the rank
  correlation degenerates to NaN. The scripts compute both and print the
  difference.
- **rho always comes with a permutation test and a bootstrap interval.** With
  10 to 16 layers the sampling variance of Spearman rho is large; reporting rho
  without a p-value is not defensible.
- **A framework-independent analysis layer.** `src/quant/core.py`,
  `src/analysis/` and `src/data/{synth,features}.py` depend only on numpy and
  can be verified without torch. Torch-dependent modules are isolated behind
  lazy imports.

## Layout

```
configs/       experiment configuration (difficulty range calibrated empirically)
src/
  data/        synthetic IQ generator, classical AMC cumulant features, torch Dataset
  models/      CLDNN, 1D-ResNet, and a pure-numpy reference implementation
  quant/       quantization math (numpy, independently testable) and torch fake-quant
  analysis/    sensitivity matrix, rank correlation, equal-error arm, proxies
  offsets/     construction and management of distribution shifts
  viz/         report figures
scripts/       training, sensitivity, equal-error arm, matrix, seeds, aggregation,
               figures, progress viewer
tests/         unit tests, end-to-end smoke test, three diagnostic scripts
tools/         small maintenance utilities (encoding, verdict refresh)
results/       measured results, one directory per seed: results/seed{N}/
               (weights excluded, see .gitignore)
docs/          experimental record and methodology
```

## Data

The default is the built-in **synthetic IQ generator** (`src/data/synth.py`):
no download, and it gives exact control over the kind and strength of the shift.
That control is necessary here, because public datasets usually provide a single
fixed channel condition and cannot support a controlled shift experiment.

To use a public dataset instead (for example RML2018.01A), implement a new
`Dataset` and point `configs/base.yaml` at it. The analysis layer needs no
changes.

## Reproducing everything

```bash
python tests/run_all.py                                        # core math and pipeline
python scripts/train_baseline.py --backbone cldnn              # train (~8 min)
python scripts/train_baseline.py --backbone resnet1d
python scripts/run_matrix.py --skip-existing                   # sensitivity + arm + summary
python scripts/run_seeds.py --seeds 43,44 --skip-existing       # extra seeds
python scripts/aggregate_seeds.py                              # mean +/- std, Fisher p
python scripts/rank_vs_bits.py                                 # rho vs reference bit-width
python scripts/make_figures.py --backbone cldnn --shift D1_snr # figures
```

Weight files (`results/seed*/baseline/*.pt`) are not committed; regenerate them
with `train_baseline.py`. The intermediate results
(`results/seed*/sensitivity/*.npz`) *are* committed, so `run_equal_error.py` and
`rank_vs_bits.py` can be re-run without training anything.

## License

MIT, see [LICENSE](LICENSE).
