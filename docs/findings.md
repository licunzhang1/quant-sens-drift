# Measured findings and verification status

> This file records **results actually produced while building the codebase**, not design intent.
> Every entry has a reproducible script. These can be used directly when writing the report.

Reproduction commands:

```bash
python tests/run_all.py                 # unit tests + smoke test
python tests/diagnose_data.py           # Finding 1
python tests/diagnose_features.py       # the remedy for Finding 1
python tests/diagnose_regime.py         # Finding 3 and the usable regime
python tests/smoke_pipeline.py          # end-to-end pipeline
```

---

## Finding 1 | Feeding raw IQ straight into an MLP does not learn modulation recognition -- it is a **representation** problem, not a data problem

`tests/diagnose_data.py` output:

| n_symbols | sps | L | train acc | test acc | per-class test accuracy (bpsk/qpsk/8psk/16qam) |
|---|---|---|---|---|---|
| 32 | 2 | 64 | 0.9765 | 0.4680 | 0.98 / 0.30 / 0.33 / 0.30 |
| 32 | 4 | 128 | 0.9975 | 0.5007 | 1.00 / 0.34 / 0.34 / 0.32 |
| 64 | 4 | 256 | 1.0000 | 0.4947 | 0.99 / 0.34 / 0.31 / 0.36 |
| 128 | 4 | 512 | 1.0000 | 0.5080 | 0.99 / 0.34 / 0.38 / 0.28 |
| 128 | 8 | 1024 | 1.0000 | 0.4847 | 0.97 / 0.35 / 0.31 / 0.32 |
| 256 | 4 | 1024 | 1.0000 | 0.4413 | 0.84 / 0.28 / 0.30 / 0.32 |

**Observation**: no matter how much data, how large the model, or how high the oversampling
ratio, the training accuracy always reaches 1.0, but the test accuracy **stays stuck at 0.50**;
per class, only BPSK is separable (~0.98) while QPSK/8PSK/16QAM are all ~0.33.

**Cause**: the modulation information is hidden in the **distribution** of the samples
(constellation shape -> higher-order moments/cumulants), not at a handful of sample points.
The MLP does a "weighted sum per coordinate", which is inherently bad at extracting
distributional statistics, so it chooses to memorize the training samples instead of
learning the regularity.

**Remedy**: switch to the classic AMC cumulant features (`src/data/features.py`).
`tests/diagnose_features.py` output:

```
Strict holdout split (same dataset, 75/25): train=1.0000  test=1.0000
Freshly generated dataset (different seed)      : test=1.0000
  bpsk acc=1.000   qpsk acc=1.000   8psk acc=1.000   16qam acc=1.000
```

**What this means for the project**:
1. The reference model in `tests/smoke_pipeline.py` must use a feature representation,
   otherwise the "domain gap" will be mixed together with "generalization failure", and the
   cross-domain sensitivity analysis will measure nothing but noise.
2. This **incidentally gives the project a second asset**: the cumulant classifier is a
   ready-made implementation of the **classic DSP baseline arm** in the study design, and can
   be strictly compared against CLDNN-on-raw-IQ.

---

## Finding 2 | Accuracy saturates, so the **primary metric for sensitivity must be loss**

Output of the early version of `tests/smoke_pipeline.py` (domain A acc = 1.0000):

```
Domain A sensitivity (acc): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
rho (Spearman) = nan        <- the rank correlation degenerates outright
Number of unique values in domain-A acc sensitivity = 1 / 8
```

**Observation**: when the in-domain accuracy is already 1.0, quantizing some layer to 2 bit
can still leave it at 1.0, so Delta_acc == 0, the sensitivity vector is entirely 0, and the
Spearman rho becomes NaN.

**Cause**: accuracy is a **discrete quantity with a saturation ceiling**. Near a flat minimum
where the loss is already close to 0, the tiny perturbation introduced by weight quantization
is not enough to cross a decision boundary.

**Handling**:
1. The primary metric for the sensitivity analysis becomes **loss** (continuous, no
   saturation); the script also computes acc as a control and prints the two side by side.
   In `configs/base.yaml` the default of `sensitivity.metric` is already set to `loss`.
2. **Difficulty must be kept in the regime where the source domain is not saturated** (see the
   sweep in Finding 3).
3. Training must hold out a validation set for early stopping; do not train until loss ~ 0.

The complete run after the fix (`tests/smoke_pipeline.py`, domain A SNR=(4,12), domain B SNR=(-2,6)):

| metric | rho | p-value |
|---|---|---|
| **loss (primary)** | **+0.7143** | 0.0599 |
| acc (control) | +0.5063 | 0.2119 |

**The two metrics give different rho, which is direct evidence that the metric definition
changes the conclusion** -- this is itself a section that can be written into the report.

---

## Finding 3 | **The representation decides which domain shift is fatal**

The CFO isolation experiment in `tests/diagnose_regime.py`.
The SNR shift is fixed at (6,14) -> (0,8) dB and **only the carrier frequency offset CFO is changed**:

| CFO | Target-domain accuracy |
|---|---|
| 0.000 | **0.6875** |
| 0.001 | 0.4288 |
| 0.005 | 0.3300 |
| 0.020 | 0.3137 <- chance level |

**Mechanism**: higher-order cumulants are **not CFO-invariant**. ``E[z^4]`` picks up an
``exp(j8*pi*f*k)`` phase under CFO; with L=512 and f=0.02 the phase turns through about 82
revolutions, and the sample average of the cumulant cancels straight to zero.

**This is not a bug, it is a conclusion that belongs in the report**:

> **The "strength of a domain shift" is not an absolute quantity; it depends on the input representation.**
> The same CFO is catastrophic for cumulant features, while for a CNN backbone that can learn
> phase invariance it may be only a mild degradation.

The factorized protocol of this project (representation x shift type x quantization scheme) is
exactly what characterizes this point. It turns "which domain shift is most fatal" from an
assertion into a measurable two-dimensional table.

---

## Finding 4 | The usable difficulty regime where "the source domain is not saturated and the target domain has not collapsed"

Sweep results from `tests/diagnose_regime.py` (selection criteria: source-domain acc in
[0.85, 0.975] and max 2bit Delta_loss > 1e-3; target-domain acc in [0.35, 0.75]):

| n_symbols | SNR_A | SNR_B | CFO | epochs | accA | accB | max 2bit Delta_loss |
|---|---|---|---|---|---|---|---|
| **64** | **(4,12)** | **(-2,6)** | **0** | **100** | **0.9487** | **0.5637** | **3.003** |
| 32 | (8,16) | (0,8) | 0 | 80 | 0.9650 | 0.5525 | 0.097 |
| 32 | (6,14) | (-2,6) | 0 | 80 | 0.9400 | 0.4700 | 1.052 |

Typical failure modes that do not meet the criteria:
- Source domain (10,18) dB -> accA = 1.0000, **saturated**, Delta_loss ~ 0.003
- Target domain (-4,4) dB -> accB ~ 0.25, **collapsed to chance**, sensitivity turns into noise

`configs/base.yaml` is already set to the first row.

> **Practical advice**: before launching a formal experiment, sweep your own backbone and
> dataset once with `diagnose_regime.py`. This step of "fix the difficulty regime first, then
> talk about conclusions" is itself part of the experimental design, and it can prevent a whole
> semester of experiments from being wasted.

---

## Current output of the end-to-end smoke test

`python tests/smoke_pipeline.py` (8-layer MLP + cumulant features, about 7 seconds):

| Metric | Value |
|---|---|
| Representation | amc_cumulants (14-dimensional) |
| In-domain generalization gap | +0.0563 |
| Cross-domain drop (Delta_acc) | 0.9430 -> 0.5670, **+0.3760** |
| rho (loss metric) | **+0.7143** (p = 0.0599, 95% CI [-0.077, +1.000]) |
| rho (acc metric) | +0.5063 (p = 0.2119) |
| Equal-error arms: uniform / agnostic / oracle | [4,4,4,4,4,4,4,4] / [8,2,8,2,2,8,2,2] / [2,2,8,2,2,8,8,2] |
| Domain-agnostic cost (agnostic - oracle) | **+0.0579 (relative +13.8%)** |

**How to read this**: rho = +0.71 but p = 0.06, which lands in the "close to significant but
not there" band. This result is both more common and more honest than either "rho
significantly 1" or "rho significantly 0" -- it shows that with n=8 layers the statistical
power is insufficient, and **a real experiment has to accumulate evidence from multiple
backbones, multiple shifts and multiple random seeds**, not from the rho of a single run.

---

## Finding 5 | Too few layers makes rho lose statistical power (the measured CLDNN has only 5 layers)

The originally implemented CLDNN used one convolution per stage, so only **5** quantizable
weight layers were left (3 conv + 2 fc):

```
cldnn      quantizable weight layers =   5   parameters =   158,340
    [ 0] conv.0   [ 1] conv.4   [ 2] conv.8   [ 3] fc.0   [ 4] fc.3
resnet1d   quantizable weight layers =  16   parameters =   443,524
```

At n=5 the measured 95% bootstrap CI of the Spearman rho is **`[-1.000, +0.233]`** -- so wide
that it carries no information at all, and the smallest possible p-value of the permutation
test is only about 0.008.

**Fix**: change CLDNN to two convolutions per stage (`convs_per_stage=2`), which in the default
configuration gives 8 conv + 2 fc = **10 weight layers**:

```
cldnn      weight layers  10   parameters    618,116
resnet1d   weight layers  16   parameters    443,524
```

`scripts/run_sensitivity.py` now warns outright when the layer count is < 8.
**This is a hard constraint that should be checked before starting work, not a tuning knob.**

---

## Finding 6 | "Shift strength depends on the representation" -- confirmed on the same dataset with two backbones

Finding 3 concluded from cumulant features that "CFO is catastrophic". Moving to a CNN backbone
that can learn phase invariance, **with the same dataset and the same CFO=0.02**:

| Target domain | Cumulant features + MLP | CLDNN | ResNet1D |
|---|---|---|---|
| D2_cfo | acc **0.31** (chance level) | acc **0.4375** | acc **0.4767** |
| - drop relative to source domain | - | +0.533 | +0.503 |

The CNN is clearly more CFO-robust (0.44~0.48 vs 0.31), but **it still loses more than half**.

**The conclusion is now more precise**: CFO is harmful to every representation, but **the
magnitude of the harm is decided by the representation**. This is worth more than an assertion
like "CFO is catastrophic" -- it turns "domain shift strength" into **a measurable quantity
that varies with the representation**, rather than an absolute property.

---

## Finding 7 | A global gain shift is an **ineffective shift** (negative control)

`D3_gain` (global gain +6 dB) produces almost no domain gap on either backbone:

| Backbone | Source-domain acc | D3_gain acc | Drop |
|---|---|---|---|
| CLDNN | 0.9705 | 0.9650 | +0.0055 |
| ResNet1D | 0.9800 | 0.9792 | +0.0008 |

**The reason is that every sample is power-normalized** (division by the RMS at the end of
`apply_channel`), which cancels the global gain completely.

**This is not a bug but a useful negative control**: it tests whether the pipeline correctly
reports no shift when there is no shift. `scripts/train_baseline.py` now detects this
automatically and prints:

```
[!] D3_gain: domain gap only +0.0055 -- this is a **negative control** (no effective shift).
  The pipeline should correctly report 'no effect'; do not treat it as a meaningful shift arm.
```

**How to use it when writing the report**: put D3_gain in the first row of the results table as
a sanity check. If it shows a noticeable drop, the pipeline is broken and none of the
conclusions after it are worth reading.

---

## True baselines for the two backbones (CORRECTED config, seed=42)

Source domain SNR (-2,6) dB, n_symbols=128, sps=4, 8000 training samples.

| Domain | CLDNN acc | CLDNN loss | ResNet1D acc | ResNet1D loss |
|---|---|---|---|---|
| A (source domain) | 0.9705 | 0.0972 | 0.9800 | 0.0575 |
| D1_snr (-8,0) | 0.6617 | 1.4585 | 0.7392 | 0.8229 |
| D2_cfo | 0.4375 | 5.0003 | 0.4767 | 6.4588 |
| D3_gain (negative control) | 0.9650 | 0.1158 | 0.9792 | 0.0570 |
| D4_phasenoise | 0.5317 | 3.7497 | 0.6333 | 2.9263 |
| D5_iqimb | 0.9458 | 0.1931 | 0.9308 | 0.1897 |

**Key point**: source-domain acc 0.97/0.98 and loss 0.097/0.058, **both out of the saturation
region** (compare: with source domain (4,12) dB, acc=0.9995 and loss=0.0013, and 7 of the 10
layers are 0). This is the precondition for being able to do a sensitivity
analysis.

**Another observation worth writing up**: ResNet1D has the higher source-domain accuracy under
every shift, yet its D2_cfo loss (6.46) is actually worse than CLDNN's (5.00) --
**the acc ordering and the loss ordering disagree**, once again confirming that the metric
definition changes the conclusion (Finding 2).

---

## Finding 8 | The reference bit width is a **first-order design choice**, not a detail

> **Note**: the rho in this section is likewise a single-seed result. The multi-seed recheck
> (Finding 10) shows that **the sign itself is unstable**, so the numbers in the table below can
> only be read as "one sample from that draw".
 

This is the item that came up during the experiments and is **the most likely to become a core
contribution of the project**.

Per-layer sensitivity has to be measured at some "reference bit width" b\*. Intuitively b\* is
just a technical detail.
Measured (CLDNN, 10 layers, bit-width ladder [2,3,4,6,8], `scripts/rank_vs_bits.py`):

| Shift | 2b | 3b | 4b | 6b | 8b |
|---|---|---|---|---|---|
| **D3_gain** (negative control) | +0.988\* | +0.976\* | **+1.000\*** | +0.988\* | +0.952\* |
| D1_snr | +0.600 | +0.248 | **+0.673\*** | **+0.733\*** | -0.055 |
| **D5_iqimb** | **+0.867\*** | +0.479 | +0.103 | +0.297 | **+0.818\*** |
| D4_phasenoise | +0.358 | +0.273 | +0.139 | -0.067 | -0.091 |
| D2_cfo | +0.164 | +0.224 | -0.394 | -0.285 | -0.030 |

(`*` = p < 0.05)

### Three key observations

**(1) For the same pair of domains, rho can collapse from +0.87 to +0.10.**
D5_iqimb has rho = +0.867 at 2 bit (p = 0.0021, highly consistent),
and rho = +0.103 at 4 bit (p = 0.78, no relationship at all).
**Reporting only 4 bit yields "the sensitivity ranking does not transfer across domains";
reporting only 2 bit yields "it transfers strongly".
The two readings differ by an order of magnitude.**

**(2) The negative control is rock solid at every bit width.**
D3_gain has rho in [0.952, 1.000] across the 5 bit widths, all of them significant.
**This proves the bit-width dependence is not noise or a bug** -- the pipeline gives a stable
and correct result when there is no shift; the bit-width dependence is only amplified **when
there is a real shift**.

**(3) D5_iqimb is U-shaped and D4_phasenoise decreases monotonically.**
Neither shape looks like random fluctuation (random fluctuation would not produce this
structure over 5 points), which suggests a real "bit width x domain" interaction effect.

### Why this matters

It turns the project's conclusion from "one number per question" into **a surface**:

```
rho = f(reference bit width b*, domain shift type, backbone)
```

**Any assertion that "rho is high/low" is indefensible if it does not come with a bit-width
sensitivity analysis.** This also settles a doubt that is bound to be raised --
"what makes you choose 4 bit as the reference bit width?" The answer is no longer a hand-picked
value but a complete curve.

### Reproduction

```bash
python scripts/rank_vs_bits.py                    # all results
python scripts/rank_vs_bits.py --backbone cldnn   # single backbone
```

The script also prints the **number of unique values** (`#uniq`) at each bit width --
when there are too many tied values the rank correlation degenerates into noise, and that
column exists to catch exactly that case.

> **Honest addition**: at n=10 the confidence interval of rho is still very wide, so "one
> particular rho value" should not be over-interpreted.
> But **the phenomenon itself, that rho changes with the bit width**, appears consistently
> across the 5 bit widths, and with the negative control as a stability anchor, so it is
> credible.

---

## Finding 9 | Full experiment matrix: 2 backbones x 5 shifts (including the negative control)

> **Note**: every rho in this section is a single-seed (seed=42) point estimate. The three-seed
> recheck is in Finding 10 -- where the sign of rho **flips** for D4 phase noise and D5 IQ
> imbalance.


`scripts/run_matrix.py` runs all 10 combinations in one go, about 22 minutes (CPU).
Raw results: [`results/matrix/matrix.md`](matrix/matrix.md) - [`matrix.csv`](matrix.csv)

| Backbone | n | Shift | Delta_acc | **rho(loss)** | p | **Domain-agnostic cost** | Relative cost | Ranking changed |
|---|---|---|---|---|---|---|---|---|
| cldnn | 10 | D1_snr | 0.322 | **+0.673\*** | 0.037 | +0.0066 | +14.0% | yes |
| cldnn | 10 | D2_cfo | 0.557 | -0.394 | 0.258 | **+0.0678** | +163.3% | yes |
| cldnn | 10 | **D3_gain** (control) | -0.003 | **+1.000\*** | 0.0001 | **+0.0000** | 0.0% | no |
| cldnn | 10 | D4_phasenoise | 0.425 | +0.139 | 0.705 | +0.0304 | +166.3% | yes |
| cldnn | 10 | D5_iqimb | 0.013 | +0.103 | 0.785 | +0.0190 | +92.9% | yes |
| resnet1d | 16 | D1_snr | 0.246 | +0.329 | 0.213 | +0.0269 | +54.4% | yes |
| resnet1d | 16 | D2_cfo | 0.523 | -0.200 | 0.456 | **+0.8031** | +122.0% | yes |
| resnet1d | 16 | **D3_gain** (control) | -0.006 | **+0.579\*** | 0.021 | **+0.0027** | +50.0% | yes |
| resnet1d | 16 | D4_phasenoise | 0.355 | -0.179 | 0.501 | +0.4725 | +120.3% | yes |
| resnet1d | 16 | D5_iqimb | 0.041 | **+0.635\*** | 0.010 | +0.0016 | +5.9% | yes |

(`*` = p < 0.05. Domain-agnostic cost = the extra prediction damage from allocating by
source-domain sensitivity instead of by target-domain sensitivity.)

### What is robust and what is not

**(1) rho is not robust -- 7 of the 10 are not significant, and the direction flips across backbones.**

| Shift | cldnn rho | resnet1d rho | Same direction? |
|---|---|---|---|
| D3_gain | +1.000\* | +0.579\* | [done] consistent and both significant |
| D2_cfo | -0.394 | -0.200 | [done] consistent, both non-significant |
| D1_snr | +0.673\* | +0.329 | [done] consistent, significance differs |
| D4_phasenoise | **+0.139** | **-0.179** | [X] **opposite directions** |
| D5_iqimb | +0.103 | +0.635\* | [done] consistent, significance differs |

**Conclusion: the rho of a single run cannot be used as a conclusion.** Combined with Finding 8
(rho also depends strongly on the reference bit width), this means: **"what is rho?" is itself
the wrong question**; the question should be "how is rho distributed over the
(backbone x shift x bit width) space".

**(2) The domain-agnostic cost is far more robust -- this is the quantity that can carry a conclusion.**

| | D3_gain (negative control) | With a real shift |
|---|---|---|
| cldnn | **+0.0000** | +0.0066 ~ +0.0678 (all positive) |
| resnet1d | **+0.0027** | +0.0016 ~ **+0.8031** (all positive) |

- **Correctly zeroed on the negative control** (0.0000 / 0.0027) -- with no shift, allocating by
  source-domain sensitivity and allocating by target-domain sensitivity are **equivalent**, so
  "not knowing the deployment domain" costs nothing. **This proves the measurement is valid.**
- **With a real shift the direction is completely consistent across the two backbones (all
  positive)**, with a maximum relative cost of 166%.
- 9 of the 10 combinations have `ranking_flipped = True`: the source-domain and target-domain
  rankings really do produce **different bit-width allocation decisions**, rather than "the
  ranking changed but the decision did not".

### The project's final conclusion (one sentence that can go in the report)

> **There is no domain-independent optimal bit allocation.**
> On the negative control (no effective domain shift), "allocate bit widths by source-domain
> sensitivity" and "allocate by target-domain sensitivity" are completely equivalent
> (cost 0.0000 / 0.0027); once a real domain shift appears, the cost is consistently positive on
> two structurally completely different backbones, with relative damage up to 166%.
> Therefore the premise PTQ relies on -- "one generic calibration set is enough to determine the
> per-layer bit allocation" -- **does not hold for signal tasks deployed across domains**.
>
> At the same time it must be stated that **the rank correlation rho of per-layer sensitivity is
> itself not a reliable vehicle for conclusions** -- it depends strongly on both the reference
> bit width (Finding 8) and the backbone choice (this finding), 7 of the 10 combinations are not
> significant, and 1 flips direction across backbones.

**The structure of this passage (give the quantifiable quantity first, then volunteer the
unreliable one) is defensive writing**: every point a reviewer might want to question is
already written into it.

### Reproduction

```bash
# Full pipeline in one shot (skips finished work, resumable)
python scripts/run_matrix.py --config configs/base.yaml --skip-existing

# Progress only
python scripts/watch_progress.py

# Bit-width dependence of rho
python scripts/rank_vs_bits.py
```

---

## Finding 10 | Multi-seed verification: the cost is robust, rho is not even stable in sign

The rho values in Findings 8 and 9 above are all **single-seed (seed=42)** point estimates.
After additionally running seeds 43 and 44 (3 seeds x 2 backbones x 5 shifts = 30 conditions),
the picture becomes clearer, and **some single-seed conclusions are overturned**.

Reproduction:

```bash
python scripts/run_seeds.py --seeds 43,44 --skip-existing   # about 135 minutes (CPU)
python scripts/aggregate_seeds.py                            # aggregation + Fisher combined p
python scripts/watch_seeds.py --seeds 43,44                  # watch progress while it runs
```

Full table: [`results/multiseed/multiseed.md`](../results/multiseed/multiseed.md).

### 10.1 Domain-agnostic cost: 26/30 positive, reference point correctly zeroed

| Backbone | Shift | Cost mean +/- std | Seeds positive |
|---|---|---|---|
| CLDNN | D3 gain (negative control) | **0.0039 +/- 0.0057** | 2/3 |
| ResNet1D | D3 gain (negative control) | **0.0015 +/- 0.0012** | 3/3 |
| CLDNN | D2 CFO | 0.1240 +/- 0.0488 | 3/3 |
| ResNet1D | D4 phase noise | 0.3785 +/- 0.1433 | 3/3 |
| ResNet1D | D2 CFO | **0.6629 +/- 0.2092** | 3/3 |

- **26 of the 30 conditions are positive**;
- 6 of the 8 (backbone x shift) combinations are **positive for all 3/3 seeds**;
- **The negative control has the smallest cost on both backbones**; a real shift reaches up to 0.66 +/- 0.21.

The pattern "the reference point zeroes out + real shifts are consistently positive" is more
worth treating as a conclusion than any single rho.
**This is the most solid item in the project.**

### 10.2 rho: the sign flips with the seed (correcting Findings 8 and 9)

Three-seed means:

| Backbone | Shift | rho mean +/- std | Significant seeds | Fisher combined p |
|---|---|---|---|---|
| CLDNN | D1 SNR | +0.778 +/- 0.094 | 3/3 | 0.0001 |
| CLDNN | D3 gain (control) | +0.802 +/- 0.273 | 2/3 | 0.0000 |
| CLDNN | D2 CFO | -0.576 +/- 0.219 | 1/3 | 0.0103 |
| CLDNN | D4 phase noise | -0.200 +/- 0.468 | 1/3 | 0.2085 |
| CLDNN | D5 IQ imbalance | -0.305 +/- 0.432 | 1/3 | 0.1031 |
| ResNet1D | D3 gain (control) | +0.590 +/- 0.234 | 2/3 | 0.0001 |
| ResNet1D | D2 CFO | -0.314 +/- 0.316 | 1/3 | 0.0479 |
| ResNet1D | D4 phase noise | -0.312 +/- 0.347 | 1/3 | 0.0384 |
| ResNet1D | D5 IQ imbalance | +0.048 +/- 0.580 | 2/3 | 0.0156 |
| ResNet1D | D1 SNR | +0.311 +/- 0.349 | 1/3 | 0.0376 |

**Correction.**

- "CLDNN x D5 IQ imbalance rho=+0.103" written in Finding 9 becomes **-0.305** under three seeds;
- "CLDNN x D4 phase noise rho=+0.139" written in Finding 9 becomes **-0.200** under three seeds.

**The sign flips, not just the magnitude.** Therefore every single-seed rho value in Findings 8
and 9 should be understood as "one sample from that draw", not as the rho of that combination.

Also, **for half the combinations the standard deviation is the same order of magnitude as the
mean or larger**
(`+0.048 +/- 0.580`, `-0.200 +/- 0.468`, `-0.305 +/- 0.432`).
A quantity whose error bar exceeds its mean by an order of magnitude cannot be a vehicle for
conclusions.

### 10.3 Fisher combination recovers effects masked by a single run

CLDNN x D2 CFO looks like only 1/3 seeds significant on its own, but after combination
**p = 0.0103**; ResNet1D x D2 CFO is likewise 1/3, and after combination **p = 0.0479**.

This shows that "not significant in a single run" does not mean "no effect"; at n=10~16 layers
the statistical power is inherently low.
**The report should lead with the Fisher combined p, not the per-seed p.**

### 10.4 This section changes how the report has to be written

| Before (single seed) | Now (three seeds) |
|---|---|
| rho = +0.673 (p=0.037), ranking moderately consistent | rho = +0.778 +/- 0.094, 3/3 significant, Fisher p = 0.0001 |
| rho = +0.103 (p=0.785), ranking almost uncorrelated | rho = **-0.305 +/- 0.432**, sign opposite to the single seed |
| Domain-agnostic cost 0.0066 | 0.0069 +/- 0.0070 (mean smaller than the standard deviation, effectively near zero) |
| "consistently positive on both backbones" | **26/30 positive**, smallest for the negative control -- this is the number that holds up |

**Conclusion**: replace "what is rho" with "under how many conditions is the cost positive, and
what is the Fisher combined p".

---
## Verification status (updated: the torch side has been run for real)

### [done] Actually executed and passing

| Item | Script | Result |
|---|---|---|
| Quantization math, rank correlation, equal-error arms | `tests/test_quant_core.py` | **70/70 passed** (including cross-validation against `scipy.stats.spearmanr`) |
| End-to-end pipeline (numpy) | `tests/smoke_pipeline.py` | passes, about 7s |
| Representation-problem diagnosis | `tests/diagnose_data.py` | passes |
| Feature separability | `tests/diagnose_features.py` | passes |
| Difficulty regime and CFO experiment | `tests/diagnose_regime.py` | passes |
| Config parsing (including the mini-YAML fallback) | manual check | passes (fixed one nested-inline parsing bug) |
| **CLDNN training + multi-domain evaluation** | `scripts/train_baseline.py` | **passes**, 8.4 minutes, early stop at epoch 21 |
| **ResNet1D training + multi-domain evaluation** | `scripts/train_baseline.py` | **passes**, early stop at epoch 27 |
| **Per-layer sensitivity + rho + permutation test** | `scripts/run_sensitivity.py` | **passes** (10 layers x 5 bit widths x 2 domains) |
| **Three equal-reconstruction-error arms** | `scripts/run_equal_error.py` | passes |
| **Figure generation** | `scripts/make_figures.py` | passes, PNGs rendered |
| **Multi-seed experiment (3 seeds x 2 backbones x 5 shifts)** | `scripts/run_seeds.py` | **passes**, 44/44 units, 135 minutes |
| **Cross-seed aggregation (mean +/- std + Fisher combination)** | `scripts/aggregate_seeds.py` | **passes** |
| Syntax of all 36 .py files | `py_compile` | 0 failures |

Environment: Python 3.13.5 + torch 2.14.1+cpu + scipy 1.18.1 + sklearn 1.9.1 + matplotlib 3.11.2,
all installed in the project-local `.venv` (it does not pollute the system Python).

###  Archived failure case

`results/figures/_saturated_example/` keeps the figures and JSON from **when the source domain
was saturated**:

- `fig1_sensitivity_heatmap.png` -- the left panel (source domain) is almost entirely black,
  Delta_loss ~ 0; only the right panel (target domain) lights up. **This is a perfect
  visualization of "source-domain saturation degrading sensitivity"**, and it can go straight
  into the report to explain why the difficulty has to be controlled.
- `rank_cldnn_D1_snr.json` -- the source-domain sensitivity vector is
  `[7e-05, 0.0014, 0.0004, 1e-05, 0, 0, -0, -0, -0, -2e-05]`,
  7 of the 10 layers are 0, and the acc metric has only 2 distinct values.

**Keeping a failure case around is worth it**: it turns "we did control the difficulty" from a
claim into a documented operation.

---

## What these three findings are worth for the report

They are not a "list of pitfalls" but **three sections that can go straight into the report**:

1. **The choice of input representation is a first-order decision** (Finding 1) --
   not "I chose cumulants" but "I measured the difference in learnability between raw IQ and
   cumulants".
2. **The evaluation metric changes the conclusion** (Finding 2) --
   the loss metric and the acc metric give rho = +0.71 and +0.51.
   Giving both in the report and explaining the saturation mechanism is far more rigorous than
   reporting a single number.
3. **Domain shift strength is representation-dependent** (Finding 3) --
   this is the one most likely to become a "nobody has done this" conclusion, because it turns
   "which domain shift is most fatal" from an assertion into a measurable quantity.

**None of these three sections needs a new model, a new SOTA, or self-collected data.**
They come from doing the control experiments cleanly.
