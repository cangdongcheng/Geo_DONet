# Geo-DeepONet PCA + slope decoder

This folder tests and trains a proposed representation in which every node
waveform is represented by:

```text
[activation time, local upstroke slope, residual PCA coefficients]
```

AT translates the waveform. Slope controls a monotone local time warp around
depolarization; outside +/-30 ms the warp is exactly the identity. A
node-specific lookup template and temporal PCA represent the waveform after AT
and slope have both been normalized.

`oracle_decoder.py` fits the template and PCA using the first 95 hearts only.
It evaluates the final 25 hearts using their true AT, true slope, and oracle PCA
projections. Consequently, this is a decoder-capacity experiment, not neural
network performance.

Run on an interactive CPU or GPU compute node with at least 64 GB RAM:

```bash
source ~/load_dimon_env.sh
cd /home/svu/e1032484/Geo_DONet/Geo_DeepONet_PCA_slope

python -u oracle_decoder.py
```

The computation itself is CPU/Numpy; a GPU allocation is acceptable but not
required. Do not run it on the login node because it loads the complete f601
archive and performs two passes over 95 hearts.

Outputs:

```text
/home/svu/e1032484/scratch/pca_at_slope_f601/summary.txt
/home/svu/e1032484/scratch/pca_at_slope_f601/recon_vs_k.csv
/home/svu/e1032484/scratch/pca_at_slope_f601/recon_vs_k.png
/home/svu/e1032484/scratch/pca_at_slope_f601/test_traces.png
/home/svu/e1032484/scratch/pca_at_slope_basis_f601.npz
```

The completed oracle result at K=8 is:

```text
cumEVR 0.99348
V_m MAE 0.0717 mV
max-dV/dt fraction 0.8146
```

The representation is accurate globally, although the upstroke fraction is
not better than the earlier AT-only oracle (~0.822). The learned experiment is
still useful for testing whether an explicit slope control helps optimization.

## Prepare eight-component targets

Run once on a compute node after producing the oracle basis:

```bash
python -u prepare_data.py
```

This writes:

```text
/home/svu/e1032484/scratch/geo_deeponet_pca_slope_f601_k8.npz
```

## Train with reconstructed-V_m supervision

The inputs and outputs are:

```text
branch input:  60 geometry PCA parameters
trunk input:   4 Cobiveco spatial coordinates
outputs/node:  AT, bounded positive slope, PC1, ..., PC8
decoder:       fixed node template + PCA + AT/slope time warp
loss:          reconstructed V_m MSE only
```

There is no direct AT, slope, or PCA loss. Their errors are diagnostics only.
Slope uses a bounded sigmoid parameterization fitted from the first 95 hearts;
this keeps the local time warp positive and monotone during training.

On an interactive A40 node:

```bash
python -u main.py \
  --device cuda \
  --epochs 5000 \
  --batch-size 8 \
  --nodes-per-step 2048 \
  --val-nodes 4096 \
  --n-components 8 \
  --width 200 \
  --depth 4 \
  --lr 5e-4
```

Early stopping is disabled by default, so this completes all 5,000 epochs.
Pass `--patience N` explicitly if a later run should stop after `N` epochs
without validation improvement.

Or submit `qsub train.pbs`. The default checkpoint is:

```text
CheckPts/geodeeponet_pca_slope_vmloss_k8_w200_d4_n2048_5000ep.pt
```

## Test

```bash
python -u main.py \
  --test-model \
  --device cuda \
  --model-path CheckPts/geodeeponet_pca_slope_vmloss_k8_w200_d4_n2048_5000ep.pt
```

Evaluation reports direct feature errors, decoded V_m/AT/slope metrics, and
max-dV/dt retention on the fixed 25-heart test split.
