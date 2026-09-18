# PCA-predicted Vm to ECG

`pca_chain.py` trains one of the existing `model.py` transfer architectures and
evaluates the fixed-split Geo_DeepONet_PCA predictions through it. This is a
learned mapping from canonical-mesh Vm to ten electrode potentials, not a
physical torso/lead-field solver. It has no explicit geometry input.

The old `main.py` and `eval_chain.py` still target the original NSCC environment
and old Geo_DONet architecture. Use the new entry point for this experiment.

## Prior checkpoint search (2026-09-19)

The original run is documented in
`/home/svu/e1032484/cardiac_simulation/ARCHIVE.md`: temporal_conv, 10,000 epochs,
f121, fixed 95/5/25 split, checkpoint named
`ECG_transfer/CheckPts/temporal_conv_10000ep.pt`. No corresponding checkpoint was
found in accessible Vanda home/scratch storage. The original NSCC paths are not
mounted here. We prepare a fresh transfer run rather than assume its weights
or normalization are available.

## Interactive compute workflow

Request a GPU node from the login/submission node (adjust walltime as needed):

```bash
qsub -I -l select=1:ncpus=8:ngpus=1:mem=64gb -l walltime=03:00:00
```

Inside the allocated node:

```bash
source ~/load_dimon_env.sh
cd /home/svu/e1032484/Geo_DONet/ECG_transfer
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python -u pca_chain.py train --device cuda
python -u pca_chain.py evaluate --device cuda
```

Training defaults: temporal_conv, 10,000 epochs, Adam 1e-4, batch eight hearts,
seed 42, no early stopping. First 95 hearts fit weights and normalization,
next five select the best validation ECG-MSE checkpoint, last 25 are test only.
This is a new run, not an exact reproduction of historical training settings.
The script refuses to overwrite an existing checkpoint: choose a different
`--checkpoint` for another training run. Pass the same path to evaluation.

Checkpoint:

```text
CheckPts/temporal_conv_f121_10000ep_pca_chain.pt
```

The checkpoint saves model configuration, normalization, case/split metadata,
node coordinates and time grid. Both GT Vm and predicted Vm are normalized using
the TRANSFER model's saved scales, not the PCA feature normalizer. Legacy
weights-only checkpoints are not silently accepted; their exact training setup
would have to be recovered first.

No new PCA inference or training is needed: the default input is the existing
`Geo_DeepONet_PCA/Predictions/geodeeponet_pca_vmloss_k5_w200_d4_n2048_5000ep/Test/test_features.npz`.
The fixed decoder uses `scratch/pca_phase_aligned_basis_f601.npz`.

## Time grid and outputs

The transfer model trains on `scratch/geo_donet_data_f121.npz` (121 frames,
5-ms spacing), matching the previous ECG-transfer approach. PCA waveforms are
decoded on their original f601 grid, THEN sampled at these 121 matching times.
The temporal-convolution model must not silently be applied at a different
sampling interval: that changes its physical-time receptive field.

Evaluation runs one test heart at a time and compares:

1. Transfer(GT Vm) against simulation ECG: the transfer model's own errors.
2. Transfer(PCA-predicted Vm) against simulation ECG: the complete pipeline.
3. Transfer(PCA-predicted Vm) against Transfer(GT Vm): change due to replacing Vm.

The first is a diagnostic baseline, not a mathematically guaranteed upper bound
on accuracy. Error components are not necessarily additive. These are fixed-split
results, not cross-validation.

Both ten-electrode and derived twelve-lead results report relative L2, MAE and
mean channelwise Pearson correlation. Flat channels are excluded from Pearson
averaging and the number of valid channels is recorded. Units follow the existing
ECG archive/code convention of mV. Electrode order: LA, RA, LL, RL, V1,...,V6.
The standard twelve-lead conversion uses limb differences and the Wilson central
terminal for chest leads; RL is not a separate output lead.

Default output directory:

```text
Predictions/pca_chain_temporal_conv_f121_10000ep_pca_chain/
  test_summary.txt
  per_case_metrics.csv
  vm_metrics_on_ecg_grid.csv
  ecg_predictions.npz
  case100_12lead.png
  case101_12lead.png
  run_config.json
```

Each plot overlays simulation ECG, Transfer(GT Vm), and Transfer(predicted Vm).
Only ECG arrays are saved, avoiding another multi-GB Vm prediction archive.
For a first single-heart reconstruction after training:

```bash
python -u pca_chain.py evaluate --device cuda --cases 100 \
  --out-dir Predictions/pca_chain_case100
```

For all 601 time points, train a separate transfer model with `--data
/home/svu/e1032484/scratch/geo_donet_data_f601.npz` and a new `--checkpoint` path.
Evaluation reads the matching data path and time grid from that checkpoint.
Request enough host memory (64 GB minimum) and check the reported runtime/GPU
memory. Heavy data loading, training, and reconstruction belong on compute nodes.

## Verification

`smoke_pca_chain.py` uses only tiny synthetic arrays to test training/evaluation,
checkpoint normalization, time-grid alignment, held-out-case checks, metrics and
the twelve-lead conversion. It does not load the cardiac datasets.
