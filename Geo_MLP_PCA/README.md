# Geo_MLP_PCA: MLP architecture ablation

Predict AT + five phase-aligned PCA coefficients with a single Tanh MLP, then
reconstruct V_m using exactly the same fixed decoder as Geo_DeepONet_PCA.

```text
[60 geometry PCA parameters, 4 Cobiveco coordinates]
 -> 4 hidden layers of width 200, Tanh
 -> 6 standardized outputs
 -> inverse feature normalization: [AT, PC1, ..., PC5]
 -> node template + PCA residual + activation-time shift
 -> V_m at all 601 times
```

Time is supplied by the decoder's time grid, not as an MLP input. The MLP is
evaluated at each sampled node. There are no separate branch/trunk networks or
latent dot products. Geometry PCA parameters and waveform PCA coefficients are
different quantities.

## Controlled comparison

These entry points reuse the actual training, decoding, and evaluation code in
`../Geo_DeepONet_PCA/`; they do not maintain copies of the training loops.
The shared architecture factory loads historical untagged DeepONet checkpoints
and explicitly tagged MLP checkpoints. Existing DeepONet defaults are preserved.

Matched settings: first 95/5/25 hearts, f601 data, five PCA components, training
normalization, fixed node lookup template, Adam at constant 5e-4, depth four,
batch eight hearts, 2,048 newly sampled nodes per step, all 601 time values,
4,096 fixed validation nodes, validation every ten epochs, seed 42.

```text
loss = mean(((decoded_Vm - ground_truth_Vm) / training_Vm_std)^2)
```

Feature loss weight is zero. AT/PCA errors are diagnostics, not objectives.
The fixed-split run reuses the existing prepared data and train-only basis;
no additional preparation is necessary.

Early stopping is off by default for this experiment; train all 5,000 epochs,
then test the checkpoint with the lowest validation loss. The historical
single-split DeepONet run used patience 1,000; to reproduce that stopping rule,
pass `--patience 1000`. Both CV runners train all 5,000 epochs. Equal seeds match
the NumPy query/heart sampling; network initialization differs by architecture.

Equal width does not mean equal parameter count:

| Architecture | Width | Depth | Parameters |
|---|---:|---:|---:|
| DeepONet AT+5PC | 200 | 4 | 255,606 |
| MLP AT+5PC, primary run | 200 | 4 | 134,806 |
| MLP AT+5PC, optional parameter comparison | 280 | 4 | 255,926 |

The width-280 run differs by about 0.13% in parameter count from the DeepONet.
Use `--width 280` to run it; width is included in checkpoint/output names.
Decide comparisons before inspecting test results and use validation for tuning.

## Run on a compute node

Use the dimon environment on an A40 with at least 64 GB host RAM. The raw
archive is about 11 GB compressed and 15 GB in memory; do not load it on login.
The pointwise MLP does more work per geometry/node pair than the separable
DeepONet, so runtime need not match the earlier operator run.

```bash
source ~/load_dimon_env.sh
cd /home/svu/e1032484/Geo_DONet/Geo_MLP_PCA
python -u main.py --device cuda
```

Test the best validation checkpoint:

```bash
python -u main.py --test-model --device cuda \
  --model-path CheckPts/geomlp_pca_vmloss_k5_w200_d4_n2048_5000ep.pt
```

Or submit `qsub train.pbs` from this folder for training followed by testing.
The two-hour allocation is a starting budget, not a measured runtime guarantee.

Outputs include loss.csv/loss.png and, under `Predictions/<checkpoint>/Test/`,
test_summary.txt, test_features.npz, vm_metrics.csv, and sample waveform plots.
Metrics match the PCA operator: full-field Vm RelL2/MAE, direct AT RelL2/MAE,
decoded AT MAE, PCA errors, and maximum-upstroke retention. Compare both MAE
and upstroke retention: the existing operator improves MAE but smooths upstrokes.

For the existing VTU exporter, pass this run's `test_features.npz` explicitly:

```bash
python -u ../Geo_DeepONet_PCA/export_vtu.py --case 100 \
  --features Predictions/geomlp_pca_vmloss_k5_w200_d4_n2048_5000ep/Test/test_features.npz
```

## Five-fold CV

```bash
python -u main_cv.py --device cuda
# Alternatively, from the login/submission node:
qsub cv.pbs
```

This reuses the exact seed-42 heart folds and refits each decoder using only
that fold's 95 fitting hearts: node templates, waveform PCA, coefficient scales,
and Vm normalization. The global fixed-split basis is not reused for CV.
Five complete trainings are evaluated on their best validation checkpoints.
Outputs default to `CV_5fold_5000ep_w200_d4_n2048_f601_vmloss_mlp/`.
The PBS requests one GPU, 96 GB host RAM and four hours; check runtime on your
first run. The existing DeepONet CV took 92.5 minutes, but MLP timing is unmeasured.
