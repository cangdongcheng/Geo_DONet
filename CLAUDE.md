# DIMON — Neural Operator for Cardiac Electrophysiology

Fork of the DIMON framework (Yin et al., Nature Computational Science 2024) adapted for biventricular 12-lead ECG surrogate modeling. Parent project: `BASE/CLAUDE.md`.

## Directory layout

| Folder | What it does | Trunk input | Physics loss |
|---|---|---|---|
| `Cobiveco/` | Baseline DIMON on Cobiveco coords | 3D Cartesian | No |
| `Cobiveco_v2/` | Trunk = coords + anisotropy (12D) | 3D + 9D fiber basis | No |
| `Cobiveco_with_fiber/` | Trunk = 4D Cobiveco | 4D Cobiveco | No |
| `Cobiveco_with_scar/` | Dual branch: geometry + scar SDF | 3D Cartesian | No |
| `DIMON_PINN/` | **PINN variant** — Eikonal PDE residual | 3D Cartesian | Yes (anisotropic Eikonal) |
| `Geo_DONet/` | Geometry-conditioned full V_m neural-operator benchmark | 4D Cobiveco + time | No |
| `Geo_DeepONet/` | Geometry-conditioned activation-time-only DeepONet | 4D Cobiveco | No |
| `Geo_DeepONet_PCA/` | Predict AT + phase-aligned PCA features; train through fixed V_m decoder | 4D Cobiveco | No |
| `Geo_MLP/` | Pointwise conditional-MLP ablation for V_m | 4D Cobiveco + time | No |
| `Geo_MLP_PCA/` | MLP predicts AT + 5 PCA coefficients using the same V_m decoder/loss as Geo_DeepONet_PCA | Concatenated 60D geometry + 4D Cobiveco | No |
| `LV/` | Upstream example 3 (180-mode, LV-only) | 3D UVC | No |
| `Laplace/` | Upstream example 1 (2D Laplace) | 2D | No |
| `ReactionDiffusion/` | Upstream example 2 (2D R-D + LDDMM) | 3D (x,y,t) | No |

Shared data at repo root:
- `DIMON_training_data_healthy.npz` — 125 hearts x 9 pacing sites x 50797 nodes. Keys: `theta` (125,60), `pacing` (9,4), `u_data` (125,9,50797), `cartesian_coords` (50797,3), `ref_anisotropy` (50797,9), `cobiveco` (50797,4).
- `reference_cobiveco.npz` — canonical reference mesh Cobiveco coordinates.

## Current V_m-surrogate track (2026-08-12)

The active data are now `/home/svu/e1032484/scratch/geo_donet_data_f601.npz`
(125 hearts, 50,797 nodes, 601 frames at 1 ms) and its f121 view (stride 5).
The fixed split is 95 train / 5 validation / 25 test unless a five-fold run is
explicitly stated.

- The clean `Geo_DONet` f121 benchmark gives **4.73 mV V_m MAE** and about
  **6.03 ms AT MAE** on the fixed 25-heart test split.
- Blurring the ground truth toward that prediction reduces error, but only
  partially: temporal Gaussian smoothing improves the 25-heart MAE by about
  **11.4%** at 7.5 ms sigma; spatial neighbour relaxation improves the first
  test heart by **18.9%** at 16 passes. This supports both spatial and temporal
  smoothing in the prediction, but neither alone explains all error.
- Phase-aligning each waveform by activation time makes the remaining temporal
  variation highly compressible. On held-out hearts, the f601 oracle decoder
  with five PCA modes has **0.080 mV MAE** and explains **97.06%** of aligned
  residual variance. The shift/interpolation floor is about **0.058 mV**.
- `Geo_DeepONet_PCA` predicts AT + five coefficients but is supervised through
  differentiable V_m reconstruction. Its current fixed-split test result is
  **1.887 +/- 0.380 mV V_m MAE**, **4.504 +/- 1.066 ms direct AT MAE**, and
  **4.533 +/- 1.045 ms decoded AT MAE**. However, it retains only
  **0.330 +/- 0.006** of the true maximum-upstroke slope, so low global MAE has
  not solved wavefront sharpness.
- The matched width-200/depth-4 old AT-only `Geo_DeepONet` gives
  **7.28 +/- 1.80 ms AT MAE**. Both use Adam at 5e-4, but this is not a pure
  loss ablation: the old default full 95-heart batch gives one update/epoch,
  whereas the PCA run uses batches of eight (about 12 updates/epoch), sampled
  nodes, and learned output heads.
- `Geo_MLP` looked strong on the fixed split (**3.82 mV V_m MAE**), but its
  five-fold pooled result is **5.09 +/- 3.70 mV** and **8.41 +/- 5.45 ms**
  interpolated AT MAE. The corresponding Geo_DONet five-fold result is
  **4.64 +/- 1.61 mV** and **6.03 +/- 1.23 ms**. The single split was therefore
  optimistic for the MLP; use cross-validation for architectural claims.

Detailed operational context and experiment paths are in `VANDA_HANDOFF.md`.

### AT/PCA MLP comparison prepared (2026-09-17)

`Geo_MLP_PCA/` now provides single-split and five-fold CV entry points and PBS
scripts. It imports the shared `Geo_DeepONet_PCA` training/evaluation routines;
only the feature predictor changes to a single Tanh MLP. Defaults: 64 inputs,
6 outputs, width 200/depth 4, reconstructed V_m MSE only, 5,000 epochs,
no early stopping, best validation checkpoint, Adam 5e-4, 8 hearts x 2,048
nodes x all 601 times. CV refits the template/PCA/scales on each fitting fold.
Width 200 matches hidden width (134,806 MLP vs 255,606 DeepONet parameters);
optional width 280 nearly matches parameter count (255,926). No cardiac-data
training has been launched for this experiment. See its README for commands.

## Architecture (all cardiac variants)

Three-branch MIONet (`opnn` class in each folder's `opnn.py`):
- **Geo branch**: PCA geometry modes theta (60D) -> 4x[200] Tanh
- **Pace branch**: Cobiveco pacing target (4D) -> 4x[200] Tanh
- **Trunk**: spatial coordinates (3D or 4D) -> 4x[200] Tanh
- **Combination**: `(geo * pace) @ trunk.T` -> output shape [M, 9, N]

All examples: `python main.py --epochs N --device cuda` to train, `--test-model 1` to evaluate.

## Training data provenance

The activation time maps in `u_data` were generated by **Eikonal** simulations (openCARP DREAM solver) on the original meshes at **~1500 um average edge length**. The 50797 reference nodes are sampled from a canonical reference mesh at the same resolution. 9 pacing configurations correspond to the existing `Eikonal/{Healthy,Scar}_eikonal_{0..8}` runs.

## PINN experiment — negative result, open investigation

### What was tried
`DIMON_PINN/main.py` adds an anisotropic Eikonal PDE residual loss on top of the data MSE:

```
Eikonal: sqrt(grad_u^T * D * grad_u) = 1
D = v_f^2 * (f f^T) + v_s^2 * (s s^T) + v_n^2 * (n n^T)
```

where `f,s,n` are the fiber/sheet/normal directions from `ref_anisotropy`, and velocities are `v_f=640, v_s=v_n=240` um/ms. The loss is:

```
loss = loss_data + lambda_pde * loss_pde
lambda_pde = (data_val / pde_val) * alpha   # adaptive weighting
```

Multiple alpha values tested (0.05, 0.1, 0.2, 0.3, 0.4) plus ablation. Prediction outputs saved under `DIMON_PINN/predictions/`.

### Why it failed (hypothesis)
The training data (Eikonal activation times) was computed on **~1500 um meshes**. The PINN loss computes `grad_u` via `torch.autograd.grad` w.r.t. the spatial coordinates, which amounts to differentiating the neural operator output — but the **ground truth** itself was generated at a resolution where the Eikonal wavefront is only ~2-4 elements wide. The PDE residual the network is trying to satisfy may be inconsistent with the coarse-mesh ground truth, creating a conflicting training signal.

### Current investigation (2026-04-10)
User wants to do empirical testing on the ground truth data to verify whether the mesh resolution is indeed the limiting factor. Possible directions:
- Compare Eikonal ATs on coarse (1500 um) vs fine (400 um) meshes for the same cases/pacing
- Compute the discrete Eikonal residual on the ground truth data directly (no neural network)
- Check how smooth/noisy the AT gradients are at 1500 um resolution

## Environment

```bash
source ~/load_dimon_env.sh   # pytorch/2.6.0-py3-cu11.8, scikit-learn
```
GPU queues: `gpu`, `gpu1`, `gpu2` (production), `gdev` (interactive). `torch.cuda.is_available()` is False on login nodes — expected.

## Conventions
- Activation times in **ms**
- Spatial coordinates in **um** (matching openCARP)
- Cobiveco order: `(ab, rt, tm, tv)` — same as parent project
- Fiber basis in `ref_anisotropy`: columns 0-2 = fiber, 3-5 = sheet, 6-8 = normal
- Eikonal velocities: longitudinal 640, transverse 240, normal 240 (um/ms). Note: `SM_eikonal.py` uses 600/240/240 — the 640 in DIMON_PINN is slightly different, verify which is canonical.
