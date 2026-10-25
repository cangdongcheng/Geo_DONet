# Geo_DONet

Self-contained geometry-conditioned DeepONet with phase-aligned temporal PCA.
Code and small settings live here. Raw arrays, preprocessing artifacts, checkpoints,
logs, predictions and figures live in external storage, separate from the source.

Examples use `/scratch/project/geo_donet` for data and results. Replace this prefix
with an available storage location outside your home and source directories.
`--output` is required for preparation, training and cross-validation, so every
destination is chosen explicitly.

The model predicts `[AT(x), c1(x), ..., cK(x)]` from heart geometry `theta` and
canonical Cobiveco coordinates. A frozen node-specific template plus temporal PCA
basis reconstructs aligned Vm; predicted AT shifts it back to physical time.
The default objective is reconstructed-Vm MSE divided by training Vm variance.
AT/coefficient errors are diagnostics; optional auxiliary feature supervision has
weight zero by default. The architecture, decoder interpolation, Tanh activation,
Adam optimizer, node sampling and validation checkpoint selection follow
`Geo_DeepONet_PCA`. There are no runtime imports from the older folders.

## Output organization

A training run owns its checkpoint, evaluation, predictions and visualizations.
Use readable `--name` values for training and preparation; timestamps are only a
fallback when no name is supplied.

```text
/scratch/project/geo_donet/
  datasets/prepared/baseline_data/       # shared, fitted dataset artifacts
    raw/
    fold_01/
  runs/pca_baseline/
    config.json                         # training settings
    checkpoints/best.pt
    logs/
    figures/loss.png
    evaluation/
      config.json                       # evaluation settings, separate from training
      predicted_features.npz
      per_case_metrics.csv
      summary.json
      figures/
      logs/
      visualization/
        vtu/case_0100/Vm.pvd
        vtu/case_0101/Vm.pvd
        svg/case_0100/
    predictions/new_cohort/
```

`--evaluate --chkpt .../checkpoints/best.pt` writes to that model's `evaluation/`.
For another comparison use `--name comparison`, producing `evaluation_comparison/`,
or supply an exact destination with `--output`. Evaluation/export do not append
a date.
Existing destinations are rejected so prior metrics and exports are not overwritten.
Existing result directories are not moved automatically.

## Environment

Use a Python environment with NumPy, PyTorch, matplotlib and meshio. Development
checks used Python 3.11 and PyTorch 2.1. Choose a PyTorch build compatible with
your CPU or GPU and the available drivers.

For example, create a virtual environment in your checkout:

```bash
cd /path/to/projects/Geo_DONet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch matplotlib meshio
```

The command-line entry points import the `Geo_DONet` package. Keep that
checkout directory name:
`git clone <repository-url> Geo_DONet`.

No scikit-learn dependency is needed: waveform PCA uses the eigendecomposition
of temporal covariance. `python main.py --help` works without importing the ML
stack. Run the examples from the checkout root. Absolute script paths also work;
`python -m Geo_DONet.main` works from its parent directory.

For large datasets on a cluster, run preparation, training and exports in a
compute allocation. Set numerical-library thread counts to fit the CPU allocation;
see the export section for settings when using multiple frame workers.

## CLI help and reading the code

```bash
python main.py -h
python prepare_dataset.py -h
```

`-h` and `--help` show grouped options, built-in defaults, input meanings and
examples. These commands use only the Python standard library: they do not load
PyTorch, open data files or need a GPU. Preparation help contains only dataset
preparation flags; training/evaluation help describes the model and run options.
JSON configs still support the shared settings used by both entry points.

The scripts use numbered stages to make each workflow easy to follow:

- `main.py`: choose the device/data, create scratch outputs, run one action.
- `prepare_dataset.py`: load raw arrays, split hearts, fit each fold, save a manifest.
- `train.py`: load the split, restore normalization, build the model, train/validate,
  save the best checkpoint and loss history.
- `cross_validation.py`: check fold membership, train/test each fold, pool metrics.
- `config.py`: explicit `add_argument` definitions, then config loading and validation.

## Raw input contract

The input is an NPZ archive containing:

| Key | Shape | Meaning |
| --- | --- | --- |
| `theta` | `(N, G)` | Heart-level geometry parameters, currently G=60 |
| `coords` | `(M, 4)` | Shared canonical Cobiveco nodes, ordered ab/rt/tm/tv |
| `vm` | `(N, M, T)` | Float32 transmembrane voltage in physical mV |
| `time` | `(T,)` | Uniform increasing time grid in ms, at least 3 frames |
| `case_names` | `(N,)` | Unique Unicode/string heart identifiers |

Arrays must be pickle-free. Extra keys such as `ecg` are ignored. Geometry and
coordinates must be floating-point arrays. All hearts share the same canonical
node order and time grid. This implementation retains the existing node-lookup
decoder; it does not interpolate its template onto arbitrary meshes.

Two different PCAs are involved. Geometry PCA is upstream and already represented
by `theta`; this project does not refit it from those coefficients. Its fitting
provenance remains an upstream responsibility. Waveform PCA is fitted here using
training hearts only, independently for every split/fold.

## Prepare a fixed split

```bash
python prepare_dataset.py \
  --input /scratch/project/geo_donet/raw/cohort.npz \
  --config configs/pca.json --name baseline_data \
  --output /scratch/project/geo_donet/datasets/prepared
```

The command prints the named preparation manifest (`baseline_data/manifest.json`).
Default membership is first 95 training hearts, next 5 validation hearts,
remaining hearts test. The
split is saved before fitting. Every heart stays entirely within one split.

Preparation extracts the required NPZ members once into mmap-readable `.npy`
arrays on scratch, validates shapes/time/case IDs, computes threshold-crossing AT,
and fits training-only aligned node templates, residual temporal PCA and feature
normalization. By default the phase reference is the median training AT, matching
the original single-split preparation. `--reference-at 80` fixes the phase origin.
AT uses an upward crossing of -10 mV with linear interpolation; alignment uses
linear shifts and constant edge padding. Nodes without crossings are rejected.

```text
datasets/prepared/<preparation_name>/
  config.json
  provenance.json
  manifest.json                 # published only after every fold completes
  raw/
    dataset.json
    theta.npy, coords.npy, vm.npy, time.npy, case_names.npy
  fold_01/
    manifest.json
    splits.json
    decoder_basis.npz
    normalizer.npz
    targets.npy                 # (N, M, 1+K): ground-truth AT and PCA projections
    diagnostics.json            # oracle reconstruction MAE, explained variance
```

Raw data are shared by all folds. Preparation temporarily stores aligned
**training** waveforms on scratch, one fold at a time, and removes that workspace
when fitting completes. For 125 hearts with 50,797 nodes and 601 time points,
budget approximately 15 GB for extracted raw Vm plus 12 GB temporary alignment
space with 95 training hearts, in addition to the
original NPZ and final fold artifacts. Memory use is bounded by node blocks,
the node template and temporal covariance, rather than the full cohort.

To prepare a new split/PCA configuration without extracting another raw copy,
pass the existing `raw/dataset.json` as `--input`. A preparation with identical
settings can be reused directly by any number of training runs; it is not
implicitly regenerated by training.

## Train

```bash
python main.py --train \
  --input /scratch/project/geo_donet/datasets/prepared/baseline_data/manifest.json \
  --config configs/pca.json --name pca_baseline \
  --output /scratch/project/geo_donet/runs
```

Use the preparation path printed by your preparation command. CLI settings override
JSON settings, which override defaults. Paths inside JSON resolve relative to the
JSON file; CLI paths resolve relative to the current working directory. Unknown
settings and invalid types/counts are rejected. The exported `config.json` contains
resolved paths and settings. The split and number of components come from the
prepared dataset, not training CLI defaults.

Default training: width 200, depth 4, 5 PCA coefficients, Adam 5e-4, batch 8,
2,048 fresh nodes per batch, all times, 4,096 fixed validation nodes, validation
every 10 epochs, 5,000 epochs maximum, patience 1,000 epochs, seed 42.
Use `--patience 0` to complete every epoch. Node-sample counts must fit the mesh.

```text
runs/pca_baseline/
  config.json
  provenance.json               # command, Git revision/dirty state, code hashes
  environment.json
  status.json
  checkpoints/best.pt
  logs/run.log
  logs/loss.csv                 # flushed each epoch
  training_summary.json
  figures/loss.png
```

Named runs/preparations use the supplied name exactly. Without `--name`, a dated
folder is created; timestamp collisions receive numeric suffixes. Existing named
folders are never replaced.
Output roots inside home/source are rejected; unavailable scratch account roots
fail explicitly. Evaluation, plotting and temporary input conversion use the same
scratch routing. This version starts fresh training; it does not implement resume.

## Evaluate or predict

```bash
python main.py --evaluate \
  --chkpt /scratch/project/geo_donet/runs/pca_baseline/checkpoints/best.pt
```

Evaluation restores model architecture and normalization from the checkpoint and
uses its saved test membership. It writes `evaluation/` inside the checkpoint's
run/fold folder, containing `predicted_features.npz`, `per_case_metrics.csv`,
`summary.json`, evaluation config/logs and trace plots. Training metadata stay
untouched. An explicit `--output` is an exact folder, not a root for another date. Metrics include Vm RelL2/MAE, direct/decoded AT errors, coefficient
errors, upstroke retention and the fraction of valid decoded activation times.

Checkpoints reference their exact prepared manifest and record its digest. New
checkpoints/features also save relative references so moving the complete project
tree preserves those links. For a preparation moved independently, use
`--prepared-manifest /new/location/fold_01/manifest.json` for evaluation or export. Keep
the referenced preparation and raw arrays available. Basis/target/scale/split
changes are rejected, rather than silently evaluated with a different decoder.
This new checkpoint schema is explicit; old checkpoints remain usable through
their original entry points. Numerical parity with the old implementation is
covered by synthetic regression checks.

```bash
python main.py --predict \
  --input /scratch/project/geo_donet/raw/new_cohort.npz \
  --chkpt /scratch/project/geo_donet/runs/pca_baseline/checkpoints/best.pt
```

Prediction needs `theta`, `coords`, `time`, and `case_names`; Vm/AT/PCA labels are
not required. It validates canonical node order and the decoder time grid and
saves compact predicted features in the model's `predictions/new_cohort/` folder.
Use `--name` to label the cohort or `--output` for an exact destination.
Full decoded Vm is opt-in with `--save-vm`,
written one heart per `.npy` file. It is never stacked into another full cohort.

## Cross-validation

```bash
python prepare_dataset.py \
  --input /scratch/project/geo_donet/raw/cohort.npz \
  --config configs/cv.json --name cv_data \
  --output /scratch/project/geo_donet/datasets/prepared

python main.py --cv \
  --input /scratch/project/geo_donet/datasets/prepared/cv_data/manifest.json \
  --config configs/cv.json --name cv_baseline \
  --output /scratch/project/geo_donet/runs
```

The CV config preserves the historical five-fold seed-42 shuffle, 25 test hearts
per fold, 95 fitting/5 validation hearts, fixed 80-ms phase origin, and no early
stopping. Every fold fits its own template, temporal PCA and normalizers. Each
fold's test heart appears exactly once in the pooled results. Folds share raw Vm;
they do not share fitted preprocessing. `--cv` runs folds sequentially, saving
fold checkpoints/evaluations and pooled `summary.json`/`per_case_metrics.csv`.
To run just one fold, pass its `fold_XX/manifest.json` to `--train`.

## Cluster execution

Submit the same preparation, training, evaluation and export commands through
your cluster's scheduler. Select CPU/GPU, memory and walltime resources according
to dataset size and workload. Environment activation and scheduler directives
are specific to your installation.

Local PBS submission scripts belong in the ignored `pbs/` folder and are not
included in the repository. Keep account names, absolute machine paths and
scheduler settings in those local scripts. Place scheduler logs, temporary files
and matplotlib caches in external storage as well.

To evaluate and export in a single allocation, run the evaluation command first,
then the VTU export command using its `predicted_features.npz`. Evaluation can use
`--device cuda`; VTU reconstruction and frame writing use CPUs. Use shell error
handling such as `set -e` so export starts only after evaluation succeeds.

## Visualization

```bash
python utils/export_svg.py \
  --input /scratch/project/geo_donet/runs/pca_baseline/evaluation/predicted_features.npz \
  --mesh /scratch/project/geo_donet/raw/canonical.vtu --case 100 --times 50 100 150

python utils/export_vtu.py \
  --input /scratch/project/geo_donet/runs/pca_baseline/evaluation/predicted_features.npz \
  --mesh /scratch/project/geo_donet/raw/canonical.vtu --cases 100 101 --frame-stride 5 --workers 8
```

Exports decode saved features without rerunning the network and validate the
mesh Cobiveco fields before rendering. VTUs default to the feature archive's
`visualization/vtu/` folder; SVGs use `visualization/svg/case_XXXX/`. An explicit
`--output` is the exact export folder. No date subfolder is added. SVGs retain the original physical color
limits; VTUs include prediction, truth and errors when truth is available.

### Two-case VTU export and neighbour diagnostics

VTU export selects the **first two saved cases** by default (one if the archive
contains only one case). Use `--cases 100 101` to choose explicit saved indices,
or the existing `--case 100` for a single case. Selected indices must exist in
that archive; a CV fold may contain different held-out indices. `--frame-stride 5`
exports every fifth recorded frame; use `--frame-stride 1` for all frames.

Use `--workers 8` with an allocation of at least eight CPUs (16 CPUs leaves
headroom); the default is one worker. Workers use threads to share geometry and
one reconstructed heart instead of copying the large arrays into separate
processes. Each worker computes neighbour fields, compresses, and writes its own
frame file. Hearts remain sequential, and PVD/CSV entries remain in time order.
Use `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` for the
export command to avoid nested numerical-library thread teams. Parallel writes
can reduce compression/CPU time but remain limited by shared-scratch throughput.
The exporter prints loading/reconstruction timings and progress every 25 frames.
On interruption or a worker error, queued tasks are cancelled; a PVD collection
is published only after every selected frame of that case succeeds.


```text
evaluation/visualization/vtu/
  export_info.json
  case_0100/
    Vm.pvd
    series/frame_0000.vtu, ...
    neighbour_metrics.csv
    export_info.json
  case_0101/
    Vm.pvd
    series/frame_0000.vtu, ...
    neighbour_metrics.csv
    export_info.json
```

Open each case's `Vm.pvd` in ParaView. All frames use the unchanged canonical
reference geometry and tetrahedral connectivity. Choose these Point Data fields:

| Fields | Meaning |
| --- | --- |
| `Vm_pred`, `Vm_true` | Predicted and simulation Vm in mV |
| `Vm_signed_error`, `Vm_abs_error` | Prediction minus GT, and its absolute value |
| `ndiff_pred`, `ndiff_true` | Mean absolute Vm difference to each node's neighbours |
| `ndiff_signed_error`, `ndiff_abs_error` | Predicted minus GT neighbour magnitude, and its absolute value |
| `signed_pred`, `signed_true` | Vm at the node minus the mean Vm of its neighbours |
| `signed_error`, `signed_abs_error` | Predicted minus GT signed contrast, and its absolute value |
| `neighbour_count` | Number of distinct mesh-edge neighbours |

Neighbours come from unique undirected tetrahedral edges; shared tetrahedra do
not count an edge twice. Voltage contrasts have mV units and are not
divided by edge length, so they are not physical Cartesian gradients. Nodes with
no neighbours have degree zero and NaN neighbour fields. The CSV records contrast
errors and total squared edge differences at each exported time, in mV and mV².

Use an **evaluation** feature archive to include GT and error fields. A prediction-
only archive produces predicted Vm and predicted neighbour fields. The exact fitted
decoder and raw GT are read from the archive's saved preparation identity. The
mesh, decoder and raw metadata are loaded once; hearts are processed sequentially,
with frame concurrency controlled by `--workers`. All outputs remain on scratch.

If the preparation folder was moved after evaluation, pass its new fold manifest
explicitly with `--prepared-manifest`, for example
`/scratch/project/geo_donet/datasets/prepared/baseline_data/fold_01/manifest.json`.
The exporter requires the manifest checksum recorded in the saved predictions to match, so
this relocates the same fitted decoder rather than substituting a new PCA basis.
The prediction archive is left unchanged.

## Source organization and verification

`data_handling/` owns schema validation, loading, heart splits, preprocessing and
normalization. `model.py` and `decoder.py` own the network and fixed reconstruction.
`train.py`, `cross_validation.py`, `evaluate.py`, and `metrics.py` own the learning/evaluation workflow.
`utils/` contains visualization and mesh-export scripts.

From the directory containing the checkout, with your Python environment active:

```bash
cd /path/to/projects
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest Geo_DONet.tests.test_workflow Geo_DONet.tests.test_neighbours -v
```

Checks use tiny synthetic arrays in `/tmp`: held-out-data isolation, exact decoder
forward/gradient parity, training-checkpoint parity with the original PCA runner,
prepare/train/evaluate/predict, CV pooling, artifact identities, config precedence,
and output routing. Export checks cover two cases, reference-mesh ordering,
hand-calculated neighbour contrasts, and duplicate-edge/isolated-node handling.
They do not load the cardiac cohort or submit HPC jobs.
Two optional parity tests compare against a legacy implementation at
`Geo_DONet_experiments/Geo_DeepONet_PCA`, relative to the checkout's parent directory. They
skip when that source is absent; the workflow/export checks run independently.
