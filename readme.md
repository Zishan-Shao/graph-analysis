# DiT-XL/2 ImageNet-256 activation dataset

This repository implements the **class-conditioned DiT-XL/2** plan below. The
checkpoint accepts ImageNet-1K class IDs. We use a fixed subset of 10 classes
with five independent initial-noise seeds each, giving 50 generated images.
These are **class indices**, not indices of photographs in an ImageNet archive:
the generator starts from noise and does not read source images.

## Reproduce the dataset

Use Python with CUDA-enabled PyTorch and install `requirements.txt` in that
environment. These results used PyTorch `2.10.0+cu128`; the other direct
dependencies are pinned in `requirements.txt`. The scripts use the pinned first-party
[`facebook/DiT-XL-2-256`](https://huggingface.co/facebook/DiT-XL-2-256)
checkpoint and its DDIM scheduler (`eta=0`). The first sampling run downloads model
weights into `data/raw/hf-cache/`.

```bash
python -m pip install -r requirements.txt
python scripts/make_imagenet_conditions.py

# Smoke test: one image plus nine activation snapshots.
python scripts/sample_dit.py --limit 1

# Generate the remaining conditions on selected usable GPUs.
python scripts/run_sharded.py --gpus 1,4,6

# Check all 50 images and 450 trace files.
python scripts/validate_dataset.py
```

The committed `data/conditions/imagenet_10_classes_5_seeds.jsonl` contains the
exact class IDs, class names, and noise seeds. Each line is one condition, and
the zero-based manifest index is its line number minus one. A clone can sample
directly from this file without downloading ImageNet. `make_imagenet_conditions.py` recreates
it from the [pinned PyTorch ImageNet class list](https://raw.githubusercontent.com/pytorch/hub/c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5/imagenet_classes.txt)
and checks that list's SHA-256. The fixed, zero-based class indices are:

| Class ID | Class name | Manifest indices |
|---:|---|---:|
| 207 | golden retriever | 0–4 |
| 281 | tabby | 5–9 |
| 340 | zebra | 10–14 |
| 386 | African elephant | 15–19 |
| 402 | acoustic guitar | 20–24 |
| 404 | airliner | 25–29 |
| 620 | laptop | 30–34 |
| 779 | school bus | 35–39 |
| 817 | sports car | 40–44 |
| 963 | pizza | 45–49 |

Each five-row group has replicates `r00` through `r04`; the manifest records
their exact seeds. To regenerate one condition without running the others:

```bash
python scripts/sample_dit.py --sample-id imagenet-0207-r00 --overwrite
# Equivalent selection by zero-based manifest row:
python scripts/sample_dit.py --manifest-index 0 --overwrite
```

The seed for each row is the first four bytes, read as an unsigned big-endian
integer, of SHA-256 of `dit-xl2-256:2026:<class_id>:<replicate>`. The committed
manifest is the source of truth for sampling; its metadata file records the
class-list URL, SHA-256, selected IDs, and seed base. Exact pixels can vary
slightly across CUDA hardware or software versions.

Each run saves `image.png`, `metadata.json`, and nine compressed `.npz` traces
under `data/generated/dit_xl2_imagenet256/<sample_id>/`. A trace stores the
block input `h_in` and block update `residual = h_out - h_in` for all 256
spatial tokens. Calls 10, 25, and 40 and blocks 4, 14, and 24 are recorded.
Token index `i` maps to row `i // 16`, column `i % 16` on the 16×16 patch grid.
With classifier-free guidance scale 4.0, traces use the conditional branch.
The arrays are saved as FP16 after subtracting in FP32. Metadata records the
true diffusion timestep, model revision, scheduler, seed, and software versions.

Downloaded weights and generated arrays are ignored by Git; the code and
condition manifest are tracked. `run_sharded.py` assigns manifest rows by
stable index modulo the number of selected GPUs and writes worker logs under
`data/generated/dit_xl2_logs/`.

## Completed graph experiment: E-2 through E-7

The main feature is the **original 1152-dimensional block-input activation**.
No PCA, channel selection, or feature normalization is applied in these results.
Each trajectory/call/block snapshot forms a separate graph of 256 token observations.
A datum is identified by its trajectory, denoiser call, block, and token index.
The update, token-grid position, class, diffusion timestep, and seed are retained
as attributes for interpretation. The independent experimental units are the
50 generation trajectories, rather than the 115,200 individual observations.

### Read the results

- [PDF report](reports/experiment_report.pdf): figures and measured findings.
- [Answers E-2–E-7](reports/answers_e2_e7.tex): one English LaTeX answer per question,
  ready to copy into the assignment.
- [Standalone LaTeX report](reports/experiment_report.tex): includes the same answers
  and vector figures; compile from `reports/` if a TeX installation is available.
- [Measured summary and hashes](reports/results_summary.json).
- [Curated figures](reports/figures/): PNG previews and vector PDF exports.

| Question | Implemented method | Measured result |
|---|---|---|
| E-2 | Gaussian affinity on raw activation distances; median-distance bandwidth per snapshot | 450 weighted 256×256 matrices |
| E-3 | Undirected union-kNN, all integer k from 2 through 64 | 315/450 graphs connected at k=2; all connected by k=15 |
| E-4 | Unweighted degree and local clustering distributions, components of at least 32 vertices | 2,739 component cases at k=2,4,8,16,32,64; all 450 main-k=16 degree distributions right-skewed |
| E-5 | Weighted normalized Laplacian, first 32 eigenvalues, 3D eigenvector embeddings, strict nodal domains | Representative selects q3 with eigenvalue 0.225608 and two domains; 17/450 main graphs lack a qualifying separated eigenvalue |
| E-6 (optional) | K=2 k-means on raw activations, five reproducible initializations | 450 partitions; 348 snapshots have a complete two-domain nodal partition for comparison |
| E-7 | Update graph energy and cosine similarity, spatial coherence, cross-layer neighbor retention | At k=16, mean update-vector energy/null ratio 0.617; neighbor retention 0.505 for blocks 4→14 and 0.539 for 14→24 |

The fixed illustrative snapshot is `imagenet-0207-r00`, call 25, block 14.
The main detailed setting is **k=16**, at which all 450 graphs have one
256-vertex component. Connected-component curves and Fiedler curves cover
every integer k=2,…,64; detailed component distributions and embeddings cover
k=2,4,8,16,32,64. Per-component plots exist for every large component at these
six settings. All six settings are retained, so the report's representative
example is accompanied by the full results rather than selected for appearance.

### Reproduce the analysis

Analysis uses the CPU and does not need PyTorch or a GPU. Start with the generated
traces described above. The pinned analysis environment uses Python 3.10.20;
the four direct numerical/plotting dependencies are in `requirements-analysis.txt`.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt

# Complete E-2–E-7 calculation, all component plots, and report.
.venv/bin/python scripts/run_graph_experiment.py --workers 4 --plot-workers 8

# After calculation, refresh the report without rerendering 2,739 component plots.
.venv/bin/python scripts/run_graph_experiment.py --from-results --skip-component-plots
```

Individual stages are also available:

```bash
.venv/bin/python scripts/build_interactions.py
.venv/bin/python scripts/analyze_graphs.py --workers 4
.venv/bin/python scripts/analyze_attributes.py
.venv/bin/python scripts/compare_clustering.py
.venv/bin/python scripts/plot_experiment.py --workers 8
.venv/bin/python scripts/plot_clustering_comparison.py
.venv/bin/python scripts/write_experiment_report.py
```

The graph outputs stay in `data/generated/graph_analysis/` and are ignored by Git:

| Output | Contents |
|---|---|
| `metadata.json`, `index.jsonl` | Configuration, fixed representative, snapshot indices and SHA-256 provenance |
| `interactions/` | All-to-all affinity, squared distance, bandwidth, and stable neighbor order |
| `connectivity.csv` | 28,350 graph records (450 snapshots × 63 k values) |
| `connectivity_summary.json` | First connected k in the scanned range, and component curves |
| `component_summary.csv`, `components/` | 2,739 large-component cases; distributions, eigenvalues, embeddings, nodal labels, update attributes |
| `component_plots/`, `component_plot_index.json` | One six-panel figure per large component at the six detailed k values |
| `fiedler_curves.csv`, `nodal_groups.csv` | Connected-graph Fiedler values and nodal-domain statistics |
| `attributes.csv`, `attribute_summary.json` | Update/spatial graph signals and trajectory-level summaries |
| `cross_layer_neighbors.csv` | Directed same-token neighbor overlap between the recorded blocks |
| `optional_clustering.csv`, `baseline_clusters/` | k-means seeds, partitions, objectives and comparable nodal-partition agreement |

A clone includes code, the exact sampling manifest, the report, and curated figures.
It regenerates the activation arrays by sampling the pinned checkpoint; it does
not download an ImageNet photograph subset. Re-running analysis on the same
saved arrays fixes neighbor ties and clustering seeds. Floating-point results
can vary slightly with the numerical library and hardware; hashes identify the
actual inputs used for the committed report.

### Definitions and interpretation

The all-to-all weights are

\[
S_{ij}=\exp\left(-\frac{\|h_i^{\mathrm{in}}-h_j^{\mathrm{in}}\|_2^2}
{2\sigma^2}\right),\qquad S_{ii}=0,
\]

where sigma is the median positive pairwise Euclidean distance in one snapshot.
The Gaussian bandwidth stays fixed as k changes. Edge weights are similarities
between activations, rather than attention weights. A pair is retained when
either vertex selects the other among its k nearest neighbors; its weight
remains the original affinity. Numerical work uses float64 on the saved FP16
activations, which does not recover precision lost during storage.

Degree and local clustering use the unweighted adjacency. Spectral analysis uses
`Lsym = I - D^(-1/2) W D^(-1/2)` with the retained nonnegative weights.
For nodal analysis, we search nontrivial eigenvalue indices 2 through min(32, |G|-1) using
absolute two-sided gaps of at least 1e-6 and a minimum-gap/eigenvalue ratio of
at least 0.05, then maximize the minimum adjacent gap. The complete spectrum
is computed to check the neighboring eigenvalue beyond the displayed first 32.
Cases without a qualifying eigenpair are marked unavailable. Strict nodal domains
are connected components in the positive and negative induced subgraphs;
numerical zeros are marked separately. See the
[spectral clustering tutorial](https://www.cs.columbia.edu/~jebara/6772/papers/Luxburg07_tutorial.pdf)
for affinity graphs and the normalized Laplacian.

The representative nodal domains have 131 and 125 tokens. Their mean update
norms differ, but the partition explains only 2.53% of update-norm variance.
The token-grid overlay is illustrative and does not accurately trace the dog's
semantic boundary. There is no token-level semantic ground truth here.

Across trajectories, update signals are smoother on activation graphs than
under random node assignment, and neighbor relationships partly persist across
layers. These findings motivate a later **cross-layer token-correction** study.
Spatial coherence may account for part of the association. Correction error,
rollout quality, and inference speed have not been measured in E-2–E-7.
A future reconstruction experiment should compare graph transfer against
spatial and global baselines on held-out trajectories before claiming an improvement.

Feature dimension can be varied in a separate sensitivity experiment using a
fixed nested channel ordering. This run uses all 1152 channels; no dimension
sensitivity or acceleration experiment is claimed. The digit-image tasks after
`\endinput` in the supplied assignment are outside this run's agreed scope.
