"""Write measured E-2--E-7 LaTeX answers, a PDF report, and a compact summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/raw/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def mean(entries: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in entries if row[field] != ""]))


def condition_rows(entries: list[dict]) -> list[dict]:
    result = []
    for call in (10, 25, 40):
        for block in (4, 14, 24):
            subset = [row for row in entries if int(row["denoiser_call"]) == call
                      and int(row["block_1based"]) == block]
            result.append({"call": call, "block": block, "n": len(subset),
                           "degree_mean": mean(subset, "degree_mean"),
                           "clustering_mean": mean(subset, "clustering_mean"),
                           "lambda2_mean": mean(subset, "fiedler_value")})
    return result


def image_page(pdf: PdfPages, title: str, text: str, path: Path) -> None:
    figure = plt.figure(figsize=(11.7, 8.3))
    figure.text(0.05, 0.95, title, fontsize=17, weight="bold", va="top")
    lines = "\n".join(textwrap.fill(paragraph, 125) for paragraph in text.split("\n"))
    figure.text(0.05, 0.9, lines, fontsize=10, va="top", linespacing=1.4)
    axis = figure.add_axes([0.045, 0.04, 0.91, 0.72])
    axis.imshow(plt.imread(path))
    axis.set_axis_off()
    pdf.savefig(figure)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/generated/graph_analysis")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    figure_root = args.report_dir / "figures"
    metadata = json.loads((args.graph_dir / "metadata.json").read_text())
    connectivity_summary = json.loads((args.graph_dir / "connectivity_summary.json").read_text())
    components = rows(args.graph_dir / "component_summary.csv")
    fiedler = rows(args.graph_dir / "fiedler_curves.csv")
    attrs = rows(args.graph_dir / "attributes.csv")
    overlaps = rows(args.graph_dir / "cross_layer_neighbors.csv")
    nodal_groups = rows(args.graph_dir / "nodal_groups.csv")
    main_components = [row for row in components if int(row["k"]) == 16]
    rep = metadata["representative"]
    match = lambda row: row["sample_id"] == rep["sample_id"] and all(
        int(row[key]) == rep[key] for key in ("denoiser_call", "block_1based"))
    representative = next(row for row in main_components if match(row))
    groups = [row for row in nodal_groups if match(row) and int(row["k"]) == 16]
    kc_histogram = Counter(int(row["k_c_tested"]) for row in connectivity_summary["snapshots"])
    by_snapshot = defaultdict(list)
    for row in fiedler:
        if row["fiedler_value"]:
            by_snapshot[row["snapshot_id"]].append((int(row["k"]), float(row["fiedler_value"])))
    downward_curves = sum(any(y2 < y1 - 1e-12 for (_, y1), (_, y2) in zip(sorted(values), sorted(values)[1:]))
                          for values in by_snapshot.values())
    selected_attrs = [row for row in attrs if int(row["k"]) == 16]
    attribute_means = {field: mean(selected_attrs, field) for field in
                       ("update_norm_energy_ratio", "residual_energy_ratio", "weighted_update_cosine",
                        "all_pair_update_cosine", "update_cosine_advantage", "grid_energy_ratio")}
    overlap_means = {}
    for source, target in ((4, 14), (14, 24), (4, 24)):
        subset = [row for row in overlaps if int(row["k"]) == 16 and int(row["source_block"]) == source
                  and int(row["target_block"]) == target]
        overlap_means[f"{source}->{target}"] = mean(subset, "mean_directed_neighbor_overlap")
    absent_main = sum(not row["selected_eigenvalue_index"] for row in main_components)
    absent_all = sum(not row["selected_eigenvalue_index"] for row in components)
    optional_path = args.graph_dir / "optional_clustering.csv"
    optional_rows = rows(optional_path) if optional_path.exists() else []
    optional_rep = next((row for row in optional_rows if match(row)), None)
    comparable_optional = [row for row in optional_rows if row["comparable_two_group_partition"] == "True"]
    optional_trajectory_values = defaultdict(list)
    for row in comparable_optional:
        optional_trajectory_values[row["sample_id"]].append(float(row["partition_ARI"]))
    optional_ari_mean = float(np.mean([np.mean(values) for values in optional_trajectory_values.values()])) if optional_trajectory_values else None
    with np.load(args.graph_dir / representative["arrays_file"]) as component_arrays:
        nodal_axes = [int(index) for index in component_arrays["nodal_embedding_eigenvector_indices_1based"]]
    axis_tex = ",".join(f"q_{index}" for index in nodal_axes)
    report_tables = ["connectivity.csv", "component_summary.csv", "fiedler_curves.csv",
                     "nodal_groups.csv", "attributes.csv", "cross_layer_neighbors.csv"]
    if optional_rows:
        report_tables.append("optional_clustering.csv")
    summary = {
        "dataset": {"trajectories": metadata["sample_count"], "snapshots": metadata["snapshot_count"],
                    "tokens_per_snapshot": 256, "feature_dimension": 1152},
        "graph_configuration": metadata["configuration"], "representative": representative,
        "connectivity": {"kc_histogram": dict(sorted(kc_histogram.items())),
                         "all_connected_by_k": max(kc_histogram),
                         "connected_at_k2": kc_histogram[2]},
        "components": {"main_k": 16, "main_k_count": len(main_components),
                       "all_detail_k_count": len(components),
                       "main_degree_shapes": dict(Counter(row["degree_shape_description"] for row in main_components)),
                       "main_clustering_shapes": dict(Counter(row["clustering_shape_description"] for row in main_components)),
                       "conditions": condition_rows(main_components)},
        "spectrum": {"curves_with_at_least_one_decrease": downward_curves,
                     "missing_candidate_main_k": absent_main, "missing_candidate_all_detail_k": absent_all},
        "attributes_k16": attribute_means, "cross_layer_neighbors_k16": overlap_means,
        "representative_nodal_groups": groups, "optional_representative": optional_rep,
        "optional_summary": {"n_snapshots": len(optional_rows), "comparable_snapshots": len(comparable_optional),
                             "comparable_trajectories": len(optional_trajectory_values),
                             "mean_partition_ARI_across_available_trajectory_averages": optional_ari_mean},
        "provenance": {"input_manifest_sha256": metadata["input_manifest_sha256"],
                       "graph_index_sha256": metadata["index_sha256"],
                       "tables_sha256": {name: hashlib.sha256((args.graph_dir / name).read_bytes()).hexdigest()
                                         for name in report_tables}},
        "limitations": ["descriptive results for 50 trajectories from ten fixed class IDs",
                        "50 trajectories, not 115200 independent replicates",
                        "no token semantic ground truth; nominal grid overlay is illustrative",
                        "no correction reconstruction, rollout quality or inference-speed measurement",
                        "permutation tests exploratory, not multiplicity-adjusted",
                        "raw activations were saved in float16; float64 arithmetic does not recover lost precision"],
    }
    (args.report_dir / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    e2 = r"""For each fixed trajectory, denoiser call, and block, we treat the 256 token observations
as a separate dataset and use their original block-input activations
$f_i=h_i^{\mathrm{in}}\in\mathbb R^{1152}$. We compute the weighted all-to-all interaction
\[
S_{ij}=\exp\!\left(-\frac{\|f_i-f_j\|_2^2}{2\sigma^2}\right),\quad i\ne j,
\qquad S_{ii}=0,
\]
where $\sigma$ is the median positive Euclidean distance over unordered token pairs
in that snapshot. Thus $S$ is symmetric, nonnegative, and has no self-loops.
We construct 450 matrices of size $256\times256$, each with 32,640 unordered
off-diagonal pairs. Computation uses float64 on the saved float16 activations.
All 256 activation rows are distinct within every snapshot, so each observation
maps to one graph vertex without merging duplicate feature vectors.
Updates, positions, and class IDs do not enter the interaction calculation.
The bandwidth is held fixed when changing $k$. This is an activation-distance
affinity, not a model attention-weight matrix. Because the bandwidth is the
median distance, the median affinity is approximately $\exp(-1/2)$ by construction;
this should not be interpreted as an empirical structural discovery."""
    e3 = rf"""For each token $i$, we select the $k$ largest off-diagonal affinities,
equivalently the $k$ smallest Euclidean distances. Exact ties are resolved by
token index. Let $B_{{ij}}^{{(k)}}$ indicate that $j$ is selected by $i$. We use
\[
A_{{ij}}^{{(k)}}=\mathbf{{1}}\{{B_{{ij}}^{{(k)}}=1\ \mathrm{{or}}\ B_{{ji}}^{{(k)}}=1\}},
\qquad W_{{ij}}^{{(k)}}=S_{{ij}}A_{{ij}}^{{(k)}}.
\]
This union symmetrization gives an undirected, weighted kNN graph. Connected
components are computed on its unweighted support. We scan every integer
$k=2,\ldots,64$, producing 28,350 graph records. Of 450 snapshots,
{kc_histogram[2]} are connected at $k=2$, and all are connected by $k={max(kc_histogram)}$.
Here $k_c$ is the smallest connected value within the scanned range; graphs may
already be connected at $k=1$. Component counts are nonincreasing because the
edge sets are nested. At $k=2$, the mean component counts are 3.08 for call 25/block 24
and 3.88 for call 40/block 24. The fixed representative snapshot is connected
throughout the scanned range."""
    e4 = rf"""We predefine a large component as one with at least 32 vertices.
Detailed component analysis is performed at $k=2,4,8,16,32,64$, with $k=16$
as the main setting. Every main-setting graph has one 256-vertex component.
For each component, the unweighted degree is $d_i^{{(0)}}=\sum_j A_{{ij}}$ and
the local clustering coefficient is
$C_i=2T_i/[d_i^{{(0)}}(d_i^{{(0)}}-1)]$, where $T_i$ counts triangles incident
to $i$. We plot both empirical distributions for all {len(components)} large-component cases.
Union symmetrization permits degrees larger than $k$.
For the representative at $k=16$, degrees range from {float(representative['degree_min']):.0f}
to {float(representative['degree_max']):.0f}, with mean {float(representative['degree_mean']):.3f}
and skewness {float(representative['degree_skewness']):.3f}; the distribution is right-skewed.
Its clustering coefficients range from {float(representative['clustering_min']):.3f}
to {float(representative['clustering_max']):.3f}, with mean
{float(representative['clustering_mean']):.3f} and mild positive skewness
{float(representative['clustering_skewness']):.3f}.
Across the 450 main graphs, all degree distributions have sample skewness above 0.5;
315 clustering distributions have absolute skewness at most 0.5 and 135 have
positive skewness above 0.5. These are empirical shape descriptions;
we do not claim a fitted normal, Poisson, or power-law family."""
    j = int(representative["selected_eigenvalue_index"])
    e5 = rf"""For each large component, we use the retained Gaussian weights and form
\[
D=\operatorname{{diag}}(W\mathbf{{1}}),\qquad
\widehat L=I-D^{{-1/2}}WD^{{-1/2}}.
\]
We record $\lambda_2(k)$ whenever the whole snapshot graph is connected and
plot the 32 smallest eigenvalues for every detailed component. In {downward_curves}/450
curves, $\lambda_2(k)$ has at least one decrease, so normalized-Laplacian
connectivity is not assumed monotone with $k$.
The baseline three-dimensional invariant subspace is spanned by $q_2,q_3,q_4$.
To choose a nodal eigenvector, we search indices $j=2,\ldots,\min(32,|G|-1)$ for a positive
eigenvalue with both adjacent gaps at least $10^{{-6}}$ and
$\min(\lambda_j-\lambda_{{j-1}},\lambda_{{j+1}}-\lambda_j)/\lambda_j\ge0.05$.
Among qualifying candidates, we maximize the minimum adjacent gap. The next
eigenvalue beyond the displayed spectrum is also checked when necessary.
For the representative, $\lambda_2={float(representative['fiedler_value']):.6f}$ and
the selected eigenpair is $\lambda_{j}={float(representative['selected_eigenvalue']):.6f}$,
with adjacent gaps {float(representative['left_eigenvalue_gap']):.6f} and
{float(representative['right_eigenvalue_gap']):.6f}.
Its $q_{j}$ produces {int(representative['n_nodal_domains'])} strict nodal domains.
Each domain is a connected component of the positive or negative induced
subgraph, rather than simply all vertices with the same sign. Numerical zeros
are marked separately using tolerance $10^{{-10}}\|q_j\|_\infty$.
We color the vertices in the embedding spanned by ${axis_tex}$ and on the
16-by-16 token grid.

At $k=16$, {absent_main}/450 graphs have no qualifying eigenvalue; across all detailed
components, {absent_all}/{len(components)} lack one. These cases are explicitly reported
without manufacturing a nodal partition.
The representative domains contain {groups[0]['n_tokens']} and {groups[1]['n_tokens']} tokens,
with mean update norms {float(groups[0]['update_norm_mean']):.2f} and
{float(groups[1]['update_norm_mean']):.2f}. Their within-domain standard deviations
are {float(groups[0]['update_norm_std']):.2f} and {float(groups[1]['update_norm_std']):.2f};
the partition explains only {100*float(representative['nodal_update_norm_explained_variance']):.2f}\%
of update-norm variance. The spatial overlay shows broad organization and
does not accurately trace the object boundary. It is a nominal latent-token
grid correspondence, not pixel-level semantic ground truth."""
    if optional_rep:
        e6 = rf"""As an optional baseline, we apply two-cluster k-means directly to the
original activation vectors, using five fixed-seed k-means++ initializations
and 50 iterations per initialization, then retain the lowest-inertia solution.
The graph remains unchanged. For the representative, the k-means partition
has normalized cut {float(optional_rep['kmeans_normalized_cut']):.6f}.
For the selected two-domain nodal partition, normalized cut is
{float(optional_rep['nodal_normalized_cut']):.6f} and the adjusted Rand index
between the two partitions is {float(optional_rep['partition_ARI']):.4f}.
These compare unsupervised partitions; they do not measure semantic accuracy.
Objective/partition comparisons are reported only where a full two-domain
nodal partition is available, to keep the number of groups comparable.
This applies to {len(comparable_optional)}/{len(optional_rows)} snapshots, spanning
{len(optional_trajectory_values)} trajectories. Averaging the available comparable
snapshots within each trajectory, then across trajectories, gives a mean
partition ARI of {optional_ari_mean:.3f}. This summary is conditional on
the availability of a two-domain nodal partition. The chosen nodal eigenvector
uses the separation rule, rather than minimizing normalized cut."""
    else:
        e6 = "This optional clustering comparison was not included in the current run."
    e7 = rf"""The graph structure varies with depth and denoising stage. At $k=16$,
mean clustering rises from 0.303 at call 10/block 4 to 0.503 at call 40/block 24.
The corresponding mean Fiedler values are 0.373 and 0.053. Together with
greater fragmentation at small $k$, this suggests stronger local organization
and weaker global connectivity in several later/deeper settings, although
the trend is not universal.

We also treat recorded updates as graph signals. For scalar $u_i=\|r_i\|_2$
and vector $r_i$, graph energy is $\sum_{{i<j}}W_{{ij}}\|z_i-z_j\|^2$.
We compare it with the exact expectation under uniformly permuting node values
on the same graph:
\[
\mathbb E_{{\pi}} E(z_{{\pi}})=
\frac{{2\sum_{{i<j}} W_{{ij}}}}{{n-1}}
\sum_i\|z_i-\bar z\|^2.
\]
At $k=16$, mean observed-to-null ratios are
{attribute_means['update_norm_energy_ratio']:.3f} for update magnitude,
{attribute_means['residual_energy_ratio']:.3f} for the full update vector, and
{attribute_means['grid_energy_ratio']:.3f} for spatial coordinates.
Edge-weighted update cosine is {attribute_means['weighted_update_cosine']:.3f},
compared with {attribute_means['all_pair_update_cosine']:.3f} over all unordered pairs.
Thus activation-neighbor relationships are associated with update and spatial
organization. These averages first summarize the nine snapshots within each
trajectory, then average over 50 trajectories; they do not treat tokens as
independent replicates.

Mean top-16 neighbor retention across layers is
{overlap_means['4->14']:.3f} for $4\to14$,
{overlap_means['14->24']:.3f} for $14\to24$, and
{overlap_means['4->24']:.3f} for $4\to24$, versus $16/255\approx0.063$
for independent uniform neighbor sets. This motivates later investigation
of cross-layer graph transfer. Spatial coherence may explain part of these
associations. We have not measured correction accuracy, rollout quality,
or inference speed, so these findings do not establish correction or reuse
performance. All results are conditional on this checkpoint, sampler,
ten fixed classes, and saved activation precision."""
    texts = [e2, e3, e4, e5, e6, e7]
    prompts = ["Weighted all-to-all pairwise feature interaction matrix.",
               "kNN graph construction and connected components.",
               "Combinatorial connectivity analysis.", "Laplacian spectral analysis and nodal domains.",
               "Optional clustering comparison.", "Preliminary findings."]
    snippet = ["% Requires amsmath, amssymb and xcolor. Paste each blue Answer into its corresponding item.",
               "% Feature dimension 1152; no PCA. All numerical findings are generated from saved results."]
    for number, (prompt, text) in enumerate(zip(prompts, texts), 2):
        snippet.append(f"\n% E-{number}\n\\item {prompt}\n{{\\color{{blue}}\\textbf{{Answer:}}\n{text}\n}}\n")
    (args.report_dir / "answers_e2_e7.tex").write_text("\n".join(snippet), encoding="utf-8")
    standalone = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,xcolor,graphicx,booktabs,hyperref}
\graphicspath{{reports/figures/}{figures/}}
\title{Graph Analysis of DiT Token Activations: E-2--E-7}
\author{}
\date{}
\begin{document}
\maketitle
We use 50 trajectories, 450 separate token snapshots, and original 1152-dimensional
features. The fixed illustrative snapshot is golden retriever, replicate 0,
denoiser call 25, block 14. The main detailed graph setting is $k=16$;
large components have at least 32 vertices. All six detailed $k$ settings
have per-component results and plots in the generated-data directory.
\begin{enumerate}
\renewcommand{\labelenumi}{E-\arabic{enumi}}
\setcounter{enumi}{1}
\input{answers_e2_e7.tex}
\end{enumerate}
\clearpage
"""
    for filename, caption in (("e2_pairwise_interactions", "E-2: weighted interaction matrix and pair distributions."),
                              ("e3_connectivity_all", "E-3: component-count curves; mean and observed range over 50 trajectories per condition."),
                              ("e4_e5_k16_component00", "E-4/E-5: representative degree, clustering, spectrum, embeddings and nodal domains."),
                              ("e5_fiedler_all", "E-5: mean and observed range of Fiedler values among connected graphs."),
                              ("e5_e7_original_data", "E-5/E-7: nodal domains, final image and measured updates."),
                              ("e6_clustering_comparison", "E-6: two-cluster k-means and selected nodal partitions; comparisons conditional on two complete domains."),
                              ("e7_attribute_smoothness", "E-7: graph-signal energy relative to random node permutations."),
                              ("e7_cross_layer_neighbors", "E-7: same-token neighbor retention across layers.")):
        if not (figure_root / f"{filename}.pdf").exists():
            continue
        standalone += f"\\begin{{figure}}[p]\n\\centering\n\\includegraphics[width=\\linewidth]{{{filename}.pdf}}\n\\caption{{{caption}}}\n\\end{{figure}}\n"
    standalone += r"""\clearpage
\begin{thebibliography}{9}
\bibitem{luxburg} U. von Luxburg, A Tutorial on Spectral Clustering, 2007.
\url{https://www.cs.columbia.edu/~jebara/6772/papers/Luxburg07_tutorial.pdf}
\bibitem{nodal} E. B. Davies, G. M. L. Gladwell, J. Leydold and P. F. Stadler,
Discrete Nodal Domain Theorems, 2001.
\end{thebibliography}
\end{document}
"""
    (args.report_dir / "experiment_report.tex").write_text(standalone)
    with PdfPages(args.report_dir / "experiment_report.pdf") as pdf:
        figure, axis = plt.subplots(figsize=(8.3, 11.7))
        axis.set_axis_off()
        axis.text(0, 0.97, "Graph Analysis of DiT Token Activations", fontsize=18, weight="bold", va="top")
        axis.text(0, 0.9, "Measured E-2--E-7 results\n50 trajectories | 450 snapshots | 256 nodes | 1152 features\n"
                  "Main k=16; detailed k=2,4,8,16,32,64; large components >=32 nodes\n"
                  "Fixed example: golden retriever / r00 / call25 / block14", fontsize=11, va="top", linespacing=1.7)
        table = axis.table(cellText=[[r["call"], r["block"], f"{r['degree_mean']:.3f}",
                                    f"{r['clustering_mean']:.3f}", f"{r['lambda2_mean']:.4f}"]
                                   for r in summary["components"]["conditions"]],
                           colLabels=["Call", "Block", "Mean degree", "Mean clustering", "Mean lambda2"],
                           bbox=[0, 0.37, 1, 0.32])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axis.text(0, 0.31, f"{kc_histogram[2]}/450 connected at k=2; all connected by k={max(kc_histogram)}.\n"
                  f"{absent_main}/450 main graphs lack a qualifying separated simple eigenvalue.\n"
                  f"{downward_curves}/450 normalized Fiedler curves contain a decrease.\n\n"
                  f"Results describe activation graphs and recorded updates.\n"
                  f"They do not establish semantic segmentation, correction accuracy or speedup.\n"
                  f"Exact method, source hashes and per-component data accompany this report.",
                  fontsize=10, va="top", linespacing=1.6)
        pdf.savefig(figure)
        plt.close(figure)
        image_page(pdf, "E-2: weighted pairwise interactions",
                   "All 1152 activation features; Sij=exp(-||hi-hj||^2/(2sigma^2)), diagonal zero. "
                   "Sigma is the per-snapshot median Euclidean distance, fixed over k. "
                   "450 symmetric 256 x 256 matrices; no PCA, updates or spatial attributes enter graph construction.",
                   figure_root / "e2_pairwise_interactions.png")
        image_page(pdf, "E-3: kNN connectivity",
                   "Union of directed kNN edges, original Gaussian edge weights retained. Scan all integers k=2..64. "
                   f"{kc_histogram[2]}/450 connected at k=2; all 450 connected by k={max(kc_histogram)}. Shading is observed range, not a confidence interval. "
                   "k_c is the smallest connected value in the scanned range.", figure_root / "e3_connectivity_all.png")
        image_page(pdf, "E-4 / E-5: representative graph",
                   f"k=16: mean degree {float(representative['degree_mean']):.3f}, "
                   f"mean clustering {float(representative['clustering_mean']):.3f}, "
                   f"lambda2={float(representative['fiedler_value']):.6f}; selected q{j}="
                   f"{float(representative['selected_eigenvalue']):.6f}, {int(representative['n_nodal_domains'])} strict nodal domains. "
                   "Degree is right-skewed; clustering mildly right-skewed. All detailed large components have saved plots.",
                   figure_root / "e4_e5_k16_component00.png")
        image_page(pdf, "E-5: normalized-Laplacian Fiedler curves",
                   "Lsym=I-D^(-1/2)WD^(-1/2). Plot connected snapshots only; their count can vary with k. "
                   f"{downward_curves}/450 curves contain at least one decrease. {absent_main}/450 main graphs have no eigenvalue satisfying "
                   "the predetermined two-sided separation rule; no nodal partition is fabricated for those cases.",
                   figure_root / "e5_fiedler_all.png")
        image_page(pdf, "E-5 / E-7: correspondence with original data",
                   f"Two domains contain {groups[0]['n_tokens']} and {groups[1]['n_tokens']} tokens; mean update norms "
                   f"{float(groups[0]['update_norm_mean']):.2f} and {float(groups[1]['update_norm_mean']):.2f}. "
                   f"Within-domain variation remains large; grouping explains only "
                   f"{100*float(representative['nodal_update_norm_explained_variance']):.2f}% of update-norm variance. "
                   "The nominal token-grid overlay does not follow the dog's semantic boundary accurately.",
                   figure_root / "e5_e7_original_data.png")
        if optional_rep and (figure_root / "e6_clustering_comparison.png").exists():
            image_page(pdf, "E-6: optional two-cluster k-means baseline",
                       f"Original 1152 features; five fixed-seed initializations. Normalized cut: "
                       f"k-means {float(optional_rep['kmeans_normalized_cut']):.4f}, "
                       f"nodal {float(optional_rep['nodal_normalized_cut']):.4f}. "
                       f"Partition ARI={float(optional_rep['partition_ARI']):.4f}; this is partition agreement, not semantic accuracy.",
                       figure_root / "e6_clustering_comparison.png")
        image_page(pdf, "E-7: graph signals and measured updates",
                   f"Mean energy/null ratios at k=16: update magnitude {attribute_means['update_norm_energy_ratio']:.3f}, "
                   f"vector update {attribute_means['residual_energy_ratio']:.3f}, spatial coordinates {attribute_means['grid_energy_ratio']:.3f}. "
                   f"Edge-weighted update cosine {attribute_means['weighted_update_cosine']:.3f} vs "
                   f"all-pairs {attribute_means['all_pair_update_cosine']:.3f}. Ratios below 1 indicate stronger graph smoothness "
                   "than random node assignment. Differences between conditions remain substantial.",
                   figure_root / "e7_attribute_smoothness.png")
        image_page(pdf, "E-7: preliminary cross-layer structure",
                   f"Top-16 neighbor retention: 4->14={overlap_means['4->14']:.3f}; "
                   f"14->24={overlap_means['14->24']:.3f}; 4->24={overlap_means['4->24']:.3f}. "
                   "Uniform independent-set reference=16/255=0.063. This is descriptive structural evidence, "
                   "potentially partly due to spatial coherence; cross-layer correction performance remains unmeasured.",
                   figure_root / "e7_cross_layer_neighbors.png")
    print(f"Wrote LaTeX answers, PDF report and results summary under {args.report_dir}")


if __name__ == "__main__":
    main()
