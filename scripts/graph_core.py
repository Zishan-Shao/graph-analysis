"""Deterministic, CPU-only graph construction from token activation features."""

from __future__ import annotations

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_weight_matrix(weights: np.ndarray) -> None:
    """Validate an undirected, nonnegative, loop-free weighted graph."""
    _require(weights.ndim == 2 and weights.shape[0] == weights.shape[1],
             "weights must be a square matrix")
    _require(bool(np.isfinite(weights).all()), "weights contain non-finite values")
    _require(bool((weights >= 0).all()), "weights must be nonnegative")
    _require(bool(np.array_equal(weights, weights.T)), "weights must be symmetric")
    _require(bool((np.diag(weights) == 0).all()), "self-loops must have zero weight")


def pairwise_affinity(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return squared distances, Gaussian affinity, bandwidth and neighbor order.

    Distances use all supplied feature coordinates in float64, with no scaling,
    normalization, PCA, or spatial-coordinate concatenation. Sigma is the median
    of positive Euclidean distances over unordered vertex pairs. Off-diagonal
    affinities are positive; self-affinities are zero. Stable ascending distance
    sorting resolves exact ties by the original token index.

    Exact duplicate feature rows require a datum-to-feature-equivalence-class
    mapping; they are rejected rather than silently treated as separate vertices.
    """
    features = np.asarray(features, dtype=np.float64)
    _require(features.ndim == 2, "features must have shape [tokens, channels]")
    n_vertices, dimension = features.shape
    _require(n_vertices >= 2 and dimension >= 1, "need at least two nonempty feature vectors")
    _require(bool(np.isfinite(features).all()), "features contain non-finite values")
    _require(np.unique(features, axis=0).shape[0] == n_vertices,
             "duplicate feature vectors: construct their equivalence classes before graph analysis")

    # The Gram formulation avoids materializing an N x N x d differences tensor.
    squared_norms = np.einsum("ij,ij->i", features, features)
    distances_squared = squared_norms[:, None] + squared_norms[None, :] - 2 * (features @ features.T)
    distances_squared = (distances_squared + distances_squared.T) / 2
    np.maximum(distances_squared, 0, out=distances_squared)
    np.fill_diagonal(distances_squared, 0)
    upper = distances_squared[np.triu_indices(n_vertices, 1)]
    _require(bool((upper > 0).all()), "distinct features produced a nonpositive pairwise distance")
    sigma = float(np.median(np.sqrt(upper)))
    _require(np.isfinite(sigma) and sigma > 0, "Gaussian bandwidth must be finite and positive")
    affinity = np.exp(-distances_squared / (2 * sigma * sigma))
    np.fill_diagonal(affinity, 0)
    _require(bool((affinity[np.triu_indices(n_vertices, 1)] > 0).all()),
             "Gaussian affinity underflowed to zero; change the bandwidth explicitly")
    validate_weight_matrix(affinity)
    ordering_distances = distances_squared.copy()
    np.fill_diagonal(ordering_distances, np.inf)
    neighbor_order = np.argsort(ordering_distances, axis=1, kind="stable")[:, :n_vertices - 1]
    return distances_squared, affinity, sigma, neighbor_order


def union_knn(affinity: np.ndarray, neighbor_order: np.ndarray, k: int) -> np.ndarray:
    """Retain W_ij=S_ij whenever i selects j OR j selects i among its k neighbors."""
    affinity = np.asarray(affinity, dtype=np.float64)
    neighbor_order = np.asarray(neighbor_order)
    validate_weight_matrix(affinity)
    n_vertices = affinity.shape[0]
    _require(isinstance(k, (int, np.integer)) and 1 <= k < n_vertices,
             "k must be an integer in [1, number of vertices - 1]")
    _require(neighbor_order.shape == (n_vertices, n_vertices - 1), "invalid neighbor-order shape")
    _require(np.issubdtype(neighbor_order.dtype, np.integer), "neighbor indices must be integers")
    _require(bool(((neighbor_order >= 0) & (neighbor_order < n_vertices)).all()),
             "neighbor index out of range")
    _require(bool((neighbor_order != np.arange(n_vertices)[:, None]).all()),
             "self indices must be excluded from neighbor order")
    # Every row must list all other vertices exactly once.
    expected = np.sort((np.arange(n_vertices)[:, None] + 1 + np.arange(n_vertices - 1)) % n_vertices, axis=1)
    _require(bool(np.array_equal(np.sort(neighbor_order, axis=1), expected)),
             "neighbor order must contain every other vertex exactly once")
    directed = np.zeros((n_vertices, n_vertices), dtype=bool)
    directed[np.arange(n_vertices)[:, None], neighbor_order[:, :k]] = True
    support = directed | directed.T
    weights = np.where(support, affinity, 0.0)
    validate_weight_matrix(weights)
    return weights


def connected_components(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic component labels and component sizes for an undirected graph.

    Labels follow increasing first-encountered token index. Boolean adjacency is
    accepted; for weighted input, any strictly positive edge belongs to support.
    """
    adjacency = np.asarray(adjacency)
    validate_weight_matrix(adjacency)
    support = adjacency > 0
    n_vertices = support.shape[0]
    labels = np.full(n_vertices, -1, dtype=np.int64)
    sizes = []
    for root in range(n_vertices):
        if labels[root] >= 0:
            continue
        visited = np.zeros(n_vertices, dtype=bool)
        frontier = np.zeros(n_vertices, dtype=bool)
        frontier[root] = True
        while frontier.any():
            visited |= frontier
            frontier = support[frontier].any(axis=0) & ~visited
        labels[visited] = len(sizes)
        sizes.append(int(visited.sum()))
    return labels, np.asarray(sizes, dtype=np.int64)
