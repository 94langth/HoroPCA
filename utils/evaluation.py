import torch

from hyperbolic_math.src.manifolds import Manifold
from hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from hyperbolic_math.src.utils.horo_pca import compute_frechet_mean


def evaluation(
    X: torch.Tensor,
    Y: torch.Tensor,
    manifold: Manifold,
    metrics: dict[str, bool] | None = None,
) -> dict:
    """
    Evaluate the performance of a dimensionality reduction method using the specified metrics.

    Parameters
    ----------
    X : torch.Tensor
        Original Embeddings of the data points
    Y : torch.Tensor
        Projected Embeddings of the data points
    manifold : Manifold
        The manifold on which the embeddings are defined
    metrics : dict[str, bool] | None
        Dictionary of metric names and whether to compute them

    Returns
    -------
        dict: A dictionary with metric names as keys and their computed values.
    """
    results = {}
    if metrics is None:
        return results

    # Compute the pairwise distances
    distances_orig = compute_pairwise_distances(X, manifold)
    distances_proj = compute_pairwise_distances(Y, manifold)

    indices = torch.triu_indices(distances_orig.shape[0], distances_orig.shape[0], 1).to(X.device)
    distances_orig = distances_orig[indices[0], indices[1]]
    distances_proj = distances_proj[indices[0], indices[1]]
    abs_diff = torch.abs(distances_proj - distances_orig)
    rel_diff = abs_diff / distances_orig

    if metrics.get("distortion", False):
        # Average Distortion (average relative error)
        results["distortion"] = torch.mean(rel_diff).item()
    if metrics.get("distortion_sq", False):
        # Average Squared Distortion (average squared relative error)
        results["distortion_sq"] = torch.mean(rel_diff.pow(2)).item()
    if metrics.get("worst_case_distortion", False):
        # Worst-case Distortion (maximum relative error)
        results["worst_case_distortion"] = torch.max(rel_diff).item()
    if metrics.get("mse_error", False):
        # Mean Squared Error
        results["mse_error"] = torch.mean(abs_diff.pow(2)).item()
    if metrics.get("avg_L1_error", False):
        # Average L1 Error
        results["avg_L1_error"] = torch.mean(abs_diff).item()
    if metrics.get("frechet_variance", False):
        # Fréchet Variance
        frechet_mean = compute_frechet_mean(Y, manifold)
        distances = manifold.distance(Y, frechet_mean.unsqueeze(0)).pow(2)
        results["frechet_variance"] = torch.mean(distances).item()
    if metrics.get("generalized_variance", False):
        # Generalized Variance
        var = torch.mean(distances_proj ** 2) / distances_proj.shape[0]
        results["generalized_variance"] = var.item()
    return results
