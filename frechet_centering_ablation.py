import torch
import numpy as np

from hyperbolic_math.src.manifolds import Hyperboloid, PoincareBall
from hyperbolic_math.src.utils.horo_pca import compute_frechet_mean, frechet_center_data
from hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from utils.helpers import frechet_center_data_poincare


##################################################
# Configuration parameters to test
configs = [ # (c, dtype, num_pts, dim)
    # Float32 configurations
    # Curvature: -1
    (torch.tensor([1.0]), 'float32', 30, 2),
    (torch.tensor([1.0]), 'float32', 100, 5),
    (torch.tensor([1.0]), 'float32', 1_000, 10),
    # Curvature: -0.1
    (torch.tensor([0.1]), 'float32', 30, 2),
    (torch.tensor([0.1]), 'float32', 100, 5),
    (torch.tensor([0.1]), 'float32', 1_000, 10),
    # Curvature: -5
    (torch.tensor([5.0]), 'float32', 30, 2),
    (torch.tensor([5.0]), 'float32', 100, 5),
    (torch.tensor([5.0]), 'float32', 1_000, 10),
    # Float64 configurations
    # Curvature: -1
    (torch.tensor([1.0]), 'float64', 30, 2),
    (torch.tensor([1.0]), 'float64', 100, 5),
    (torch.tensor([1.0]), 'float64', 1_000, 10),
    # Curvature: -0.1
    (torch.tensor([0.1]), 'float64', 30, 2),
    (torch.tensor([0.1]), 'float64', 100, 5),
    (torch.tensor([0.1]), 'float64', 1_000, 10),
    # Curvature: -5
    (torch.tensor([5.0]), 'float64', 30, 2),
    (torch.tensor([5.0]), 'float64', 100, 5),
    (torch.tensor([5.0]), 'float64', 1_000, 10),
]
num_runs_per_config = 100
##################################################

def create_data(num_pts, dim, poincare, hyperboloid):
    """Create random points in the PoincareBall and the Hyperboloid."""
    random_dirs = torch.normal(0, 1, size=(num_pts, dim), dtype=poincare.dtype)
    random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
    random_radii = torch.rand((num_pts, 1), dtype=poincare.dtype).pow(1 / dim)
    points_poincare = poincare.c**-0.5 * (random_dirs * random_radii)
    points_hyper = poincare.to_hyperboloid(points_poincare)
    points_hyper = hyperboloid.proj(points_hyper)
    return points_poincare, points_hyper

def validate_centering(x_centered, original_distances, manifold):
    """
    Evaluates the quality of the Fréchet centering operation by computing:
    1. New Mean: The centered data should have a mean close to origin
    2. Isometry: Distances should be preserved (average and worst-case relative errors)
    """
    # 1. Compute the distance of the Fréchet mean to the origin
    mean_centered = compute_frechet_mean(x_centered, manifold)
    new_mean_dist = manifold.dist_0(mean_centered).item()
    # 2. Isometry
    new_distances = compute_pairwise_distances(x_centered, manifold)
    isometry_error_avg, isometry_error_wc = validate_isometry(original_distances, new_distances)
    return new_mean_dist, isometry_error_avg, isometry_error_wc

def validate_isometry(original_distances, new_distances):
    """Computes the worst-case/average relative error between the original and the pairwise distances after Fréchet centering."""
    indices = torch.triu_indices(original_distances.shape[0], original_distances.shape[0], 1).to(original_distances.device)
    abs_diff = torch.abs(original_distances - new_distances)[indices[0], indices[1]]
    rel_diff = abs_diff / (original_distances[indices[0], indices[1]])
    max_error = rel_diff.max().item()
    mean_error = rel_diff.mean().item()
    return mean_error, max_error


print("="*60 + "\nComparing Fréchet Centering: 'HoroPCA' vs 'HoroPCA++'")
print("\n METRICS:")
print(" - Mean Distance to the Origin (Mean): Measures centering quality (lower = better)")
print(" - Average Relative Error (Avg_Rel): Mean isometry preservation (lower = better)")
print(" - Worst-case Relative Error (WC_Rel): Worst isometry preservation (lower = better)\n" + "="*60)
results = []

# Collect ALL individual trial results for overall statistics
all_trials_h_mean = []
all_trials_h_avg_rel = []
all_trials_h_wc_rel = []
all_trials_inv_mean = []
all_trials_inv_avg_rel = []
all_trials_inv_wc_rel = []

# Main evaluation loop. Each configuration's results are averaged over 100 runs/seeds
for config_idx, (c, dtype, num_pts, dim) in enumerate(configs):
    print(f"\nConfiguration {config_idx + 1}/{len(configs)}: curvature={-c.item():.1f}, dtype={dtype}, num_pts={num_pts}, dim={dim}")

    # Deterministic objects per configuration
    poincare = PoincareBall(c=c, dtype=dtype)
    hyperboloid = Hyperboloid(c=c, dtype=dtype)

    # Track metrics across trials
    poincare_new_mean_dists_inversion = []
    poincare_avg_rel_errors_inversion = []
    poincare_wc_rel_errors_inversion = []

    hyperboloid_new_mean_dists = []
    hyperboloid_avg_rel_errors = []
    hyperboloid_wc_rel_errors = []

    for seed in range(num_runs_per_config):
        torch.manual_seed(seed)
        # Create random data
        points_poincare, points_hyperboloid = create_data(num_pts, dim, poincare, hyperboloid)

        # Compute the original pairwise distances
        original_distances_poincare = compute_pairwise_distances(points_poincare, poincare)
        original_distances_hyperboloid = compute_pairwise_distances(points_hyperboloid, hyperboloid)

        # Compute the Fréchet means
        mean_poincare = compute_frechet_mean(points_poincare, poincare)
        mean_hyperboloid = compute_frechet_mean(points_hyperboloid, hyperboloid)

        # Fréchet-center the data with different methods and validate them
        x_centered_inversion = frechet_center_data_poincare(points_poincare, mean_poincare, poincare)
        # PoincareBall: Fréchet-centering validation [Inverse Geometry method (HoroPCA)]
        new_mean_dist_inversion, avg_rel_error_inversion, wc_rel_error_inversion = validate_centering(x_centered_inversion, original_distances_poincare, poincare)
        poincare_new_mean_dists_inversion.append(new_mean_dist_inversion)
        poincare_avg_rel_errors_inversion.append(avg_rel_error_inversion)
        poincare_wc_rel_errors_inversion.append(wc_rel_error_inversion)
        # Hyperboloid: Fréchet-centering validation [Direct method expressed as Lorentz transformations (ours - HoroPCA++)]
        # Lorentz transformation: Weize Chen, et al. "Fully hyperbolic neural networks.", arXiv preprint arXiv:2105.14686 (2021).
        x_centered_hyperboloid = frechet_center_data(points_hyperboloid, mean_hyperboloid, hyperboloid)
        hyperboloid_new_mean_dist, hyperboloid_avg_rel_error, hyperboloid_wc_rel_error = validate_centering(x_centered_hyperboloid, original_distances_hyperboloid, hyperboloid)
        hyperboloid_new_mean_dists.append(hyperboloid_new_mean_dist)
        hyperboloid_avg_rel_errors.append(hyperboloid_avg_rel_error)
        hyperboloid_wc_rel_errors.append(hyperboloid_wc_rel_error)

    # Add all trials from this config to the global lists
    all_trials_h_mean.extend(hyperboloid_new_mean_dists)
    all_trials_h_avg_rel.extend(hyperboloid_avg_rel_errors)
    all_trials_h_wc_rel.extend(hyperboloid_wc_rel_errors)
    all_trials_inv_mean.extend(poincare_new_mean_dists_inversion)
    all_trials_inv_avg_rel.extend(poincare_avg_rel_errors_inversion)
    all_trials_inv_wc_rel.extend(poincare_wc_rel_errors_inversion)

    # Compute statistics (mean and standard deviation)
    avg_mean_dist_p_inv = np.mean(poincare_new_mean_dists_inversion)
    std_mean_dist_p_inv = np.std(poincare_new_mean_dists_inversion, ddof=1)
    avg_avg_rel_p_inv = np.mean(poincare_avg_rel_errors_inversion)
    std_avg_rel_p_inv = np.std(poincare_avg_rel_errors_inversion, ddof=1)
    avg_wc_rel_p_inv = np.mean(poincare_wc_rel_errors_inversion)
    std_wc_rel_p_inv = np.std(poincare_wc_rel_errors_inversion, ddof=1)

    avg_mean_dist_h = np.mean(hyperboloid_new_mean_dists)
    std_mean_dist_h = np.std(hyperboloid_new_mean_dists, ddof=1)
    avg_avg_rel_h = np.mean(hyperboloid_avg_rel_errors)
    std_avg_rel_h = np.std(hyperboloid_avg_rel_errors, ddof=1)
    avg_wc_rel_h = np.mean(hyperboloid_wc_rel_errors)
    std_wc_rel_h = np.std(hyperboloid_wc_rel_errors, ddof=1)

    # Store results for summary
    results.append({
        'config': f"curvature={-c.item()}, dtype={dtype}, n={num_pts}, d={dim}",
        'h_mean': avg_mean_dist_h,
        'h_mean_std': std_mean_dist_h,
        'h_avg_rel': avg_avg_rel_h,
        'h_avg_rel_std': std_avg_rel_h,
        'h_wc_rel': avg_wc_rel_h,
        'h_wc_rel_std': std_wc_rel_h,
        'inv_mean': avg_mean_dist_p_inv,
        'inv_mean_std': std_mean_dist_p_inv,
        'inv_avg_rel': avg_avg_rel_p_inv,
        'inv_avg_rel_std': std_avg_rel_p_inv,
        'inv_wc_rel': avg_wc_rel_p_inv,
        'inv_wc_rel_std': std_wc_rel_p_inv,
    })

    # Print header if first config
    if config_idx == 0:
        print(f"\n  {'Method':<12} {'Mean Distance':<22} {'Avg Rel Error':<25} {'WC Rel Error':<25}")
        print(f"  {'-'*12} {'-'*22}  {'-'*22} {'-'*23}")
    
    print(f"  {'HoroPCA++':<12} {avg_mean_dist_h:.2e} (+/-{std_mean_dist_h:.2e})   {avg_avg_rel_h:.2e} (+/-{std_avg_rel_h:.2e})   {avg_wc_rel_h:.2e} (+/-{std_wc_rel_h:.2e})")
    print(f"  {'HoroPCA':<12} {avg_mean_dist_p_inv:.2e} (+/-{std_mean_dist_p_inv:.2e})   {avg_avg_rel_p_inv:.2e} (+/-{std_avg_rel_p_inv:.2e})   {avg_wc_rel_p_inv:.2e} (+/-{std_wc_rel_p_inv:.2e})")

# Compute overall statistics across all configs and seeds
print("\n" + "="*80)
print("OVERALL SUMMARY (aggregated across all configurations and seeds)")
print("="*80)

# Compute overall statistics from ALL individual trials
overall_h_mean = np.mean(all_trials_h_mean)
overall_h_mean_std = np.std(all_trials_h_mean, ddof=1)
overall_h_avg_rel = np.mean(all_trials_h_avg_rel)
overall_h_avg_rel_std = np.std(all_trials_h_avg_rel, ddof=1)
overall_h_wc_rel = np.mean(all_trials_h_wc_rel)
overall_h_wc_rel_std = np.std(all_trials_h_wc_rel, ddof=1)

overall_inv_mean = np.mean(all_trials_inv_mean)
overall_inv_mean_std = np.std(all_trials_inv_mean, ddof=1)
overall_inv_avg_rel = np.mean(all_trials_inv_avg_rel)
overall_inv_avg_rel_std = np.std(all_trials_inv_avg_rel, ddof=1)
overall_inv_wc_rel = np.mean(all_trials_inv_wc_rel)
overall_inv_wc_rel_std = np.std(all_trials_inv_wc_rel, ddof=1)

print(f"\n  {'Method':<12} {'Mean Distance':<22} {'Avg Rel Error':<25} {'WC Rel Error':<25}")
print(f"  {'-'*12} {'-'*22}  {'-'*22} {'-'*23}")
print(f"  {'HoroPCA++':<12} {overall_h_mean:.2e} (+/-{overall_h_mean_std:.2e})   {overall_h_avg_rel:.2e} (+/-{overall_h_avg_rel_std:.2e})   {overall_h_wc_rel:.2e} (+/-{overall_h_wc_rel_std:.2e})")
print(f"  {'HoroPCA':<12} {overall_inv_mean:.2e} (+/-{overall_inv_mean_std:.2e})   {overall_inv_avg_rel:.2e} (+/-{overall_inv_avg_rel_std:.2e})   {overall_inv_wc_rel:.2e} (+/-{overall_inv_wc_rel_std:.2e})")
print("\n" + "="*80)
