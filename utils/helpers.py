import torch

from hyperbolic_math.src.manifolds import Hyperboloid, PoincareBall
from hyperbolic_math.src.utils.math_utils import cosh, sinh


def frechet_center_data_poincare(x: torch.Tensor, mean: torch.Tensor, poincare: PoincareBall) -> torch.Tensor:
    # Inverse geometry method adapted for general curvature magnitudes
    # Details: https://github.com/HazyResearch/HoroPCA/blob/main/geom/poincare.py
    # 1) Compute reflection center and squared radius of the inversion sphere
    mean2 = mean.pow(2).sum(dim=-1, keepdim=True)
    reflection_center = mean / (poincare.c * mean2)
    reflection_center2 = reflection_center.pow(2).sum(dim=-1, keepdim=True)
    r2 = reflection_center2 - (1.0 / poincare.c)
    # 2) Perform the inversion
    u = x - reflection_center
    x_centered_inversion = r2 * u / u.pow(2).sum(dim=-1, keepdim=True) + reflection_center
    return x_centered_inversion

def to_hyperboloid_ideals(ideals: torch.Tensor) -> torch.Tensor:
    """
    Convert the orthonormalized PoincareBall ideal point(s) to Hyperboloid ideal point(s).

    Parameters
    ----------
    ideals : torch.Tensor
        Orthonormalized PoincareBall ideal point(s)

    Returns
    -------
    res : torch.Tensor
        The Hyperboloid ideal point(s)
    """
    res = torch.cat([torch.ones_like(ideals[:,:1]), ideals], dim=-1)
    return res

def poincare_geodesic_projection(x: torch.Tensor, Q: torch.Tensor, poincare: PoincareBall) -> torch.Tensor:
    # Projection used by BSA and PGA as implemented in
    # Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
    #     International Conference on Machine Learning (2021).
    # 1) Point reflection
    ref = 2 * Q.transpose(0, 1) @ Q - torch.eye(x.shape[-1], device=x.device)
    x_ = x @ ref
    # 2) Hyperbolic Midpoint
    t1 = poincare.addition(-x, x_)
    t2 = poincare.scalar_mul(torch.tensor([0.5]), t1)
    proj = poincare.addition(x, t2)
    return proj

def minkowski_orthogonal_projection(Q: torch.Tensor, x: torch.Tensor, hyperboloid: Hyperboloid) -> torch.Tensor:
    # Projection used by BSA as implemented in
    # Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
    #     International Conference on Machine Learning (2021).
    # torch.solve is deprecated -- replaced with torch.linalg.solve which has arguments reversed
    coefs = torch.linalg.solve(
        hyperboloid._minkowski_inner(Q.unsqueeze(-1), Q.T.unsqueeze(0), axis=1).squeeze(1),
        hyperboloid._minkowski_inner(x.unsqueeze(-1), Q.T.unsqueeze(0), axis=1).squeeze(1),
        left=False
    )
    return coefs @ Q

def euclidean_orthonormal(Q: torch.Tensor) -> torch.Tensor:
    k = Q.size(-2)
    _, _, v = torch.svd(Q, some=False)
    Q_ = v[:, :k]
    return Q_.transpose(-1, -2)

def hyperboloid_horo_projection(x: torch.Tensor, ideals: torch.Tensor, hyperboloid: Hyperboloid) -> torch.Tensor:
    # Compute orthogonal (geodesic) projection from x to the geodesic submanifold spanned by ideals
    # We call this submanifold the "spine" because of the "open book" intuition
    minkowski_proj = minkowski_orthogonal_projection(ideals, x, hyperboloid)  # shape (batch_size, Minkowski_dim)
    squared_norms = hyperboloid._minkowski_inner(minkowski_proj, minkowski_proj)  # shape (batch_size, )
    spine_ortho_proj = minkowski_proj / torch.sqrt(- squared_norms)
    spine_dist = hyperboloid.dist(spine_ortho_proj, x)  # shape (batch_size, )

    # poincare_origin = [1,0,0,0,...], # shape (Minkowski_dim, )
    poincare_origin = torch.zeros(x.shape[1], device=x.device)
    poincare_origin[0] = 1

    # Find a tangent vector of the hyperboloid at spine_ortho_proj that is tangent to the target submanifold
    # and orthogonal to the spine.
    # This is done in a Gram-Schmidt way: Take the Euclidean vector pointing from spine_ortho_proj to poincare_origin,
    # then subtract a projection part so that it is orthogonal to the spine and tangent to the hyperboloid
    # Everything below has shape (batch_size, Minkowski_dim)
    chords = poincare_origin - spine_ortho_proj
    tangents = chords - minkowski_orthogonal_projection(ideals, chords, hyperboloid)

    unit_tangents = tangents / hyperboloid._minkowski_inner(tangents, tangents).sqrt()
    # Exponential map with unit tangents
    proj = spine_ortho_proj * cosh(spine_dist) + unit_tangents * sinh(spine_dist)
    return proj
