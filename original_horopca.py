import torch

from hyperbolic_math.src.manifolds import Hyperboloid, PoincareBall
from hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from utils.helpers import to_hyperboloid_ideals, poincare_geodesic_projection, minkowski_orthogonal_projection, euclidean_orthonormal, hyperboloid_horo_projection


class PCA(torch.nn.Module):
    """Dimensionality reduction model class."""
    def __init__(self, dim, n_components, lr, max_steps, dtype):
        super(PCA, self).__init__()
        self.dim = dim
        self.n_components = n_components
        self.components = torch.nn.ParameterList(torch.nn.Parameter(torch.randn(1, dim, dtype=dtype)) for _ in range(self.n_components))
        self.max_steps = max_steps
        self.lr = lr

    def _project(self, x, Q):
        raise NotImplementedError

    def compute_loss(self, x, Q):
        raise NotImplementedError

    def gram_schmidt(self):
        def inner(u, v):
            return torch.sum(u * v)

        Q = []
        for k in range(self.n_components):
            v_k = self.components[k][0]
            proj = 0.0
            for v_j in Q:
                v_j = v_j[0]
                coeff = inner(v_j, v_k) / inner(v_j, v_j).clamp_min(1e-15)
                proj += coeff * v_j
            v_k = v_k - proj
            v_k = v_k / torch.norm(v_k).clamp_min(1e-15)
            Q.append(torch.unsqueeze(v_k, 0))
        return torch.cat(Q, dim=0)

    def orthogonalize(self):
        Q = torch.cat([self.components[i] for i in range(self.n_components)])
        res = euclidean_orthonormal(Q)
        return res

    def get_components(self):
        Q = self.gram_schmidt()
        return Q

    def fit_optim(self, x, iterative):
        loss_vals = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if not iterative:
            for _ in range(self.max_steps):
                # Forward pass: compute _projected variance
                Q = self.get_components()
                loss = self.compute_loss(x, Q)
                loss_vals.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1e5)
                optimizer.step()
        else:
            for k in range(self.n_components):
                for _ in range(self.max_steps):
                    # Forward pass: compute _projected variance
                    Q = self.get_components()
                    # Project on first k components
                    loss = self.compute_loss(x, Q[:k + 1, :])
                    loss_vals.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.components[k].data = self.get_components()[k].unsqueeze(0)
                self.components[k].requires_grad = False
        return loss_vals

    def fit(self, x, iterative):
        self.fit_optim(x, iterative)

    def transform(self, x):
        Q = self.get_components()
        x_p = self._project(x, Q)
        Q_orthogonal = self.orthogonalize()
        return x_p @ Q_orthogonal.transpose(0, 1)

class BSA(PCA):
    """
    Default implementation of hyperbolic BSA as specified in https://github.com/HazyResearch/HoroPCA/tree/main
    Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
        International Conference on Machine Learning (2021).
    """
    def __init__(self, dim, n_components, lr, max_steps, poincare: PoincareBall):
        # This implementation corresponds to using the following parameters:
        # ['optim': True, 'iterative': False, 'hyperboloid': True, 'auc': True, 'keep_orthogonal': True]
        super(BSA, self).__init__(dim, n_components + 1, lr, max_steps, poincare.dtype)
        self.poincare = poincare
        self.hyperboloid = Hyperboloid(c=poincare.c, trainable_c=False, dtype=poincare.dtype)

    def _project(self, x, Q):
        proj = poincare_geodesic_projection(x, Q, self.poincare)
        return proj

    def compute_loss(self, x, Q):
        auc = []
        Q = to_hyperboloid_ideals(Q)
        x = self.poincare.to_hyperboloid(x)
        for i in range(1, self.n_components):
            Q_ = Q[:i + 1, :]
            proj = minkowski_orthogonal_projection(Q_, x, self.hyperboloid)
            residual_variance = torch.sum(self.hyperboloid.dist(x, proj) ** 2)
            auc.append(residual_variance)
        return sum(auc)

class PGA(PCA):
    """
    Default implementation of hyperbolic PGA as specified in https://github.com/HazyResearch/HoroPCA/tree/main
    Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
        International Conference on Machine Learning (2021).
    """
    def __init__(self, dim, n_components, lr, max_steps, poincare: PoincareBall):
        # This implementation corresponds to using the following parameters:
        # ['optim': True, 'iterative': True, 'keep_orthogonal': True]
        super(PGA, self).__init__(dim, n_components, lr, max_steps, poincare.dtype)
        self.poincare = poincare

    def _project(self, x, Q):
        proj = poincare_geodesic_projection(x, Q, self.poincare)
        return proj

    def compute_loss(self, x, Q):
        proj = self._project(x, Q)
        sq_distances = self.poincare.dist_0(proj) ** 2
        var = torch.mean(sq_distances)
        return -var

class HoroPCA(PCA):
    """
    Default implementation of HoroPCA as specified in https://github.com/HazyResearch/HoroPCA/tree/main
    Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
        International Conference on Machine Learning (2021).
    """
    def __init__(self, dim, n_components, lr, max_steps, poincare: PoincareBall):
        # This implementation corresponds to using the following parameters:
        # ['optim': True, 'iterative': False, 'hyperboloid': True, 'auc': False, 'keep_orthogonal': True, 'frechet_variance': False]
        super(HoroPCA, self).__init__(dim, n_components, lr, max_steps, poincare.dtype)
        self.poincare = poincare
        self.hyperboloid = Hyperboloid(c=poincare.c, trainable_c=False, dtype=poincare.dtype)

    def _project(self, x, Q):
        hyperboloid_ideals = to_hyperboloid_ideals(Q)
        hyperboloid_x = self.poincare.to_hyperboloid(x)
        hyperboloid_proj = hyperboloid_horo_projection(hyperboloid_x, hyperboloid_ideals, self.hyperboloid)
        proj = self.hyperboloid.to_poincare(hyperboloid_proj)
        return proj

    def compute_variance(self, x):
        distances = compute_pairwise_distances(x, self.poincare)
        var = torch.mean(distances ** 2)
        return var

    def compute_loss(self, x, Q):
        proj = self._project(x, Q)
        var = self.compute_variance(proj)
        return -var
