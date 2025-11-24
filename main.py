import hydra
import torch
import traceback
import wandb
import networkx as nx
import numpy as np
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

from original_horopca import BSA, PGA, HoroPCA
from utils.evaluation import evaluation
from utils.dataloader import load_graph, load_poincare_embeddings
from utils.sarkar import sarkar, pick_root
from hyperbolic_math.src.manifolds import Hyperboloid, PoincareBall
from hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from hyperbolic_math.src.utils.horo_pca import compute_frechet_mean, frechet_center_data
from hyperbolic_math.src.utils.horo_pca import HoroPCA as HoroPCA_ours


@hydra.main(version_base=None, config_path="configs", config_name="config")
@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def main(cfg: DictConfig) -> None:
    print(cfg, flush=True)
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode='online', config=OmegaConf.to_container(cfg, resolve=True))
    #TODO: mode=    disabled / 'online'
    #TODO: python main.py -m seed=44,45,46,47,48 dataset.name=smalltree,phylo_tree,bio_diseasome,ca_CSphd  method=HoroPCA dtype=float32,float64 +poincare_loss=False
    try:
        if torch.cuda.is_available():
            torch.set_default_device('cuda:0')

        poincare = PoincareBall(c=torch.tensor(cfg.curvature), dtype=cfg.dtype)
        hyperboloid = Hyperboloid(c=torch.tensor(cfg.curvature), dtype=cfg.dtype)

        # Load a graph dataset
        graph = load_graph(cfg.dataset.name)
        n_nodes = graph.number_of_nodes()
        nodelist = np.arange(n_nodes)
        graph_dist = torch.from_numpy(nx.floyd_warshall_numpy(graph, nodelist=nodelist)).to(torch.get_default_device())

        # Get hyperbolic embeddings
        if cfg.dataset.use_sarkar:
            # Embed with Sarkar
            root = pick_root(graph)
            z = sarkar(graph, tau=cfg.dataset.sarkar_scale, root=root, dim=cfg.original_dim)
            z = torch.from_numpy(z).to(torch.get_default_device())
            z_dist = compute_pairwise_distances(z, poincare) / cfg.dataset.sarkar_scale
        else:
            # Load pre-trained embeddings
            assert cfg.original_dim in [2, 10, 50], "pretrained embeddings are only for 2, 10 and 50 dimensions"
            z = load_poincare_embeddings(cfg.dataset.name, dim=cfg.original_dim)
            z = torch.from_numpy(z).to(torch.get_default_device())
            z_dist = compute_pairwise_distances(z, poincare)

        # Compute the pre-trained/Sarkar embedding distortion
        indices = torch.triu_indices(z_dist.shape[0], z_dist.shape[0], 1)
        abs_diff = torch.abs(z_dist - graph_dist)[indices[0], indices[1]]
        rel_diff = abs_diff / graph_dist[indices[0], indices[1]]
        run.log({"Initial embedding distortion": torch.mean(rel_diff).item()})

        torch.manual_seed(cfg.seed)

        # Center the data via Fréchet mean
        # (Hyperboloid centering is best - see ablation study)
        z_hyper = poincare.to_hyperboloid(z)
        mean_hyper = compute_frechet_mean(z_hyper, hyperboloid)
        x_hyper = frechet_center_data(z_hyper, mean_hyper, hyperboloid)
        x_poincare = hyperboloid.to_poincare(x_hyper)

        if cfg.method == 'BSA':
            # Dimensionality reduction with BSA
            model = BSA(dim=cfg.original_dim, n_components=cfg.target_dim, lr=cfg.learning_rate, max_steps=cfg.max_steps, poincare=poincare)
            model.fit(x_poincare, iterative=False)
            y = model.transform(x_poincare)
            evaluation_results = evaluation(x_poincare, y, poincare, metrics=cfg.evaluation)
        elif cfg.method == 'PGA':
            # Dimensionality reduction with PGA
            model = PGA(dim=cfg.original_dim, n_components=cfg.target_dim, lr=cfg.learning_rate, max_steps=cfg.max_steps, poincare=poincare)
            model.fit(x_poincare, iterative=True)
            y = model.transform(x_poincare)
            evaluation_results = evaluation(x_poincare, y, poincare, metrics=cfg.evaluation)
        elif cfg.method == 'HoroPCA':
            # Dimensionality reduction with original HoroPCA
            model = HoroPCA(dim=cfg.original_dim, n_components=cfg.target_dim, lr=cfg.learning_rate, max_steps=cfg.max_steps, poincare=poincare)
            model.fit(x_poincare, iterative=False)
            y = model.transform(x_poincare)
            evaluation_results = evaluation(x_poincare, y, poincare, metrics=cfg.evaluation)
        elif cfg.method == 'HoroPCA++':
            # Dimensionality reduction with HoroPCA++
            model = HoroPCA_ours(n_components=cfg.target_dim, n_in_features=cfg.original_dim+1, manifold=hyperboloid, lr=cfg.learning_rate, max_steps=cfg.max_steps)
            model.fit(x_hyper, center_data=False)
            #y = model.transform(x_hyper, center_data=False)
            #evaluation_results = evaluation(x_poincare, y, poincare, metrics=cfg.evaluation)
            y_hyper = model.transform(x_hyper, center_data=False)
            evaluation_results = evaluation(x_hyper, y_hyper, hyperboloid, metrics=cfg.evaluation)
            print(evaluation_results)
        else:
            raise ValueError(f"Unknown method: {cfg.method}")

        run.log(evaluation_results)
        wandb.finish()

    except Exception as e:
        wandb.finish(exit_code=-1)
        print(f"Error occurred: {e}", flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
