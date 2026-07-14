import argparse
import json
import os
import itertools

import numpy as np
import yaml
import scipy
import scipy.spatial
from typing import Dict, List, Any, Optional

from scipy.stats import hmean

from domain_orchestrator.domain_orchestrator import DomainOrchestrator, NormalizationMethod
from domain_orchestrator.embedding import VocabEmbeddingMethod

from bayes_opt import BayesianOptimization

EXCLUDE_FROM_HMEAN = ["coconutL"]

# Define distance measure mappings
NAME_MEASURE_MAPPING = {
    "euclidean": lambda u, v: 1. / scipy.spatial.distance.euclidean(u.squeeze(), v.squeeze()),
    "cosine": lambda u, v: 1. - scipy.spatial.distance.cosine(u.squeeze(), v.squeeze()),
}

# Hyperparameter options for ablation studies
GRID_TOP_K = [5, 7, 9, 12]
GRID_TAU = [0.001, 0.005, 0.01, 0.05, 0.1]
GRID_GAMMA = [0.3, 0.5, 0.7]

def load_domains_from_yaml(file_path: str) -> List[str]:
    """Load domains from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """Load configuration parameters from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_results(results: Dict, weights: Optional[Dict] = None, correlation_log: Optional[List] = None, output_dir: str = "./results") -> None:
    """Save results and weights to JSON files."""
    
    # Change the current working directory to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(f"Changing current working directory to {root_dir}")
    abs_output_dir = os.path.join(root_dir, output_dir) if not os.path.isabs(output_dir) else output_dir
    
    os.makedirs(abs_output_dir, exist_ok=True)
    
    with open(os.path.join(abs_output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    if weights is not None:
        with open(os.path.join(abs_output_dir, "weights.json"), 'w') as f:
            json.dump(weights, f, indent=4)

    if correlation_log is not None and len(correlation_log) > 0:
        with open(os.path.join(abs_output_dir, "correlation_log.json"), 'w') as f:
            json.dump(correlation_log, f, indent=4, cls=NumpyEncoder)
    
    print(f"Results saved to {abs_output_dir}")

def compute_hmean(results: Dict[str, float]) -> float:
    """calculates the harmonic mean for all domains (excluding EXCLUDE_FROM_HMEAN)"""
    values = [v for k, v in results.items() if k not in EXCLUDE_FROM_HMEAN]
    if not values or any(v < 0 for v in values):
        return 0.0
    return float(hmean(values))

def benchmark_zeroshot(target_domains: List[str],
                      output_dir: str, orchestrator: DomainOrchestrator) -> None:
    """Run zero-shot benchmark experiment."""
    orchestrator = orchestrator
    results = orchestrator.benchmark_zeroshot(target_domains)
    save_results(results, output_dir=output_dir)

def benchmark_oracle(target_domains: List[str],
                    output_dir: str, orchestrator: DomainOrchestrator) -> None:
    """Run oracle benchmark experiment."""
    orchestrator = orchestrator
    results = orchestrator.benchmark_oracle(target_domains=target_domains)
    save_results(results, output_dir=output_dir)

def uniform_merge(target_domains: List[str],
                 remove_target_adapter: bool, output_dir: str,
                 orchestrator: DomainOrchestrator) -> None:
    """Run uniform merge experiment."""
    orchestrator = orchestrator
    results, weights = orchestrator.benchmark_uniform(
        target_domains=target_domains,
        remove_target_adapter=remove_target_adapter,
    )
    save_results(results, weights, output_dir=output_dir)

def qualitative_vocab_expansion(target_domains: List[str], config: Dict[str, Any],
                                 top_x: int, image_paths: List[str],
                                 output_dir: str, orchestrator: DomainOrchestrator,
                                 vocab_embedding_method: VocabEmbeddingMethod,
                                 normalization_method: NormalizationMethod) -> None:
    orchestrator.benchmark_semla_qualitative(
        target_domains=target_domains,
        top_x=top_x,
        image_paths=image_paths,
        output_dir=output_dir,
        top_k=config.get("top_k", 5),
        gamma=config.get("gamma", 0.5),
        top_q=config.get("top_q", 5),
        vocab_embedding_method=vocab_embedding_method,
        normalization_method=normalization_method,
    )

def semla_merge(target_domains: List[str],
                config: Dict[str, Any], remove_target_adapter: bool, 
                output_dir: str,
                orchestrator: DomainOrchestrator,
                vocab_embedding_method: VocabEmbeddingMethod = VocabEmbeddingMethod.NONE,
                normalization_method: NormalizationMethod = NormalizationMethod.L1,
                ) -> None:
    """Run SemLA experiment (single evaluation, does not optimize)."""
    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    temperature = config.get("temperature", 0.05)
    top_k = config.get("top_k", 5)
    combination_type = config.get("combination_type", "cat")
    top_q = config.get("top_q", 5)
    top_q_frac = config.get("top_q_frac", None)
    gamma = config.get("gamma", 0.5)

    results, weights, correlation_log = orchestrator.benchmark_semla(
        target_domains=target_domains,
        remove_target_adapter=remove_target_adapter,
        similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
        softmax_temperature=temperature,
        top_k=top_k,
        combination_type=combination_type,
        vocab_embedding_method=vocab_embedding_method,
        top_q=top_q,
        gamma=gamma,
        save_per_image_log=True,
        normalization_method=normalization_method,
        top_q_frac=top_q_frac
    )
    save_results(results, weights, correlation_log, output_dir=output_dir)


# Bayesian Optimization for the 4 methods (semla/None, global, patch, objectdetection)
def bo_optimize(
        source_domains_path: str,
        target_domains_path: str,
        config: Dict[str, Any],
        output_dir: str,
        vocab_embedding_method: VocabEmbeddingMethod,
        normalize_centroids: bool = True,
        normalization_method: NormalizationMethod = NormalizationMethod.L1,
        top_q_frac: float | None = None,
        top_q_fixed: int | None = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
        val_test_split: float = 0.0,
        val_test_seed: int = 123,
        use_val_portion: bool = True,
        init_points: int = 5,
        n_iter: int = 25,
) -> None:
    source_domains = load_domains_from_yaml(source_domains_path)
    target_domains = load_domains_from_yaml(target_domains_path)
    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    combination_type = config.get("combination_type", "cat")

    base_output_dir = os.path.join(output_dir, f"bo_{vocab_embedding_method.value}")

    orchestrator = DomainOrchestrator(
        source_domains, vocab_embedding_method,
        normalize_centroids=normalize_centroids,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        val_test_split=val_test_split,
        val_test_seed=val_test_seed,
        use_val_portion=use_val_portion,
    )

    # TODO
    if vocab_embedding_method is VocabEmbeddingMethod.NONE:
        pbounds = {
            "top_k_opt": (5, 11),
            "temperature_opt": (0.01, 0.2),
        }
    elif vocab_embedding_method is VocabEmbeddingMethod.GLOBAL:
        pbounds = {
            "top_k_opt": (5, 11),
            "temperature_opt": (0.01, 0.2),
            "gamma_opt": (0.1, 0.9),
        }
    elif vocab_embedding_method is VocabEmbeddingMethod.PATCH or \
            vocab_embedding_method is VocabEmbeddingMethod.OBJECTDETECTION:
        pbounds = {
            "top_k_opt": (5, 11),
            "temperature_opt": (0.01, 0.2), # falls fail: andere freezen und noch mit 0.1 probieren
            "gamma_opt": (0.1, 0.9),
        }
    else:
        raise ValueError(f"Unknown method: {vocab_embedding_method.value}")

    print(f"\nBO: Methode: {vocab_embedding_method.value}, pbounds: {pbounds}")

    def objective(**kwargs):
        top_k = int(round(kwargs["top_k_opt"]))

        tau = kwargs["temperature_opt"]
        gamma = kwargs.get("gamma_opt", 0.5)

        results, weights, _ = orchestrator.benchmark_semla(
            target_domains=target_domains,
            remove_target_adapter=True,
            similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
            softmax_temperature=tau,
            top_k=top_k,
            combination_type=combination_type,
            vocab_embedding_method=vocab_embedding_method,
            top_q=top_q_fixed,
            gamma=gamma,
            normalization_method=normalization_method,
            top_q_frac=top_q_frac
        )

        h = compute_hmean(results)

        if top_q_frac is not None:
            run_name = f"k{top_k}_tau{tau}_g{gamma}_qfrac{top_q_frac}"
        else:
            run_name = f"k{top_k}_tau{tau:.4f}_g{gamma:.4f}_q{top_q_fixed}"
        run_dir = os.path.join(base_output_dir, run_name)
        save_results(results, weights, output_dir=run_dir)

        print(f"BO {run_name} => h-mIoU = {h}")
        return h

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Save best results
    best = optimizer.max
    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, "best_config.json"), "w") as f:
        json.dump(best, f, indent=4)

    print(f"\nBO DONE: Best parameters = {best['params']}, h-mIoU = {best['target']}")


# Ablation 1: normalized vs unnormalized Embeddings on NONE method (only top_k and tau as parameters)
def grid_search_centroid_ablation(
        source_domains_path: str,
        target_domains_path: str,
        config: Dict[str, Any],
        output_dir: str,
        normalize_centroids: bool,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
        val_test_split: float = 0.0,
        val_test_seed: int = 123,
        use_val_portion: bool = True,
) -> None:
    source_domains = load_domains_from_yaml(source_domains_path)
    target_domains = load_domains_from_yaml(target_domains_path)
    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    combination_type = config.get("combination_type", "cat")

    centroid_tag = "centroidsnorm" if normalize_centroids else "centroidsraw"
    base_output_dir = os.path.join(output_dir, f"ablation_{centroid_tag}_none")

    orchestrator = DomainOrchestrator(
        source_domains, VocabEmbeddingMethod.NONE,
        normalize_centroids=normalize_centroids,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        val_test_split=val_test_split,
        val_test_seed=val_test_seed,
        use_val_portion=use_val_portion,
    )

    summary = {}

    for top_k, tau in itertools.product(GRID_TOP_K, GRID_TAU):
        run_name = f"k{top_k}_tau{tau}"
        run_output_dir = os.path.join(base_output_dir, run_name)

        print(f"\nABLATION CENTROID: {run_name} ({centroid_tag})")

        results, weights, _ = orchestrator.benchmark_semla(
            target_domains=target_domains,
            remove_target_adapter=True,
            similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
            softmax_temperature=tau,
            top_k=top_k,
            combination_type=combination_type,
            vocab_embedding_method=VocabEmbeddingMethod.NONE,
        )
        save_results(results, weights, output_dir=run_output_dir)

        h = compute_hmean(results)
        summary[run_name] = {
            "top_k": top_k, "tau": tau,
            "h_miou": h, "results": results,
        }
        print(f"ABLATION CENTROID: {run_name} => h-mIoU = {h}")

    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, "grid_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    best_name = max(summary, key=lambda k: summary[k]["h_miou"])
    best = summary[best_name]
    print(f"\nABLATION BEST: {best_name}: h-mIoU: {best['h_miou']} "
          f"(K: {best['top_k']}, tau: {best['tau']})")


# Ablation 2: l1 / zscore / minmax nromalization of the 2 distance terms.
# Only applicable for GLOBAL / PATCH / OBJECTDETECTION, since D_voc only exists there
def grid_search_normalization_ablation(
        source_domains_path: str,
        target_domains_path: str,
        config: Dict[str, Any],
        output_dir: str,
        vocab_embedding_method: VocabEmbeddingMethod,
        normalization_method: NormalizationMethod,
        normalize_centroids: bool = True,
        top_q_fixed: int | None = None,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
        val_test_split: float = 0.0,
        val_test_seed: int = 123,
        use_val_portion: bool = True,
) -> None:
    if vocab_embedding_method is VocabEmbeddingMethod.NONE:
        raise ValueError(
            "Ablation experiment is not applicable for NONE method"
        )

    source_domains = load_domains_from_yaml(source_domains_path)
    target_domains = load_domains_from_yaml(target_domains_path)
    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    combination_type = config.get("combination_type", "cat")

    norm_name = normalization_method.value
    centroid_tag = "centroidsnorm" if normalize_centroids else "centroidsraw"
    base_output_dir = os.path.join(
        output_dir, f"ablation_norm_{norm_name}_{centroid_tag}_{vocab_embedding_method.value}"
    )

    orchestrator = DomainOrchestrator(
        source_domains, vocab_embedding_method,
        normalize_centroids=normalize_centroids,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        val_test_split=val_test_split,
        val_test_seed=val_test_seed,
        use_val_portion=use_val_portion
    )

    summary = {}

    for top_k, tau, gamma in itertools.product(GRID_TOP_K, GRID_TAU, GRID_GAMMA):
        run_name = f"k{top_k}_tau{tau}_g{gamma}"
        run_output_dir = os.path.join(base_output_dir, run_name)

        print(f"\nABLATION GRID: {run_name} (norm: {norm_name}, {centroid_tag})")

        results, weights, _ = orchestrator.benchmark_semla(
            target_domains=target_domains,
            remove_target_adapter=True,
            similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
            softmax_temperature=tau,
            top_k=top_k,
            combination_type=combination_type,
            vocab_embedding_method=vocab_embedding_method,
            top_q=top_q_fixed,
            gamma=gamma,
            normalization_method=normalization_method,
        )
        save_results(results, weights, output_dir=run_output_dir)

        h = compute_hmean(results)
        summary[run_name] = {
            "top_k": top_k, "tau": tau, "gamma": gamma,
            "h_miou": h, "results": results,
        }
        print(f"ABLATION GRID: {run_name} => h-mIoU: {h}")

    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, "grid_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    best_name = max(summary, key=lambda k: summary[k]["h_miou"])
    best = summary[best_name]
    print(f"\nABLATION BEST: {best_name}: h-mIoU: {best['h_miou']} "
          f"(K: {best['top_k']}, tau: {best['tau']}, gamma: {best['gamma']})")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Domain adaptation experiments")

    parser.add_argument("--experiment", type=str, required=True,
                        choices=[
                            "zeroshot", "oracle", "uniform",
                            "semla",
                            "bo_optimize",
                            "grid_centroid_ablation",
                            "grid_norm_ablation",
                            "qualitative_vocab"
                        ],
                        help="Type of experiment to run")

    parser.add_argument("--source_domains", type=str,
                        help="Path to YAML file containing source domains")
    parser.add_argument("--target_domains", type=str, 
                        help="Path to YAML file containing target domains")
    parser.add_argument("--semla_config", type=str, 
                        help="Path to YAML file containing configuration parameters")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--remove_target_adapter", action="store_true", 
                        help="Whether to remove target adapter")

    parser.add_argument("--voc_distance_method", type=str,
                        choices=["none", "global", "patch", "objectdetection"],
                        default="none")

    parser.add_argument("--normalize_centroids", action="store_true", default=True,
                        help="Normalize centroids")
    parser.add_argument("--no_normalize_centroids", dest="normalize_centroids", action="store_false",
                        help="Do not normalize centroids")
    parser.add_argument("--normalization_method", type=str,
                        choices=["l1", "zscore", "minmax"],
                        default="l1",
                        help="Normalization method for combining the distance terms")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Portion of subset for valid / test split")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Seed for subset sampling")

    # BO parameter
    parser.add_argument("--bo_init_points", type=int, default=5,
                        help="Number of initial random configration for BO")
    parser.add_argument("--bo_n_iter", type=int, default=25,
                        help="Number of BO iterations")

    parser.add_argument("--top_q_frac", type=float, default=None,
                        help="Fester top_q_frac aus dem Pilot-Grid (nicht im BO gesucht)")


    # Val / Test Split
    parser.add_argument("--val_test_split", type=float, default=0.0,
                        help="Portion of the split")
    parser.add_argument("--val_test_seed", type=int, default=123,
                        help="Seed for the split")
    parser.add_argument("--eval_on_test", action="store_true",
                        help="if set to true, the test set is used for evaluation")

    # Für qualitative Auswertung mit erweitertem Vokabular
    parser.add_argument("--top_x", type=int, default=3,
                        help="Anzahl zusätzlicher Klassen pro gewähltem Adapter")
    parser.add_argument("--image_list", type=str, required=False,
                        help="Pfad zu einer txt Datei mit den absoluten Bildpfaden für die qualitative Auswertung")
    return parser.parse_args()

def main():
    """Main function to run experiments based on command line arguments."""

    args = parse_args()

    args.use_val_portion = not args.eval_on_test

    # Load source domains
    source_domains = load_domains_from_yaml(args.source_domains) if args.source_domains else []
    # Load target domains
    target_domains = load_domains_from_yaml(args.target_domains) if args.target_domains else []
    # Load config if provided
    semla_config = load_config_from_yaml(args.semla_config) if args.semla_config else {}

    norm_method = NormalizationMethod(args.normalization_method)

    voc_method = VocabEmbeddingMethod(args.voc_distance_method)

    top_q_fixed = semla_config.get("top_q", None)

    if args.experiment == "zeroshot":
        orchestrator = DomainOrchestrator(source_domains, args.voc_distance_method,
                                          normalize_centroids=args.normalize_centroids,
                                          subset_fraction=args.subset_fraction,
                                          subset_seed=args.subset_seed,
                                          val_test_split=args.val_test_split,
                                          val_test_seed=args.val_test_seed,
                                          use_val_portion=args.use_val_portion)
        benchmark_zeroshot(target_domains, args.output_dir, orchestrator)

    elif args.experiment == "oracle":
        orchestrator = DomainOrchestrator(source_domains, args.voc_distance_method,
                                          normalize_centroids=args.normalize_centroids,
                                          subset_fraction=args.subset_fraction,
                                          subset_seed=args.subset_seed,
                                          val_test_split=args.val_test_split,
                                          val_test_seed=args.val_test_seed,
                                          use_val_portion=args.use_val_portion)
        benchmark_oracle(target_domains, args.output_dir, orchestrator)

    elif args.experiment == "uniform":
        orchestrator = DomainOrchestrator(source_domains, args.voc_distance_method,
                                          normalize_centroids=args.normalize_centroids,
                                          subset_fraction=args.subset_fraction,
                                          subset_seed=args.subset_seed,
                                          val_test_split=args.val_test_split,
                                          val_test_seed=args.val_test_seed,
                                          use_val_portion=args.use_val_portion)
        uniform_merge(target_domains, args.remove_target_adapter, args.output_dir, orchestrator)

    elif args.experiment == "semla":
        orchestrator = DomainOrchestrator(source_domains, voc_method,
                                          normalize_centroids=args.normalize_centroids,
                                          subset_fraction=args.subset_fraction,
                                          subset_seed=args.subset_seed,
                                          val_test_split=args.val_test_split,
                                          val_test_seed=args.val_test_seed,
                                          use_val_portion=args.use_val_portion)
        semla_merge(target_domains, semla_config, args.remove_target_adapter,
                    args.output_dir, orchestrator,
                    vocab_embedding_method=voc_method,
                    normalization_method=norm_method)

    elif args.experiment == "bo_optimize":
        bo_optimize(
            source_domains_path=args.source_domains,
            target_domains_path=args.target_domains,
            config=semla_config,
            output_dir=args.output_dir,
            vocab_embedding_method=voc_method,
            normalize_centroids=args.normalize_centroids,
            normalization_method=norm_method,
            top_q_fixed=top_q_fixed,
            top_q_frac=args.top_q_frac,
            subset_fraction=args.subset_fraction,
            subset_seed=args.subset_seed,
            val_test_split=args.val_test_split,
            val_test_seed=args.val_test_seed,
            use_val_portion=args.use_val_portion,
            init_points=args.bo_init_points,
            n_iter=args.bo_n_iter
        )

    elif args.experiment == "grid_centroid_ablation":
        grid_search_centroid_ablation(
            source_domains_path=args.source_domains,
            target_domains_path=args.target_domains,
            config=semla_config,
            output_dir=args.output_dir,
            normalize_centroids=args.normalize_centroids,
            subset_fraction=args.subset_fraction,
            subset_seed=args.subset_seed,
            val_test_split=args.val_test_split,
            val_test_seed=args.val_test_seed,
            use_val_portion=args.use_val_portion,
        )

    elif args.experiment == "grid_norm_ablation":
        grid_search_normalization_ablation(
            source_domains_path=args.source_domains,
            target_domains_path=args.target_domains,
            config=semla_config,
            output_dir=args.output_dir,
            vocab_embedding_method=voc_method,
            normalization_method=norm_method,
            normalize_centroids=args.normalize_centroids,
            top_q_fixed=top_q_fixed,
            subset_fraction=args.subset_fraction,
            subset_seed=args.subset_seed,
            val_test_split=args.val_test_split,
            val_test_seed=args.val_test_seed,
            use_val_portion=args.use_val_portion
        )

    # python experiments.py --experiment qualitative_vocab --voc_distance_method none  --image_list qualitative_images.txt --top_x 3 --output_dir ./results/qual
    # python experiments.py --experiment qualitative_vocab --voc_distance_method patch --image_list qualitative_images.txt --top_x 3 --output_dir ./results/qual
    elif args.experiment == "qualitative_vocab":
        image_paths = None
        if args.image_list:
            with open(args.image_list, "r") as f:
                if args.image_list.endswith((".yaml", ".yml")):
                    image_paths = yaml.safe_load(f)
                else:
                    image_paths = [line.strip() for line in f if line.strip()]
        orchestrator = DomainOrchestrator(source_domains, voc_method,
                                          normalize_centroids=args.normalize_centroids,
                                          subset_fraction=args.subset_fraction,
                                          subset_seed=args.subset_seed,
                                          val_test_split=args.val_test_split,
                                          val_test_seed=args.val_test_seed,
                                          use_val_portion=args.use_val_portion)
        qualitative_vocab_expansion(target_domains, semla_config,
                                    args.top_x, image_paths,
                                    args.output_dir, orchestrator,
                                    voc_method, norm_method)

if __name__ == "__main__":
    main()