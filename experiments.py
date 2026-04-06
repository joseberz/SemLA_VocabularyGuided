import argparse
import copy
import json
import os
from itertools import product

import numpy as np
import yaml
import scipy
import scipy.spatial
from typing import Dict, List, Callable, Any, Optional, Tuple

from scipy.stats import hmean

from domain_orchestrator.domain_orchestrator import DomainOrchestrator
from domain_orchestrator.embedding import EmbeddingManager, OpenClipEmbeddingModel, VocabEmbeddingMethod

from bayes_opt import BayesianOptimization

EXCLUDE_FROM_HMEAN = ["coconutL"]

# Define distance measure mappings
NAME_MEASURE_MAPPING = {
    "euclidean": lambda u, v: 1. / scipy.spatial.distance.euclidean(u.squeeze(), v.squeeze()),
    "cosine": lambda u, v: scipy.spatial.distance.cosine(u.squeeze(), v.squeeze()),
}

def load_domains_from_yaml(file_path: str) -> List[str]:
    """Load domains from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """Load configuration parameters from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results: Dict, weights: Optional[Dict] = None, output_dir: str = "./results") -> None:
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
    
    print(f"Results saved to {abs_output_dir}")

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

def semla_merge(target_domains: List[str],
                config: Dict[str, Any], remove_target_adapter: bool, 
                output_dir: str,
                orchestrator: DomainOrchestrator,
                vocab_embedding_method: VocabEmbeddingMethod = VocabEmbeddingMethod.NONE,
                optimize: bool = False) -> None:
    """Run online merge experiment."""
    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    temperature = config.get("temperature", 0.05)
    top_k = config.get("top_k", 5)
    combination_type = config.get("combination_type", "cat")
    top_q = config.get("top_q", 5)
    gamma = config.get("gamma", 0.5)

    if not optimize:
        orchestrator = orchestrator
        results, weights = orchestrator.benchmark_semla(
            target_domains=target_domains,
            remove_target_adapter=remove_target_adapter,
            similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
            softmax_temperature=temperature,
            top_k=top_k,
            combination_type=combination_type,
            vocab_embedding_method=vocab_embedding_method,
            top_q=top_q,
            gamma=gamma
        )
        save_results(results, weights, output_dir=output_dir)
    elif optimize and vocab_embedding_method == VocabEmbeddingMethod.NONE.value:
        base_output_dir = output_dir
        def objective(top_k_opt, temperature_opt):
            top_k_opt = int(round(top_k_opt))

            results, weights = orchestrator.benchmark_semla(
                target_domains=target_domains,
                remove_target_adapter=remove_target_adapter,
                similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
                softmax_temperature=temperature_opt,
                top_k=top_k_opt,
                combination_type=combination_type,
                vocab_embedding_method=vocab_embedding_method
            )
            print(results)
            run_name = f"top_k_{top_k_opt}_tau_{temperature_opt}"
            run_output_dir = os.path.join(base_output_dir, run_name)
            save_results(results, weights, output_dir=run_output_dir)

            values = [v for k, v in results.items() if k not in EXCLUDE_FROM_HMEAN]

            return hmean(values)

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                "top_k_opt":      (3, 9),
                "temperature_opt": (0.003, 0.1),
            },
            random_state=42,
        )
        optimizer.maximize(
            init_points=4,
            n_iter=7,
        )

        print(optimizer.max)  # beste Parameter
    else:
        base_output_dir = output_dir
        def objective(top_q_opt, gamma_opt, top_k_opt, temperature_opt):
            top_q_opt = int(round(top_q_opt))
            top_k_opt = int(round(top_k_opt))

            results, weights = orchestrator.benchmark_semla(
                target_domains=target_domains,
                remove_target_adapter=remove_target_adapter,
                similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
                softmax_temperature=temperature_opt,
                top_k=top_k_opt,
                combination_type=combination_type,
                vocab_embedding_method=vocab_embedding_method,
                top_q=top_q_opt,
                gamma=gamma_opt
            )
            print(results)
            run_name = f"top_q_{top_q_opt}_gamma_{gamma_opt}_top_k_{top_k_opt}_tau_{temperature_opt}"
            run_output_dir = os.path.join(base_output_dir, run_name)
            save_results(results, weights, output_dir=run_output_dir)

            values = [v for k, v in results.items() if k not in EXCLUDE_FROM_HMEAN]

            return hmean(values)

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                "top_q_opt":      (3, 9),
                "gamma_opt":      (0.1, 0.9),
                "top_k_opt":      (5, 9),
                "temperature_opt": (0.005, 0.1),
            },
            random_state=42,
        )
        optimizer.maximize(
            init_points=5,
            n_iter=20,
        )

        print(optimizer.max)  # beste Parameter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Domain adaptation experiments")
    
    # Required arguments
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["zeroshot", "oracle", "uniform", "semla", "semla_optimize"],
                        help="Type of experiment to run")
    
    # Optional arguments with defaults
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

    parser.add_argument(
        "--voc_distance_method",
        type=str,
        choices=["none", "global", "patch"],
        default="none"
    )
    
    return parser.parse_args()

def main():
    """Main function to run experiments based on command line arguments."""

    args = parse_args()
    
    # Load source domains
    source_domains = load_domains_from_yaml(args.source_domains) if args.source_domains else []
    
    # Load target domains
    target_domains = load_domains_from_yaml(args.target_domains) if args.target_domains else []
    
    # Load config if provided
    semla_config = load_config_from_yaml(args.semla_config) if args.semla_config else {}

    orchestrator = DomainOrchestrator(source_domains)
    
    # Run the specified experiment
    if args.experiment == "zeroshot":
        benchmark_zeroshot(target_domains, args.output_dir, orchestrator)
    elif args.experiment == "oracle":
        benchmark_oracle(target_domains, args.output_dir, orchestrator)
    elif args.experiment == "uniform":
        uniform_merge(target_domains, args.remove_target_adapter, args.output_dir, orchestrator)
    elif args.experiment == "semla":
        semla_merge(target_domains, semla_config, args.remove_target_adapter, args.output_dir, orchestrator, vocab_embedding_method=args.voc_distance_method, optimize=False)
    elif args.experiment == "semla_optimize":
        semla_merge(target_domains, semla_config, args.remove_target_adapter, args.output_dir, orchestrator, vocab_embedding_method=args.voc_distance_method, optimize=True)

if __name__ == "__main__":
    main()