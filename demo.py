from hydra.core.hydra_config import HydraConfig
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from biological_random_walks.BiologicalRandomWalks import BiologicalRandomWalks
from biological_random_walks.BiologicalRandomWalksWithGeneSim import BiologicalRandomWalksWithGeneSim

import math
import csv


def load_seed_file(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def recall_at_k(ranked_list, truth_set, k, test_size=None):
    top_k = [g for g, _ in ranked_list[:k]]
    hit = len(set(top_k) & truth_set)
    if test_size is not None:
        return hit / min(test_size, len(truth_set)) if len(truth_set) > 0 else 0
    return hit / len(truth_set) if len(truth_set) > 0 else 0


def precision_at_k(ranked_list, truth_set, k):
    top_k = [g for g, _ in ranked_list[:k]]
    hit = len(set(top_k) & truth_set)
    return hit / k if k > 0 else 0


def dcg_at_k(ranked_list, truth_set, k):
    dcg = 0.0
    for i, (g, _) in enumerate(ranked_list[:k]):
        if g in truth_set:
            dcg += 1 / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_list, truth_set, k):
    dcg = dcg_at_k(ranked_list, truth_set, k)
    ideal_hits = min(len(truth_set), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0


def mrr(ranked_list, truth_set):
    for rank, (gene, _) in enumerate(ranked_list, start=1):
        if gene in truth_set:
            return 1.0 / rank
    return 0.0


METHOD_MAP = {
    "original": "experimental_ori",
    "gene_sim2": "experimental_gs",
}

OmegaConf.register_new_resolver(
    "method_name",
    lambda x: METHOD_MAP.get(x, x)   # nếu không có thì giữ nguyên
)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Change output directory

    run_dir = HydraConfig.get().runtime.output_dir

    seed_gene = load_seed_file(cfg.experiment.seed)

    ground_truth = load_seed_file("data_set/gene_ids.txt")

    output_path = os.path.join(run_dir, "result.txt")

    if cfg.method == "experimental_gs":

        gene_similarity_file_path = cfg.experiment.matrix_similarity

        brw = BiologicalRandomWalksWithGeneSim(

            gene_similarity_file_path=gene_similarity_file_path,

            seed_file_path=cfg.experiment.seed,
            seed_set_override=seed_gene,

            secondary_seed_file_path=cfg.experiment.de,

            ppi_file_path=cfg.paths.ppi,
            co_expression_file_path=cfg.experiment.coexpr,

            disease_ontology_file_path=cfg.experiment.disease_ontology,
            map__gene__ontologies_file_path=cfg.paths.ontology_network,
            personalization_vector_creation_policies=["biological", "topological"],

            restart_prob=cfg.params.restart_prob,
            alpha=cfg.params.alpha,
            beta=cfg.params.beta,

            network_weight_flag=False,

            output_file_path=output_path
        )

    else:

        brw = BiologicalRandomWalks(

            seed_file_path=cfg.experiment.seed,
            seed_set_override=seed_gene,

            secondary_seed_file_path=cfg.experiment.de,

            ppi_file_path=cfg.paths.ppi,
            co_expression_file_path=cfg.experiment.coexpr,

            disease_ontology_file_path=cfg.experiment.disease_ontology,
            map__gene__ontologies_file_path=cfg.paths.ontology_network,
            personalization_vector_creation_policies=["biological", "topological"],

            restart_prob=cfg.params.restart_prob,
            alpha=cfg.params.alpha,
            beta=cfg.params.beta,

            output_file_path=output_path
        )

    ranked_list = brw.ranked_list

    k_list = cfg.evaluation.fixed.k

    print(f"Seed size: {len(seed_gene)} | k_list: {k_list}")

    run_metrics = []

    for k in k_list:
        r = recall_at_k(ranked_list, ground_truth, k, len(seed_gene))
        p = precision_at_k(ranked_list, ground_truth, k)
        ndcg = ndcg_at_k(ranked_list, ground_truth, k)

        print(f"Recall@{k}: {r:.4f} | Precision@{k}: {p:.4f} | nDCG@{k}: {ndcg:.4f}")

        run_metrics.extend([r, p, ndcg])

    mrr_score = mrr(ranked_list, ground_truth)
    print(f"MRR: {mrr_score:.4f}")

    run_metrics.append(mrr_score)

    header = ["Run"]

    for k in k_list:
        header += [f"Recall@{k}", f"Precision@{k}", f"nDCG@{k}"]

    header.append("MRR")

    metric_path = os.path.join(run_dir, "metrics.csv")

    with open(metric_path, "w") as f:
        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerow(["result"] + run_metrics)

    print("\nDone!")
    print("Results folder:", run_dir)
    print("Metrics file:", metric_path)


if __name__ == "__main__":
    main()