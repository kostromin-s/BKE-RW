from hydra.core.hydra_config import HydraConfig
import os
import hydra
from omegaconf import DictConfig

from biological_random_walks.BiologicalRandomWalks import BiologicalRandomWalks
from biological_random_walks.BiologicalRandomWalksWithGeneSim import BiologicalRandomWalksWithGeneSim

import random
import math
import csv


def load_seed_file(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def split_seed(seed_set, train_ratio=0.7):
    seed_list = list(seed_set)
    random.shuffle(seed_list)
    split_idx = int(len(seed_list) * train_ratio)
    return set(seed_list[:split_idx]), set(seed_list[split_idx:])


def recall_at_k(ranked_list, test_seed, k):
    top_k = [g for g, _ in ranked_list[:k]]
    hit = len(set(top_k) & test_seed)
    return hit / len(test_seed) if len(test_seed) > 0 else 0


def dcg_at_k(ranked_list, test_seed, k):
    dcg = 0.0
    for i, (g, _) in enumerate(ranked_list[:k]):
        if g in test_seed:
            dcg += 1 / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_list, test_seed, k):
    dcg = dcg_at_k(ranked_list, test_seed, k)
    ideal_hits = min(len(test_seed), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0


# 🔹 NEW: compute k list based on evaluation mode
def compute_k_list(cfg, test_size):
    mode = cfg.evaluation.mode

    if mode == "fixed":
        return list(cfg.evaluation.fixed.k)

    elif mode == "dynamic":
        k_list = []
        for c in cfg.evaluation.dynamic.c_values:
            k = int(c * test_size)

            if "k_max" in cfg.evaluation.dynamic:
                k = min(k, cfg.evaluation.dynamic.k_max)

            k_list.append(k)

        return k_list

    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    run_dir = HydraConfig.get().runtime.output_dir

    full_seed = load_seed_file(cfg.experiment.seed)

    all_metrics = []
    header_written = False
    header = []

    for i in range(10):
        print(f"\n===== RUN {i} =====")

        random.seed(42 + i)

        train_seed, test_seed = split_seed(full_seed, 0.7)

        output_path = os.path.join(run_dir, f"result{i}.txt")

        if cfg.method == "gene_sim":

            brw = BiologicalRandomWalksWithGeneSim(

                gene_similarity_file_path=cfg.paths.gene_similarity,

                seed_file_path=cfg.experiment.seed,
                seed_set_override=train_seed,

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
                seed_set_override=train_seed,

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

        # (giữ nguyên logic gốc: KHÔNG loại train_seed)
        # ranked_list = [(g, s) for g, s in brw.ranked_list if g not in train_seed]

        k_list = compute_k_list(cfg, len(test_seed))

        print(f"Test size: {len(test_seed)} | k_list: {k_list}")

        run_metrics = []

        for k in k_list:
            r = recall_at_k(ranked_list, test_seed, k)
            ndcg = ndcg_at_k(ranked_list, test_seed, k)

            print(f"Recall@{k}: {r:.4f} | nDCG@{k}: {ndcg:.4f}")

            run_metrics.extend([r, ndcg])

        # tạo header 1 lần (dựa trên run đầu)
        if not header_written:
            header = ["Run"]
            for k in k_list:
                header += [f"Recall@{k}", f"nDCG@{k}"]
            header_written = True

        all_metrics.append(run_metrics)

    metric_path = os.path.join(run_dir, "metrics.csv")

    avg = [sum(x) / len(x) for x in zip(*all_metrics)]

    with open(metric_path, "w") as f:
        writer = csv.writer(f)

        writer.writerow(header)

        for i, row in enumerate(all_metrics):
            writer.writerow([f"result{i}"] + row)

        writer.writerow(["AVG"] + avg)

    print("\nDone!")
    print("Results folder:", run_dir)
    print("Metrics file:", metric_path)


if __name__ == "__main__":
    main()