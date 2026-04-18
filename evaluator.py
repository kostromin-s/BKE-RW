import math
from collections import defaultdict


# ===== METRIC (GIỮ NGUYÊN LOGIC) =====

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


# ===== COLLECT RESULTS FOR HYBRID =====

class HybridEvaluator:

    def __init__(self, K=200):
        self.K = K
        self.freq = defaultdict(int)
        self.rank_sum = defaultdict(float)
        self.count = defaultdict(int)
        self.num_runs = 0

    def add_run(self, ranked_list):
        self.num_runs += 1

        for rank, (g, _) in enumerate(ranked_list[:self.K], start=1):
            self.freq[g] += 1
            self.rank_sum[g] += rank
            self.count[g] += 1

    def compute_scores(self, theta=0.5):
        scores = {}

        for g in self.freq.keys():
            f = self.freq[g] / self.num_runs

            avg_rank = self.rank_sum[g] / self.count[g]

            # normalized rank score
            s_r = 1 - (avg_rank - 1) / (self.K - 1)

            score = theta * f + (1 - theta) * s_r
            scores[g] = score

        # sort descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)