import numpy as np
from biological_random_walks.personalization_vector_creation.pv_creation import PersonalizationVectorCreation

class EmbeddingPersonalizationVectorCreation(PersonalizationVectorCreation):
    def __init__(self, universe, path_gene_embedding, seed_gene, k=0):
        self.universe = list(universe)
        self.gene2vec, self.dim = self._load_embedding(path_gene_embedding)
        self.seed_gene = seed_gene
        self.k = k

    def run(self, use_numpy=True):
        if use_numpy:
            return self._set_up_embedding_personalization_vector_numpy()
        else:
            return self._set_up_embedding_personalization_vector_loop()

    # =========================================================
    # LOAD EMBEDDING
    # =========================================================
    def _load_embedding(self, file_path):
        gene2vec = {}

        with open(file_path, 'r') as f:
            header = f.readline().strip().split('\t')
            dim = len(header) - 1

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                gene = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                if len(vec) != dim:
                    raise ValueError(f"Vector size mismatch at gene {gene}")

                gene2vec[gene] = vec

        return gene2vec, dim

    # =========================================================
    # COSINE SIMILARITY (CHUẨN HÓA THỐNG NHẤT)
    # =========================================================
    def _cosine_similarity(self, a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # =========================================================
    # LOOP VERSION (REFERENCE)
    # =========================================================
    def _set_up_embedding_personalization_vector_loop(self):
        S = self.seed_gene
        k = self.k

        assert isinstance(S, (list, set, dict))
        assert self.gene2vec is not None

        personalization_vector = {}

        # ---- chuẩn hóa S ----
        if isinstance(S, dict):
            S_dict = {g: w for g, w in S.items() if g in self.gene2vec}
        else:
            S_dict = {g: 1.0 for g in S if g in self.gene2vec}

        S_genes = list(S_dict.keys())

        for gene in self.universe:

            if gene not in self.gene2vec or len(S_genes) == 0:
                personalization_vector[gene] = 0.0
                continue

            g_vec = self.gene2vec[gene]

            # =========================
            # CASE 1: k = 0
            # =========================
            if k == 0:
                weighted_sum = 0.0
                weight_total = 0.0

                for s in S_genes:
                    if s == gene:
                        continue

                    s_vec = self.gene2vec[s]
                    w = S_dict[s]

                    sim = self._cosine_similarity(g_vec, s_vec)

                    weighted_sum += w * sim
                    weight_total += w

                score = weighted_sum / weight_total if weight_total > 0 else 0.0

            # =========================
            # CASE 2: k > 0
            # =========================
            else:
                sims = []

                for s in S_genes:
                    if s == gene:
                        continue

                    s_vec = self.gene2vec[s]
                    sim = self._cosine_similarity(g_vec, s_vec)

                    sims.append((s, sim))

                sims.sort(key=lambda x: x[1], reverse=True)

                k_eff = min(k, len(sims))
                top_k = sims[:k_eff]

                weighted_sum = 0.0
                weight_total = 0.0

                for s, sim in top_k:
                    w = S_dict[s]
                    weighted_sum += w * sim
                    weight_total += w

                score = weighted_sum / weight_total if weight_total > 0 else 0.0

            # clamp giống numpy
            score = max(score, 0.0)

            personalization_vector[gene] = score

        # ---- L1 normalization trên universe ----
        l1_norm = sum(personalization_vector.values())
        if l1_norm > 0:
            personalization_vector = {
                g: v / l1_norm for g, v in personalization_vector.items()
            }

        return personalization_vector

    # =========================================================
    # NUMPY VERSION (FAST + MATCH LOOP)
    # =========================================================
    def _set_up_embedding_personalization_vector_numpy(self):
        S = self.seed_gene
        k = self.k

        assert isinstance(S, (list, set, dict))

        # ---- chuẩn hóa S ----
        if isinstance(S, dict):
            S_dict = {g: w for g, w in S.items() if g in self.gene2vec}
        else:
            S_dict = {g: 1.0 for g in S if g in self.gene2vec}

        S_genes = list(S_dict.keys())

        genes = self.universe
        scores = {g: 0.0 for g in genes}

        valid_genes = [g for g in genes if g in self.gene2vec]

        if len(valid_genes) == 0 or len(S_genes) == 0:
            return scores

        G = np.stack([self.gene2vec[g] for g in valid_genes])
        S_mat = np.stack([self.gene2vec[g] for g in S_genes])

        # ---- normalize ----
        G_norm = np.linalg.norm(G, axis=1, keepdims=True)
        S_norm = np.linalg.norm(S_mat, axis=1, keepdims=True)

        G_norm[G_norm == 0] = 1
        S_norm[S_norm == 0] = 1

        G = G / G_norm
        S_mat = S_mat / S_norm

        # cosine matrix
        C = G @ S_mat.T

        W = np.array([S_dict[g] for g in S_genes])

        for i, gene in enumerate(valid_genes):
            sims = C[i]

            # mask skip self
            mask = np.array([s != gene for s in S_genes])

            sims = sims[mask]
            W_masked = W[mask]

            if len(sims) == 0:
                score = 0.0

            elif k == 0:
                score = np.sum(sims * W_masked) / np.sum(W_masked)

            else:
                k_eff = min(k, len(sims))
                idx = np.argsort(sims)[-k_eff:]

                score = np.sum(sims[idx] * W_masked[idx]) / np.sum(W_masked[idx])

            score = max(score, 0.0)
            scores[gene] = float(score)

        # ---- L1 normalize ----
        l1 = sum(scores.values())
        if l1 > 0:
            scores = {g: v / l1 for g, v in scores.items()}

        return scores

