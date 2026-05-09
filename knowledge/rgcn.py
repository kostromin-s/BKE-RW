import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=64, dropout=0.3):
        super().__init__()

        self.embedding = torch.nn.Embedding(num_nodes, hidden_dim)

        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        x = self.embedding(x)

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)
        return x

class Preprocessor:
    def __init__(self, df, hidden_dim = 64, dropout=0.3, nameDataset = "", k=10, epochs=20, lr=0.01, num_neg=5):
        self.df = df
        self.hidden_dim = hidden_dim
        self.nameDataset = nameDataset
        self.dropout = dropout
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.num_neg = num_neg

    def encode_kg(self, df = None):
        if df is None:
            df = self.df
        nodes = set(df["head"]).union(set(df["tail"]))
        node2id = {n: i for i, n in enumerate(nodes)}
        id2node = {i: n for n, i in node2id.items()}

        rels = df["relation"].unique()
        rel2id = {r: i for i, r in enumerate(rels)}

        return node2id, id2node, rel2id

    def build_pyg_data(self, df, node2id, rel2id):
        edge_index = []
        edge_type = []

        for _, row in df.iterrows():
            u = node2id[row["head"]]
            v = node2id[row["tail"]]
            r = rel2id[row["relation"]]

            edge_index.append([u, v])
            edge_type.append(r)

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_type = torch.tensor(edge_type)

        num_nodes = len(node2id)

        x = torch.arange(num_nodes)

        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type

        return data
    
    def train(self, model, data, epochs=None, lr=None, num_neg=None):
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr
        if num_neg is None:
            num_neg = self.num_neg

        optimizer = torch.optim.adam.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5  # ✅ L2 regularization
        )

        u, v = data.edge_index

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.edge_type)

            # ===== POSITIVE =====
            pos_score = (out[u] * out[v]).sum(dim=1)

            # ===== MULTI-NEGATIVE =====
            neg_v = torch.randint(0, out.size(0), (v.size(0), num_neg))
            u_expand = u.unsqueeze(1).expand(-1, num_neg)

            neg_score = (out[u_expand] * out[neg_v]).sum(dim=2)

            # ===== LOSS =====
            loss_pos = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            loss_neg = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

            loss = loss_pos + loss_neg

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | Loss {loss.item():.4f}")

        return model

    def get_gene_embeddings(self, model, data, id2node, df):
        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index, data.edge_type)

        gene_nodes = set(df[df["head_type"] == "Gene"]["head"]) | \
                    set(df[df["tail_type"] == "Gene"]["tail"])

        gene_indices = []
        for i, node in id2node.items():
            if node in gene_nodes:
                gene_indices.append(i)

        gene_emb = emb[gene_indices]
        gene_ids = [id2node[i] for i in gene_indices]

        return gene_emb, gene_ids
        

    def save_gene_embeddings(self, gene_emb, gene_ids, nameDataset=""):
        if nameDataset == "":
            out_path = "gene_embeddings/default_gene_embeddings.tsv"
        else:
            out_path = f"gene_embeddings/{nameDataset}_gene_embeddings.tsv"
        emb_np = gene_emb.cpu().numpy()

        df = pd.DataFrame(emb_np)
        df.insert(0, "gene", gene_ids)

        df.to_csv(out_path, sep="\t", index=False)
    
    def build_similarity(self, gene_emb):
        norm_emb = F.normalize(gene_emb)
        sim = norm_emb @ norm_emb.T
        return sim
        
    def build_weighted_edges(self, sim, gene_ids, k=10):
        edges = []

        for i in range(sim.shape[0]):
            vals, idx = torch.topk(sim[i], k=k+1)

            for j, v in zip(idx[1:], vals[1:]):
                edges.append((gene_ids[i], gene_ids[j.item()], v.item()))

        return edges
    
    def run(self):
        
        print("================Dataset:", self.nameDataset if self.nameDataset else "Default" ,"================")
        node2id, id2node, rel2id = self.encode_kg()
        data = self.build_pyg_data(self.df, node2id, rel2id)

        model = RGCN(
            num_nodes=data.num_nodes,
            num_relations=len(rel2id),
            hidden_dim=64,
            dropout=0.3   # ✅ dễ tune
        )

        model.eval()
        with torch.no_grad():
            emb_before = model(data.x, data.edge_index, data.edge_type)
        
        print("Before training mean/std:", emb_before.mean().item(), emb_before.std().item())

        model = self.train(model, data)

        print("Extract gene embeddings...")
        gene_emb, gene_ids = self.get_gene_embeddings(model, data, id2node, self.df)

        print("Saving gene embeddings...")
        self.save_gene_embeddings(gene_emb, gene_ids, self.nameDataset)

        print("After training mean/std:",
            gene_emb.mean().item(), gene_emb.std().item())

        print("Compute similarity...")
        sim = self.build_similarity(gene_emb)

        print("Build weighted network...")
        edges = self.build_weighted_edges(sim, gene_ids, k=10)

        print("Saving...")
        out_df = pd.DataFrame(edges, columns=["u", "v", "weight"])
        out_df.to_csv(f"matrix_gene_sim/{self.nameDataset}_gene_similarity.csv", sep="\t", index=False)

        print("DONE 🚀")
