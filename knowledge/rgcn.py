import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

def load_kg(path):
    df = pd.read_csv(path, sep="\t")
    return df

def encode_kg(df):
    nodes = set(df["head"]).union(set(df["tail"]))
    node2id = {n: i for i, n in enumerate(nodes)}
    id2node = {i: n for n, i in node2id.items()}

    rels = df["relation"].unique()
    rel2id = {r: i for i, r in enumerate(rels)}

    return node2id, id2node, rel2id

def build_pyg_data(df, node2id, rel2id):
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

    # ❌ OLD: one-hot → dễ collapse + tốn RAM
    # x = torch.eye(num_nodes)

    # ✅ NEW: dùng embedding learnable
    x = torch.arange(num_nodes)  # chỉ là index

    data = Data(x=x, edge_index=edge_index)
    data.edge_type = edge_type

    return data

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim=64):
        super().__init__()

        # ✅ NEW: embedding layer thay cho one-hot
        self.embedding = torch.nn.Embedding(num_nodes, hidden_dim)

        # ❌ OLD: input dim = num_nodes
        # self.conv1 = RGCNConv(num_nodes, hidden_dim, num_relations)

        # ✅ NEW: input dim = hidden_dim
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

    def forward(self, x, edge_index, edge_type):
        # ✅ NEW: lấy embedding từ index
        x = self.embedding(x)

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

def train(model, data, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    u, v = data.edge_index  # dùng cho link prediction

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_type)

        # ❌ OLD LOSS (gây collapse)
        # loss = (out ** 2).mean()

        # ✅ NEW LOSS: link prediction
        pos_score = (out[u] * out[v]).sum(dim=1)

        # negative sampling
        neg_v = torch.randint(0, out.size(0), (v.size(0),))
        neg_score = (out[u] * out[neg_v]).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() \
               -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

    return model

def get_gene_embeddings(model, data, id2node, df):
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

def build_similarity(gene_emb):
    norm_emb = F.normalize(gene_emb)
    sim = norm_emb @ norm_emb.T
    return sim

def build_weighted_edges(sim, gene_ids, k=10):
    edges = []

    for i in range(sim.shape[0]):
        vals, idx = torch.topk(sim[i], k=k+1)

        for j, v in zip(idx[1:], vals[1:]):
            edges.append((gene_ids[i], gene_ids[j.item()], v.item()))

    return edges

def main():
    kg_path = "KGs/kg.csv"

    print("Loading KG...")
    df = load_kg(kg_path)

    print("Encoding...")
    node2id, id2node, rel2id = encode_kg(df)

    print("Building graph...")
    data = build_pyg_data(df, node2id, rel2id)

    print("Training R-GCN...")
    model = RGCN(
        num_nodes=data.num_nodes,
        num_relations=len(rel2id)
    )

    # check BEFORE training
    model.eval()
    with torch.no_grad():
        emb_before = model(data.x, data.edge_index, data.edge_type)

    print("Before training:")
    print(emb_before[:5])
    print("Before mean/std:", emb_before.mean().item(), emb_before.std().item())

    model = train(model, data, epochs=10)

    print("Extract gene embeddings...")
    gene_emb, gene_ids = get_gene_embeddings(model, data, id2node, df)

    print("After training:")
    print(gene_emb[:5])
    print("After mean/std:", gene_emb.mean().item(), gene_emb.std().item())

    print("Compute similarity...")
    sim = build_similarity(gene_emb)

    print("Build weighted network...")
    edges = build_weighted_edges(sim, gene_ids, k=10)

    print("Saving...")
    out_df = pd.DataFrame(edges, columns=["u", "v", "weight"])
    out_df.to_csv("gene_similarity_network.csv", sep="\t", index=False)

    print("DONE 🚀")

if __name__ == "__main__":
    main()