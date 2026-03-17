# 🚀 END-TO-END PIPELINE: Build Knowledge Graph từ CSV/TXT → Train R-GCN với Embedding (chỉ train node là GENE)

## 🎯 Mục tiêu
- Xây dựng Knowledge Graph (KG) từ nhiều file `.csv`, `.txt`
- Train mô hình :contentReference[oaicite:0]{index=0}
- Chỉ học embedding cho node **gene**
- Thời gian train ~1–2 giờ (tuỳ data + GPU)

---

# 🧩 PHASE 1: DATA INGESTION (Đọc & chuẩn hoá dữ liệu)

## 1.1 Input dữ liệu
Các file có thể dạng:

### CSV

gene1,interaction,gene2
TP53,activates,BRCA1
BRCA1,binds,RAD51

### TXT


TP53 activates BRCA1
BRCA1 binds RAD51

---

## 1.2 Chuẩn hoá về triple format
Chuẩn chung:

(head, relation, tail)


👉 Output unified:


TP53, activates, BRCA1
BRCA1, binds, RAD51

---

## 1.3 Merge nhiều nguồn
- concat tất cả file → 1 dataframe
- drop duplicate
- normalize text (lowercase nếu cần)

```python
import pandas as pd

df1 = pd.read_csv("file1.csv")
df2 = pd.read_csv("file2.csv")

df = pd.concat([df1, df2])
df.columns = ["head", "relation", "tail"]
df = df.drop_duplicates()
```

---
# 🧠 PHASE 2: FILTER (Chỉ giữ node là gene)

## 2.1 Xác định gene list

Ví dụ:

```
gene_list = {TP53, BRCA1, EGFR, ...}
```

## 2.2 Filter triples

👉 chỉ giữ edge mà:

* head ∈ gene
* tail ∈ gene

```python
df = df[df["head"].isin(gene_list) & df["tail"].isin(gene_list)]
```

---
# 🔢 PHASE 3: ENCODING (ID mapping)

## 3.1 Map entity → ID

```python
entities = list(set(df["head"]) | set(df["tail"]))
entity2id = {e:i for i,e in enumerate(entities)}
```

## 3.2 Map relation → ID

```python
relations = df["relation"].unique()
rel2id = {r:i for i,r in enumerate(relations)}
```

---

## 3.3 Convert sang index

```python
df["h_id"] = df["head"].map(entity2id)
df["t_id"] = df["tail"].map(entity2id)
df["r_id"] = df["relation"].map(rel2id)
```

---

# 🧱 PHASE 4: BUILD GRAPH (PyTorch Geometric)

Dùng PyTorch Geometric

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([
    df["h_id"].values,
    df["t_id"].values
], dtype=torch.long)

edge_type = torch.tensor(df["r_id"].values, dtype=torch.long)
```

---

## 4.1 Node features (embedding init)

```python
import torch.nn as nn

num_nodes = len(entity2id)
embedding_dim = 128

node_emb = nn.Embedding(num_nodes, embedding_dim)
```

---

# 🤖 PHASE 5: MODEL (R-GCN)

```python
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_rels):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, 128)
        self.conv1 = RGCNConv(128, 128, num_rels)
        self.conv2 = RGCNConv(128, 128, num_rels)

    def forward(self, edge_index, edge_type):
        x = self.emb.weight
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x
```

---

# 🎯 PHASE 6: TRAINING TASK (Link Prediction)

## 6.1 Positive samples

```
(h, r, t) từ data
```

## 6.2 Negative sampling

```
(h, r, t') random
```

```python
import random

def negative_sample(h, t, num_nodes):
    t_neg = random.randint(0, num_nodes-1)
    return h, t_neg
```

---

## 6.3 Scoring function (DistMult style)

```python
def score(h, t):
    return (h * t).sum(dim=1)
```

---

## 6.4 Loss

```python
loss = -log(sigmoid(pos_score - neg_score))
```

---
# ⏱️ PHASE 7: TRAIN LOOP (~1–2h)

```python
model = RGCN(num_nodes, len(rel2id))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):

    model.train()
    optimizer.zero_grad()

    x = model(edge_index, edge_type)

    h = x[df["h_id"]]
    t = x[df["t_id"]]

    pos_score = (h * t).sum(dim=1)

    # negative
    t_neg_idx = torch.randint(0, num_nodes, t.shape)
    t_neg = x[t_neg_idx]

    neg_score = (h * t_neg).sum(dim=1)

    loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item())
```

👉 Với:

* ~100k–1M edges
* GPU (hoặc CPU mạnh)

⏱️ ~1–2 giờ

---
# 🧬 PHASE 8: OUTPUT (Gene Embedding)

```python
gene_embeddings = model.emb.weight.data
```

---

## 8.1 Tính similarity giữa 2 gene

```python
from torch.nn.functional import cosine_similarity

sim = cosine_similarity(
    gene_embeddings[id_A].unsqueeze(0),
    gene_embeddings[id_B].unsqueeze(0)
)
```
---
# 📌 GHI NHỚ QUAN TRỌNG

✔ ID chỉ để index → KHÔNG có ý nghĩa
✔ Embedding mới chứa:

* chức năng gene
* vai trò trong network
* mức độ tương đồng

✔ Relational Graph Convolutional Network học:

* structure graph
* loại quan hệ sinh học

---

# 🎯 PIPELINE TỔNG KẾT

```
CSV/TXT
   ↓
Normalize triples
   ↓
Filter gene nodes
   ↓
Encode ID
   ↓
Build graph (edge_index, edge_type)
   ↓
Embedding init
   ↓
R-GCN
   ↓
Link prediction training
   ↓
Gene embeddings
   ↓
Similarity / clustering / downstream tasks
```

---

# 🚀 NÂNG CẤP (OPTIONAL)

* Pretrain bằng TransE
* Dùng Graph Attention Network
* Thêm gene expression features