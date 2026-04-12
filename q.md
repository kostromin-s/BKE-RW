
# 🧬 **Gene Prioritization bằng Embedding**

## (Approach 1 & Approach 3)

# I. Bài toán

**Input:**

*  $X \in \mathbb{R}^{N \times d}$: embedding của toàn bộ gene $N ≈ 12k$
*  $S \subset X$: tập seed genes (30–300 gene đã biết liên quan bệnh)

**Output:**

* Vector $q \in \mathbb{R}^N$
*  $q_i$: mức độ liên quan của gene $i$ tới bệnh

---

# II. Điều kiện tiên quyết

* Tất cả embedding:

  * cùng dimension
  * cùng model
  * đã normalize (L2 norm = 1)

---

# III. Approach 1 — Mean Similarity (Baseline mạnh)

## 💡 Ý tưởng

Gene liên quan bệnh sẽ **trung bình giống seed genes**.

---

## 📐 Công thức

$$
q_i = \frac{1}{|S|} \sum_{s \in S} \cos(g_i, s)
$$

---

## 🔧 Implementation

```python id="a1_impl"
import numpy as np

# X: (N, d), S_idx: index seed genes

# normalize
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# lấy seed vectors
S = X[S_idx]

# similarity matrix
sim_matrix = X @ S.T   # (N, |S|)

# score
q = sim_matrix.mean(axis=1)
```

---

## ✅ Ưu điểm

* Đơn giản, dễ triển khai
* Ổn định
* Không cần tuning nhiều

---

## ❌ Nhược điểm

* Nhạy với noise trong seed genes
* Không bắt được cấu trúc “local” (pathway riêng)

---

## 📌 Khi nên dùng

* Seed genes ít (<30)
* Muốn baseline nhanh
* Embedding chưa chắc chắn chất lượng

---

# IV. Approach 3 — k-Nearest Seed Similarity (Khuyến nghị)

## 💡 Ý tưởng

Gene liên quan bệnh thường chỉ gần **một nhóm nhỏ seed genes**, không phải tất cả.

---

## 📐 Công thức

$$
q_i = \frac{1}{k} \sum_{s \in kNN(g_i)} \cos(g_i, s)
$$

---

## 🔧 Implementation

```python id="a3_impl"
import numpy as np

# normalize
X = X / np.linalg.norm(X, axis=1, keepdims=True)
S = X[S_idx]

# similarity
sim_matrix = X @ S.T   # (N, |S|)

# top-k
k = 10
topk_sim = np.partition(sim_matrix, -k, axis=1)[:, -k:]

# score
q = topk_sim.mean(axis=1)
```

---

## 🚀 Variants (khuyến nghị)

### 1. Weighted kNN

```python id="a3_weighted"
weights = topk_sim / topk_sim.sum(axis=1, keepdims=True)
q = (weights * topk_sim).sum(axis=1)
```

---

### 2. Softmax weighting

```python id="a3_softmax"
temp = 0.1
exp_sim = np.exp(topk_sim / temp)
weights = exp_sim / exp_sim.sum(axis=1, keepdims=True)
q = (weights * topk_sim).sum(axis=1)
```

---

## ✅ Ưu điểm

* Robust với noise
* Bắt được structure theo pathway
* Phù hợp biological reality

---

## ❌ Nhược điểm

* Cần chọn k
* Seed quá ít → không hiệu quả

---

## 📌 Khi nên dùng

* Seed genes ≥ 30
* Dữ liệu có nhiều pathway
* Muốn ranking chính xác hơn

---

# V. So sánh tổng quan

| Tiêu chí            | Approach 1 | Approach 3 |
| ------------------- | ---------- | ---------- |
| Độ đơn giản         | ⭐⭐⭐⭐       | ⭐⭐⭐        |
| Độ chính xác        | ⭐⭐⭐        | ⭐⭐⭐⭐       |
| Robust với noise    | ❌          | ✔          |
| Bắt local structure | ❌          | ✔          |
| Cần tuning          | ❌          | ✔ (k)      |

---

# VI. Gợi ý thực chiến

## 🔥 Pipeline khuyến nghị

1. Normalize embedding
2. Tính cosine similarity (matrix multiplication)
3. Chạy:

   * Approach 1 (baseline)
   * Approach 3 (k = 5, 10, 20)
4. Normalize score về [0, 1]

---

## 🎯 Chọn k

* |S| = 30–100 → k = 5–10
* |S| = 100–300 → k = 10–30

---

## 📊 Chuẩn hóa output

```python id="normalize_q"
q = (q - q.min()) / (q.max() - q.min())
```

---

# VII. Kết luận

* **Approach 1**: baseline mạnh, đơn giản
* **Approach 3**: phù hợp hơn cho bài toán sinh học thực tế

👉 Khuyến nghị:

> Dùng Approach 1 làm baseline, Approach 3 làm model chính

---