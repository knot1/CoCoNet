## ⚙️ **Installation & Dependencies**

Before running the code, make sure you have the following dependencies installed:

```python
conda env create -f requirements.yml
```

## 🛰️ **Datasets**

Extensive experiments were conducted on four public datasets:

- ISPRS Vaihingen  &nbsp; &nbsp;          [Download Dataset](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
- ISPRS Potsdam  &nbsp; &nbsp;            [Download Dataset](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)

## 🚀 **Usage: Training CAF-Net**


To train **CAF-Net** on the ISPRS Vaihingen dataset, use the following command:

```python
bash train_Vaihingen.sh 0 
```

## 🔥 **Contribution: Multi-Level Conflict Modeling**

We explicitly model RGB–DSM conflicts as a **first-class supervisory signal** instead of treating them as an implicit byproduct.

**Conflict definition (cosine similarity on L2-normalized features)**

$$
C(h,w) = 1 - \text{cosine}(F_{RGB}, F_{DSM})
$$

Here, $C(h,w)$ denotes a **per-pixel scalar conflict map** obtained by channel-wise cosine similarity aggregation.

**Multi-level conflict maps**

- Stage 1–4 each produces a conflict map.
- We distinguish:
  - **Global conflict**: stage-wise global averages to capture cross-modality inconsistency.
  - **Boundary conflict**: spatial conflicts aligned with segmentation boundaries.

**Why it matters**

> Existing methods ignore conflicts or only handle them implicitly,  
> while we explicitly model conflict as a supervisory signal.

## 🧪 **Minimal Ablation Setup**

- Baseline (no conflict supervision)
- + Boundary conflict only
- + Global conflict only
- + Multi-level (Stage 1–4) conflict supervision (full model)
