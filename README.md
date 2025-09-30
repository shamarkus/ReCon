# Rehabilitation Exercise Quality Assessment via Supervised Contrastive Learning

This repository contains the experiments and models from our paper published in *Medical & Biological Engineering & Computing* (2024):  
➡️ “Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives”  
🔗 https://link.springer.com/article/10.1007/s11517-024-03177-x

---

## Preface

In retrospect, a stronger pipeline would first leverage a **pretrained action recognition model** and then **fine-tune** it using our supervised contrastive learning approach. This would likely improve generalization and training efficiency for quality assessment tasks.

---

## Datasets

Our experiments were conducted on two commonly used rehabilitation datasets:

- **KIMORE** – Kinect-based motion recordings of rehabilitation exercises.  
- **UI-PRMD** – Human pose and rehabilitation movement dataset for posture analysis.

These datasets enabled evaluation of model performance in assessing the quality of exercise execution from real-world movement data.

---

## Models & Methods

We explored a range of spatial-temporal deep neural networks and graph-based models, including:

- **ST-GCN** (Spatio-Temporal Graph Convolutional Networks)  
- **GAT** (Graph Attention Network)  
- **GATv2** (an improved version with dynamic attention mechanisms)

Our method uses **supervised contrastive learning** with both **hard negatives** (challenging, incorrect examples) and **soft negatives** (similar but distinct examples) to improve embedding space structure and representation quality.

---

## Results & Observations

- The contrastive learning framework enhanced the separation between high- and low-quality exercise embeddings.  
- Incorporating hard and soft negatives provided complementary benefits for model training.  
- GAT-based models often outperformed standard ST-GCN when paired with contrastive objectives.  
- Pretraining on large-scale action recognition datasets remains a promising direction for future improvements.

---

## Citation

If you use this work, please cite:

> Your Name, et al. *Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives*. Medical & Biological Engineering & Computing (2024). DOI: [10.1007/s11517-024-03177-x](https://link.springer.com/article/10.1007/s11517-024-03177-x)
