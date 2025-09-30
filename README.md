# Rehabilitation Exercise Quality Assessment via Supervised Contrastive Learning

This repository contains the experiments and models from our paper published in *Medical & Biological Engineering & Computing* (2024):  
"Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives"
https://link.springer.com/article/10.1007/s11517-024-03177-x

---

## Preface

In retrospect, a stronger pipeline would first leverage a **pretrained action recognition model** and then **fine-tune** it using our supervised contrastive learning approach. This would likely improve generalization and training efficiency for quality assessment tasks.

---

## Datasets

Our experiments were conducted on two commonly used rehabilitation datasets:

- **KIMORE** – Kinect-based motion recordings of rehabilitation exercises.  
- **UI-PRMD** – Human pose and rehabilitation movement dataset for posture analysis.

---

## Models & Methods

We explored a range of spatial-temporal deep neural networks and graph-based models, including:

- **ST-GCN** (Spatio-Temporal Graph Convolutional Networks)  
- **GAT** (Graph Attention Network)  
- **GATv2** (an improved version with dynamic attention mechanisms)

Our method uses **supervised contrastive learning** with both **hard negatives** (challenging, incorrect examples) and **soft negatives** (similar but distinct examples) to improve embedding space structure and representation quality.

---
