# DINOv3-SAM-FViT
Official PyTorch implementation of the DINOv3–SAM F-ViT dual-encoder framework for binary chest X-ray classification.

---

## 🔥 Overview

This repository contains the full implementation of a **DINOv3 + SAM fusion model** for binary medical image classification.  
The method combines:

- **DINOv3 (Vision Transformer)** for global semantic representation  
- **SAM image encoder** for anatomical and structural features  
- **CBAM attention** for adaptive feature fusion  

The framework is applied to chest X-ray datasets such as **RSNA Pneumonia**.

---

## ✨ Features

- Single-file end-to-end implementation  
- DINOv3 + SAM dual encoders  
- CBAM fusion module  
- Multi-head classification (DINO, SAM, Fused)  
- CrossEntropy + MSE consistency loss  
- Automatic train/val/test split  
- Threshold search for best F1-score  
- Confusion matrix + ROC + PR curves  
- Model checkpoint saving  

---

