# DINOv3–SAM–FViT: Dual-Encoder Transformer Framework for Cross-Domain Medical Image Classification

Official PyTorch implementation of the **DINOv3–SAM–FViT** framework for robust **multi-dataset and cross-domain medical image classification**.

---

## 🔥 Overview

This repository presents a novel **dual-encoder transformer-based framework** that integrates complementary representations from:

- **DINOv3 (Vision Transformer)** for global semantic understanding  
- **Segment Anything Model (SAM) image encoder** for anatomy-aware spatial features  

The two feature streams are spatially aligned and fused using a **CBAM attention module**, enabling robust classification across heterogeneous datasets and imaging modalities.

---

## 📄 Citation

If this repository contributes to your research, please consider citing:

```bibtex
@article{AlGharib2026,
  title={DINOv3--SAM: A Dual-Encoder Transformer Framework for Cross-Domain and Multi-Dataset Medical Image Classification},
  author={Al Gharib, Saeed and Dornaika, Fadi and Charafeddine, Jinan and Haddad, Samir},
  journal={Under Review},
  year={2026}
}
```

---

## 🧠 Key Contributions

- ✔️ Dual-encoder architecture combining **semantic (DINOv3)** and **anatomical (SAM)** representations  
- ✔️ Attention-based fusion using **CBAM**  
- ✔️ Multi-head classification with consistency regularization  
- ✔️ Robust performance across **multiple datasets and domains**  
- ✔️ Unified evaluation across **binary and multi-class tasks**

---

## 🏗️ Architecture

The framework consists of:

1. **DINOv3 Encoder (ViT-Small)**  
   - Global semantic feature extraction  
   - Last transformer blocks are fine-tuned  

2. **SAM Image Encoder (ViT-Base)**  
   - Frozen encoder capturing anatomical structures  

3. **Feature Fusion**  
   - Spatial alignment  
   - Channel concatenation  
   - CBAM attention refinement  

4. **Multi-Head Prediction**  
   - DINO branch  
   - SAM branch  
   - Fusion branch (final output)

5. **Training Objective**  
   - Cross-Entropy Loss  
   - Consistency Loss (MSE)

---

## 📈 Results (Summary)

The proposed framework is evaluated across five publicly available datasets:

| Dataset                          | Task         | Accuracy |
|----------------------------------|-------------|---------|
| Chest X-ray Pneumonia (Mooney)  | Binary      | **97.6%** |
| Cardiomegaly Dataset            | Binary      | **79.9%** |
| RSNA Pneumonia                  | Binary      | **84.6%** |
| COVID-19 Radiography            | Multi-class | **99.41%** |
| Dermatology MNIST               | Multi-class | **78.75%** |

The model demonstrates strong generalization across heterogeneous datasets and maintains robust performance under cross-domain conditions.

---

## 📂 Dataset

This framework is evaluated on five publicly available datasets:

- Chest X-ray Pneumonia (Mooney)  
- Cardiomegaly Disease Prediction Dataset  
- RSNA Pneumonia Detection Dataset  
- COVID-19 Radiography Dataset  
- Dermatology MNIST Dataset  

⚠️ **Datasets are NOT included** due to licensing restrictions.

Users must download them from official sources (e.g., Kaggle, MedMNIST).

Example dataset structure:

```
dataset/
├── train/
├── val/
├── test/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/DINOv3-SAM-FViT.git
cd DINOv3-SAM-FViT

pip install -r requirements.txt
```

---

## 🚀 Training

Run the training script:

```bash
python dinosam_rsna_full.py
```

The script handles:

- Data loading (user-provided datasets)  
- Preprocessing and augmentation  
- Model training  
- Evaluation  

---

## 🔬 Reproducibility

- Fixed random seeds  
- Standardized 80/10/10 splits (binary datasets)  
- Predefined split for COVID-19 dataset  
- Unified evaluation protocol  

---

## 📌 Code Availability

The full implementation is available at:

https://github.com/yourusername/DINOv3-SAM-FViT

---

## ⚠️ Limitations

- Performance depends on dataset quality and labeling  
- SAM encoder increases computational cost  
- Multi-label classification is not explored  

---

## 🔮 Future Work

- Multi-label classification  
- Uncertainty estimation  
- Cross-modal learning  
- Model compression for deployment  

---

## 👨‍💻 Authors

- Saeed Al Gharib  
- Fadi Dornaika  
- Jinan Charafeddine  
- Samir Haddad  

---

## 📬 Contact

salgharib001@ikasle.ehu.eus
