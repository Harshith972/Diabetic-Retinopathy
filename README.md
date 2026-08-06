

# Early Diabetic Retinopathy Detection Using Deep Learning & Medical Preprocessing

An automated screening system designed to detect and classify the severity of Diabetic Retinopathy (DR) from retinal fundus images using an EfficientNetB3 Convolutional Neural Network, custom Green Channel CLAHE preprocessing, and categorical focal crossentropy loss.

---

## Interactive Links

* Live Web Application: [Streamlit Screening App](https://diabetic-retinopathy-sfsdapp56zgeupklkmskxrb.streamlit.app/)
* Google Colab Development Notebook: [diabetic_rethiopathy.ipynb](https://colab.research.google.com/drive/1Q21H3GiLxDMyR_yrjksKrzcUP_BFDTdx)
* Interactive Portfolio: [Harshith Reddy Janga's Portfolio](https://jangaharshithportfolio.lovable.app/)
* github : [Github](https://github.com/Harshith972/Diabetic-Retinopathy)
---

## Executive Summary

Diabetic Retinopathy is a microvascular complication of diabetes and a leading cause of preventable blindness worldwide. Early identification is crucial, yet traditional manual diagnosis by ophthalmologists is resource-intensive and hard to scale in high-volume or low-resource clinical environments.

This capstone project implements an end-to-end multi-class AI screening pipeline that classifies fundus images into four clinical severity grades:

* Grade 0: Normal / No DR
* Grade 1: Mild DR (Microaneurysms)
* Grade 2: Moderate DR (Hemorrhages / Hard Exudates)
* Grade 3: Severe DR (Neovascularization / Extensive Hemorrhages)

By combining multi-source dataset fusion (Messidor-2 + E-Ophtha), Green Channel CLAHE feature enhancement, and Categorical Focal Crossentropy loss, the model overcomes severe clinical class imbalance and achieves robust diagnostic accuracy on early-stage and advanced pathology.

---

## Key Technical Features

* Green Channel CLAHE Enhancement: Isolates the green spectral channel—which offers maximum optical contrast for retinal vasculature—and applies Contrast Limited Adaptive Histogram Equalization to highlight subtle microaneurysms and exudates.
* Multi-Source Data Fusion: Fuses Messidor-2 (1,200 images) and E-Ophtha (658 images) datasets to enrich minority early-stage lesion representations.
* Deep Learning Architecture: Fine-tuned EfficientNetB3 backboned with transfer learning from ImageNet weights, Global Average Pooling, Batch Normalization, and high dropout rates (0.4–0.5) to combat overfitting.
* Class Imbalance Handling: Employs Categorical Focal Crossentropy (Gamma = 2.0) to dynamically penalize hard-to-classify minority samples during training.
* Heuristic Input Safeguard: Automatically computes the red-to-blue channel intensity ratio of uploaded images to verify biological tissue presence before triggering model inference.
* Deployment via Streamlit & GitHub LFS: Tracks large model binary weights (`final_clahe_model_V4.h5`) using Git Large File Storage for real-time web inference.

---

## Multi-Source Dataset Distribution

The V3/V4 pipeline fuses data from Messidor-2 and E-Ophtha to build a comprehensive dataset totaling 1,663 evaluated images:

* Grade 0 (Normal): 814 images
* Grade 1 (Mild DR): 301 images
* Grade 2 (Moderate DR): 294 images
* Grade 3 (Severe DR): 254 images

---

## System Architecture & Workflow

```text
+------------------------+      +-------------------------------+
|  Raw Fundus Input      | ---> |  Red/Blue Heuristic Check     |
|  (JPG / PNG / TIF)     |      |  (Validates Retinal Tissue)   |
+------------------------+      +-------------------------------+
                                                |
                                                v
+------------------------+      +-------------------------------+
|  Model Inference       | <--- |  Green Channel CLAHE          |
|  (EfficientNetB3)      |      |  (300x300 RGB Resized Input)  |
+------------------------+      +-------------------------------+
            |
            v
+---------------------------------------------------------------+
|  Diagnostic Output & Report                                   |
|  (Severity Grade, Class Probabilities, Clinical Guidance)     |
+---------------------------------------------------------------+

```

---

## Performance & Diagnostic Results

Model V4 (Final CLAHE Pipeline) Evaluation Metrics:

* Overall System Accuracy: 70.84%
* Weighted F1-Score: 0.69
* Grade 0 (Normal) Precision: 0.73 | Recall: 0.90 | F1-Score: 0.80
* Grade 1 (Mild) Precision: 0.58 | Recall: 0.31 | F1-Score: 0.41
* Grade 2 (Moderate) Precision: 0.57 | Recall: 0.50 | F1-Score: 0.53
* Grade 3 (Severe) Precision: 0.85 | Recall: 0.82 | F1-Score: 0.83

Model Progression across Iterations:

* V2 Baseline Model: 55.00% Accuracy
* V3 Multi-Source Fusion: 66.45% Accuracy
* V4 CLAHE Preprocessing (Final): 70.84% Accuracy

---

## Repository Structure

```text
Diabetic-Retinopathy/
│
├── .gitattributes             # Git Large File Storage (LFS) configuration
├── app.py                      # Streamlit web application & interface logic
├── final_clahe_model_V4.h5     # Trained EfficientNetB3 model weights (via Git LFS)
├── requirements.txt            # Python runtime dependencies
└── README.md                   # Complete repository documentation

```

---

## Installation & Local Setup

1. Clone the repository:

```bash
git clone https://github.com/Harshith972/Diabetic-Retinopathy.git
cd Diabetic-Retinopathy

```

2. Fetch trained model weights via Git LFS:

```bash
git lfs install
git lfs pull

```

3. Install required Python packages:

```bash
pip install -r requirements.txt

```

4. Launch the local Streamlit web dashboard:

```bash
streamlit run app.py

```

---

## Tech Stack

* Programming Language: Python 3
* Deep Learning Frameworks: TensorFlow, Keras
* Computer Vision: OpenCV, PIL (Pillow)
* Data Science & Analysis: Pandas, NumPy, Scikit-learn
* Model Deployment & MLOps: Streamlit, GitHub LFS

---

## Authors & Academic Citation

### Project Team
- Janga Harshith Reddy (Reg No: 22BCE3057)
- Pamisetty Venkata Krishna (Reg No: 22BCE3545)
- Pydikondala Devisri Satya Avinash (Reg No: 22BCT0321)

### Academic Supervision
- Project Supervisor: Prof. Jafar Ali Ibrahim S (Assistant Professor Sr. Grade 2)
- Institution: School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT Vellore)
- Degree Program: Bachelor of Technology in Computer Science and Engineering
- Academic Term: Winter Semester 2025–2026 (April 2026)

---

