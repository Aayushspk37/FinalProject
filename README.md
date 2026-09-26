# A Hybrid ViT with Confidence-Aware Clinical Interpretation for Medical Imaging Diagnosis

An explainable medical imaging diagnosis system that combines **Convolutional Neural Networks (CNNs)** for local feature extraction with a **Vision Transformer (ViT)** for global contextual learning. The system is enhanced with a **Confidence-Aware Clinical Interpretation (CACI)** layer to provide structured and clinically meaningful interpretation of model predictions.

<p align="center">
  <img width="1436" height="730" alt="Medical Imaging Diagnosis System" src="https://github.com/user-attachments/assets/929c5831-a45e-4e24-bf75-1db6fbee5550" />
</p>

---

## Objectives

- Develop a **Hybrid Vision Transformer (HViT)** for automated medical image classification.
- Combine **CNN-based local feature extraction** with **Transformer-based global context modeling**.
- Support multiple medical imaging modalities including **MRI, X-ray, CT, and microscopic/dermoscopic images**.
- Develop a **Confidence-Aware Clinical Interpretation (CACI)** layer for structured interpretation of predictions.
- Provide confidence, differential diagnosis, decision margin, severity information, and an LLM-generated clinical narrative.
- Evaluate the system across **seven publicly available medical imaging datasets** using standard and class-imbalance-aware metrics.
- Provide a dedicated **Explainability Hub** for presenting prediction and interpretation results.

The system processes medical images through preprocessing, CNN feature extraction, patch tokenization, Transformer encoding, classification, and CACI-based interpretation.

---

## System Architecture

The system follows an end-to-end medical image diagnosis pipeline:

**Medical Image → Preprocessing → CNN Feature Extraction → Patch Tokenization → Transformer Encoder → Classification → CACI → Explainability Hub**

### Architecture Components

- **Frontend:** HTML, CSS, JavaScript and Django templates.
- **Backend:** Django-based web application for image upload, processing, prediction and result management.
- **Deep Learning:** PyTorch-based Hybrid Vision Transformer.
- **CNN Module:** Extracts local anatomical structures, textures and fine-grained features.
- **Transformer Module:** Captures long-range relationships and global contextual information.
- **CACI Layer:** Converts raw model predictions into structured clinical interpretation.
- **Explainability Hub:** Presents the prediction and CACI outputs through a dedicated interface.
- **Database:** Stores users, medical images, preprocessing records, predictions, CACI outputs and reports.

The report describes the methodology as a pipeline consisting of data collection, preprocessing, model training, inference, CACI signal generation and Explainability Hub rendering.

---

## CACI — Confidence-Aware Clinical Interpretation

The major interpretability component of the system is the **CACI layer**.

For every prediction, CACI generates five structured trust signals:

1. **Prediction Confidence**
   - Represents the confidence level associated with the predicted class.

2. **Ranked Differential Diagnosis**
   - Provides alternative predicted classes ranked according to their model probabilities.

3. **Decision Margin**
   - Indicates how separated the selected prediction is from competing predictions.

4. **Disease Severity Band**
   - Maps the predicted condition to a structured severity category using the project's clinical knowledge base.

5. **LLM-Generated Clinical Interpretation**
   - Produces a natural-language interpretation of the prediction.

These outputs are presented through the **Explainability Hub**, allowing users to examine more than the raw predicted class.

---

## Supported Medical Imaging Datasets

The system was trained and evaluated using seven publicly available datasets comprising **45,594 images**, covering four imaging modalities.

| Dataset | Modality | Images | Classes |
|---|---|---:|---:|
| Brain Tumor MRI | MRI | 7,200 | 4 |
| FracAtlas | X-ray / CT | 4,083 | 2 |
| TB Chest X-Ray | X-ray / CT | 4,200 | 2 |
| Pneumonia Chest X-Ray | X-ray / CT | 5,856 | 2 |
| Diabetic Retinopathy | Microscopic | 3,662 | 5 |
| Bone Fracture | X-ray / CT | 10,578 | 2 |
| HAM10000 Skin Cancer | Microscopic | 10,015 | 7 |

### Imaging Modalities

- 🧠 **MRI** — Brain tumor classification
- 🩻 **X-ray** — Fracture, tuberculosis and pneumonia classification
- 🖥️ **CT** — Used within FracAtlas and TB-related datasets
- 🔬 **Microscopic/Dermoscopic Imaging** — Diabetic retinopathy and skin cancer classification

---

## Data Preprocessing

The preprocessing pipeline standardizes medical images before they are passed to the Hybrid ViT model.

### Preprocessing Steps

- Image resizing
- Pixel/intensity normalization
- Noise reduction and smoothing
- Data augmentation
- Standardized input dimensions

Images are resized to a fixed resolution such as **224 × 224**, followed by normalization and augmentation to improve model robustness and generalization.

---

## Hybrid Vision Transformer

The proposed HViT architecture combines the complementary strengths of CNNs and Transformers.

### CNN Feature Extraction

The CNN component extracts:

- Local spatial features
- Edges
- Textures
- Fine-grained anatomical structures
- Local pathological patterns

### Patch Tokenization

CNN-generated feature maps are divided into non-overlapping patches and transformed into fixed-dimensional embeddings.

### Transformer Encoder

The Transformer processes the resulting token sequence using self-attention to capture:

- Global spatial relationships
- Long-range dependencies
- Contextual relationships between image regions

### Classification

The final Transformer representation is passed through a classification head to predict the disease class.

The hybrid architecture therefore combines **local feature learning** from CNNs with **global contextual modeling** from Transformers.

---

## Model Training

The experimental configuration reported in the project includes:

| Parameter | Configuration |
|---|---|
| Loss Function | Categorical Cross-Entropy |
| Optimizer | AdamW |
| Epochs | 40 |
| Batch Size | 32 |
| Learning Strategy | Learning Rate Scheduler |
| Framework | PyTorch |
| Training Platforms | Google Colab / Kaggle |

PyTorch was used for model implementation, training, evaluation and visualization, with GPU acceleration used during experimentation.

---

## Results

The system was evaluated using:

- Accuracy
- Weighted F1-score
- AUC
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Per-class Precision
- Per-class Recall
- Per-class F1-score

### Overall Dataset Results

| Dataset | Accuracy | F1 | AUC | MCC | Kappa |
|---|---:|---:|---:|---:|---:|
| Brain Tumor MRI | 98.7% | 0.987 | 0.999 | 0.982 | 0.982 |
| FracAtlas | 84.5% | 0.820 | 0.767 | 0.353 | 0.320 |
| TB Chest X-Ray | 97.0% | 0.970 | 0.994 | 0.894 | 0.893 |
| Pneumonia Chest X-Ray | 94.5% | 0.946 | 0.986 | 0.863 | 0.863 |
| Diabetic Retinopathy | 81.3% | 0.813 | 0.964 | 0.719 | 0.719 |
| Bone Fracture | 99.9% | 0.999 | 1.000 | 0.997 | 0.997 |
| HAM10000 Skin Cancer | 84.4% | 0.850 | 0.967 | 0.719 | 0.716 |

The report shows substantial variation across datasets, with class imbalance being a major factor affecting performance.

---

## System Features and UI

The implemented system provides multiple interfaces for interacting with the medical diagnosis platform.

### Main Features

- User authentication
- Home page
- Medical imaging modality selection
- Medical image upload
- Automated disease prediction
- Explainability Hub
- CACI interpretation
- Prediction dashboard
- Medical reports
- Full report generation
- Prediction history
- User profile

The final report documents screenshots for the Login, Home, Modalities, Diagnose, Explainability Hub, Dashboard, Reports, Full Reports, History and User Profile pages.

### Diagnosis Interface

Users can upload a medical image and obtain the corresponding model prediction.

### Explainability Hub

The Explainability Hub presents the structured CACI signals associated with the prediction.

### Dashboard

The dashboard provides an overview of evaluation and prediction-related information.

### Reports

The system provides detailed reports containing prediction and interpretation information.

---

## Technology Stack

### Programming

- Python

### Deep Learning

- PyTorch
- Hybrid CNN–Vision Transformer
- Transformer Encoder
- Self-Attention

### Medical Image Processing

- OpenCV
- NumPy

### Explainability

- Confidence-Aware Clinical Interpretation (CACI)
- Explainability Hub
- LLM-generated clinical interpretation

### Web Development

- Django
- HTML
- CSS
- JavaScript

The report specifically identifies Python, PyTorch, OpenCV, HTML and CSS as core technologies used in the system.

---

## Project Workflow

```text
                Medical Image
                     │
                     ▼
              Image Preprocessing
                     │
                     ▼
             CNN Feature Extraction
                     │
                     ▼
              Patch Tokenization
                     │
                     ▼
             Linear Projection
                     │
                     ▼
             Transformer Encoder
                     │
                     ▼
                Classification
                     │
                     ▼
              ┌──────────────┐
              │     CACI     │
              └──────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Confidence   Differential   Decision
      Band       Diagnosis      Margin
        │            │            │
        └────────────┼────────────┘
                     │
              Severity Band
                     │
                     ▼
          LLM Clinical Narrative
                     │
                     ▼
            Explainability Hub
                     │
                     ▼
              Clinical Report
```

---

## Project Structure

```text
Medical_Image/
│
├── manage.py
├── requirements.txt
│
├── Diagnosis/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── ...
│
├── templates/
│   ├── login/
│   ├── home/
│   ├── diagnose/
│   ├── explainability/
│   ├── dashboard/
│   ├── reports/
│   ├── history/
│   └── profile/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── models/
│   ├── brain_tumor/
│   ├── bone_fracture/
│   ├── tb/
│   ├── pneumonia/
│   ├── diabetic_retinopathy/
│   ├── skin_cancer/
│   └── fracatlas/
│
└── README.md
```

---

## Model Evaluation

The evaluation was conducted separately for each dataset because the datasets differ in:

- Number of classes
- Dataset size
- Imaging modality
- Class distribution
- Disease characteristics

The report emphasizes that accuracy alone can hide minority-class weaknesses. MCC, Cohen's Kappa and per-class metrics were therefore included to provide a more complete evaluation.

---

## Limitations

The project is a **research-oriented prototype** rather than a deployment-ready clinical system.

Current limitations include:

- No K-fold cross-validation.
- Significant class imbalance in some datasets.
- Weak minority-class performance in datasets such as FracAtlas.
- Limited performance on difficult multi-class severity classification.
- CACI has not yet undergone formal clinician validation.
- Clinical deployment requires further validation and testing.

The report explicitly states that CACI should currently be treated as a **structured decision-support aid rather than a replacement for clinical judgment**.

---

## Future Improvements

Planned future work includes:

- Class-weighted loss
- Focal loss
- Oversampling and SMOTE
- Targeted data collection
- K-fold cross-validation
- Clinician-validated CACI assessment
- Validated visual explainability
- Integration of Grad-CAM++, attention rollout and other attribution techniques
- Extension to additional medical imaging modalities
- Multi-task learning
- Further clinical evaluation

The report specifically proposes adding validated spatial explainability as a complement to the structured CACI signals.

---

## Dataset Sources

The project used publicly available datasets including:

1. **Bone Fracture Multi-Region X-ray Data**
2. **FracAtlas**
3. **Brain Tumor MRI Dataset**
4. **Tuberculosis Chest X-ray Database**
5. **Diabetic Retinopathy 224x224 Dataset**
6. **Skin Cancer MNIST: HAM10000**
7. **Chest X-Ray Images (Pneumonia)**

The dataset sources and corresponding links are documented in the project report.

---

## Repository

**GitHub Repository:**  
https://github.com/Aayushspk37/FinalProject

---

## Academic Project

**Project:** A Hybrid ViT with Confidence-Aware Clinical Interpretation for Medical Imaging Diagnosis

**Submitted by:**

- Aayush Sapkota — BEC [22070089]
- Sandip Lamsal — BEC [22070109]
- Suraj Jha — BEC [22070116]
- Yuresh Gurung — BEC [22070118]

**Supervisor:** Er. Prashant Poudel

**Institution:** United Technical College  
**Affiliation:** Pokhara University  
**Department:** Computer Engineering  
**Year:** 2026

---

## Disclaimer

This project is developed as an academic and research prototype for medical image classification and structured interpretation. It is **not intended to replace professional medical diagnosis or clinical judgment**.

The CACI layer has not yet undergone formal clinician validation, and further clinical evaluation is required before considering real-world clinical deployment.

---

## License

This project is developed for academic and research purposes.
