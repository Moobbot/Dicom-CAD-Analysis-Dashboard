# CAD Analysis Dashboard for COVID-19 Radiomics

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/DL-TensorFlow%20%2F%20UNet-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Introduction

The **CAD Analysis Dashboard** is an interactive Computer-Aided Diagnosis (CAD) and exploratory data analysis (EDA) platform designed for **COVID-19 vs. Normal** chest CT/X-ray radiomics evaluation. 

The end-to-end pipeline integrates:
1. **Deep Learning Segmentation (U-Net)**: Automated lung lesion & parenchyma segmentation.
2. **Medical Radiomics Feature Extraction (PyRadiomics & SimpleITK)**: Multi-angle extraction of first-order, texture (GLCM, GLDM, GLSZM, GLRLM), and shape statistics.
3. **Interactive Visual Dashboard (Streamlit & Plotly)**: Comprehensive statistical summaries, hypothesis testing (two-sample t-test, normality tests), PCA dimensional reduction, and comparative machine learning classification (Random Forest, SVM, Logistic Regression).

---

## Key Features

- **Dataset Overview & Missing Value Imputation**:
  - Displays row/column dimensions, numerical vs. categorical feature summaries.
  - Interactive missing value imputation (Mean, Median, Zero-fill, Drop).
- **Descriptive Statistics & KPIs**:
  - Real-time computation of central tendency (Mean, Median, Mode) and variability (Std Dev, IQR) metrics.
- **Hypothesis Testing & Statistical Inference**:
  - Two-sample Welch's t-test comparing COVID-19 vs. Normal cohorts.
  - Normality testing using Anderson-Darling and Kolmogorov-Smirnov (KS) tests before and after data transformation.
- **Dimensionality Reduction & Clustering**:
  - Principal Component Analysis (PCA) with 2D interactive projection and explained variance ratio tables.
- **Machine Learning Classification & Evaluation**:
  - 5-Fold Cross Validation comparison across **Random Forest**, **Support Vector Machine (SVC)**, and **Logistic Regression**.
  - Interactive visualizations for ROC-AUC curves, Confusion Matrices, and Random Forest feature importance rankings.
- **Interactive Visual Comparison**:
  - Plotly interactive correlation heatmaps, top correlated feature bar charts, box plots, and QQ-plots before/after normalization (Min-Max, Z-score, Max-Abs).

- **Deep Learning & Radiomics Clinical Diagnosis (New Image Diagnosis)**:
  - **Full DICOM Support (`.dcm`, `.dicom`)**: Reads clinical CT scans, applies Hounsfield Unit (HU) conversion with medical Lung Windowing (WL: -600, WW: 1500 HU). Standard image formats (`.jpg`, `.png`) are also fully supported.
  - **Multi-Image / Multi-Slice Upload**: Batch upload sequential CT slices (`accept_multiple_files=True`) with automatic natural numerical sorting.
  - **Overall Clinical Conclusion**: Patient-level diagnostic verdict and confidence aggregation (e.g., *"Cả 11 lát cắt của bệnh nhân đều được chẩn đoán là NORMAL, với xác suất trung bình 71.3%"*).
  - **Animated Cine-Loop GIF**: Generates animated GIF loops scrolling through the CT series with side-by-side or mask overlay views, adjustable frame rate, and one-click GIF download.
  - **Summary Diagnosis Table & CSV Export**: Per-slice diagnosis breakdown with probability scores, confidence ratings, and instant CSV export.
  - **Individual Slice Inspector**: Detailed examination of lung mask segmentation, cyan lesion overlay, 24 extracted radiomic biomarkers, and DICOM metadata header.

---

## Architecture & Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend Dashboard** | Streamlit, Plotly Express | Interactive web dashboard, multi-image diagnosis, and reactive controls |
| **Backend & Modeling** | Python, Scikit-learn, SciPy | Statistical tests, PCA, and calibrated Random Forest classification |
| **Deep Learning** | TensorFlow / Keras (U-Net) | 2D U-Net architecture for automated lung parenchyma segmentation |
| **Medical Imaging** | pydicom, Pillow, OpenCV | DICOM parsing, Hounsfield Unit conversion, Cine-Loop GIF generation |
| **Feature Extraction** | PyRadiomics, SimpleITK | 24 core radiomic texture (GLCM, GLDM, GLSZM, GLRLM) and first-order features |
| **Database** | SQLite, SQLAlchemy | Persistent storage of extracted radiomic features (3,449 samples) |

---

## Quick Start & Installation

### 1. Prerequisites
- Python 3.10 is recommended (Python 3.8 - 3.10 supported).
- Git.

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/cad-analysis-dashboard.git
cd CAD-Analysis-Dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install core requirements for the Web Dashboard, DICOM processing & Inference:
pip install -r requirements.txt

# (Optional) Install full ML & Radiomics extraction requirements:
pip install -r requirements-ml.txt
```

### 4. Run the Application
```bash
streamlit run "Analysis Dashboard/dashboard.py"
```
The application will open automatically in your browser at `http://localhost:8501`.

### 5. Run CLI Diagnosis (Command Line)
```bash
# Diagnose a single CT scan or DICOM slice:
python predict.py --image "sample_data/data-test/Tuong_20230828_52.dcm" --output output/

# Batch diagnose an entire CT scan directory:
python predict.py --dir "sample_data/data-test" --output output/
```

For detailed Vietnamese instructions and troubleshooting, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## Directory Structure

```text
CAD-Analysis-Dashboard/
├── Analysis Dashboard/
│   ├── dashboard.py                  # Main Streamlit Dashboard application
│   ├── database_setup.py             # Script to load CSV features into SQLite DB
│   ├── ex.py                         # SQLite query sample test
│   ├── extracted_features_Covid.csv  # Extracted radiomics for COVID cases (2,246 samples)
│   ├── extracted_features_normal.csv # Extracted radiomics for Normal cases (1,203 samples)
│   ├── params.yaml                   # PyRadiomics extraction parameters
│   └── radiomics_data.db             # SQLite database storing feature table
├── ML/
│   ├── Feature Extraction/           # PyRadiomics extraction notebooks
│   ├── Random_Classifier.ipynb       # ML experiments (Random Forest, SMOTE, etc.)
│   └── UNET Training/                # U-Net segmentation training and weights (.h5)
├── models/
│   └── cad_classifier.joblib         # Serialized Random Forest classification pipeline
├── modules/
│   ├── eda.py                        # EDA, statistical analysis & ML pipeline
│   └── inference.py                  # End-to-end inference engine (U-Net + Radiomics + DICOM + GIF)
├── output/                           # Output directory for saved masks, overlays, and batch CSV
├── predict.py                        # Standalone CLI tool for single/batch diagnosis
├── sample_data/
│   ├── data-test/                    # 11 clinical DICOM CT slices for testing
│   ├── create_sample.py              # Script to generate synthetic test slices
│   └── sample_chest_ct.jpg           # Sample CT scan image
├── requirements.txt                  # Dependencies for Dashboard, DICOM & EDA
├── requirements-ml.txt               # Dependencies for DL & Radiomics
├── ARCHITECTURE.md                   # Complete system architecture specification
├── SETUP_GUIDE.md                    # Detailed Vietnamese setup guide
└── README.md                         # Project overview and documentation
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
