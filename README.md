# 🧠 Multimodal Deep Tree-Based Ensemble for Early Alzheimer's Identification

This repository presents a robust machine learning framework designed for the early identification of Alzheimer's disease (AD). By integrating multimodal data sources—including neuroimaging, clinical assessments, and genetic information—into a deep tree-based ensemble model, the system aims to enhance diagnostic accuracy and facilitate timely intervention.

---

## 📌 Overview

Alzheimer's disease is a complex neurodegenerative disorder characterized by progressive cognitive decline. Early and accurate diagnosis is crucial for effective management and intervention. Traditional diagnostic methods often rely on single-modal data, which may not capture the full spectrum of information necessary for precise diagnosis.

Our approach leverages a multimodal dataset comprising:

- **Neuroimaging Data:** Structural MRI scans providing insights into brain morphology.
- **Clinical Assessments:** Standardized cognitive and functional evaluations.
- **Genetic Information:** Genetic markers associated with AD risk.

These diverse data sources are processed and integrated into a unified framework, enabling the model to learn complex patterns and relationships indicative of early-stage Alzheimer's disease.

---

## 🧬 Methodology

The proposed system employs a deep tree-based ensemble model that combines the strengths of deep learning and tree-based algorithms. The methodology includes:

1. **Data Preprocessing:** Standardization and normalization of multimodal data to ensure compatibility and enhance model performance.
2. **Feature Extraction:** Utilization of advanced techniques to extract meaningful features from neuroimaging, clinical, and genetic data.
3. **Model Training:** Implementation of a deep tree-based ensemble model that integrates multiple base learners to improve predictive accuracy.
4. **Evaluation:** Rigorous assessment of model performance using cross-validation and external validation datasets.

This integrated approach allows for the simultaneous consideration of multiple data modalities, leading to a more comprehensive understanding of Alzheimer's disease pathology.

---

### Key Features:
- Combines **clinical** and **neuroimaging** data for Alzheimer’s prediction.
- Uses **Deep Neural Networks** for feature extraction.
- Applies **Random Forest** classifiers for robust prediction.
- Achieves up to **92% balanced accuracy** and **97% AUC**, outperforming existing studies.

---

## Repository Structure
The repository includes the following files:

### Data:
- **`AD_raw_data.csv`**: Raw dataset containing clinical and neuroimaging features used in the study.

### Code:
- **`preprocess.py`**: Code for data preprocessing, including missing value imputation and dataset balancing using SMOTE.
- **`train_ann.py`**: Implementation of the Deep Neural Network (DNN) for feature extraction.
- **`train_classifiers.py`**: Code for training Random Forest and other classifiers on extracted features.
- **`train_tabnet.py`**: Implementation of the TabNet model for additional experimentation.
- **`models.py`**: Contains configurations for different models and classifiers used in the study.
- **`plot.py`**: Utility functions for visualizing results, including feature distributions and performance metrics.

---

## Results
The proposed framework achieves the following results:
1. **Using Clinical Data Only**:
   - Balanced Accuracy: **88%**
   - AUC: **94.6**
2. **Combining Clinical and Neuroimaging Data**:
   - Balanced Accuracy: **92%**
   - AUC: **97%**

These results demonstrate the effectiveness of multimodal data in improving predictive performance for early Alzheimer’s detection.

---

## Prerequisites
- Python 3.8 or higher
- Libraries listed in `requirements.txt`

### Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

---

## Usage
### 1. Data Preprocessing
Use the `preprocess.py` script to clean, impute missing values, and balance the dataset:
```bash
python preprocess.py
```

### 2. Train the Models
- **ANN Training**:
  ```bash
  python train_ann.py
  ```
- **Classifier Training**:
  ```bash
  python train_classifiers.py
  ```
- **TabNet Training**:
  ```bash
  python train_tabnet.py
  ```

### 3. Visualizations
Generate plots using `plot.py`:
```bash
python plot.py
```

