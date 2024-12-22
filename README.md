# (https://doi.org/10.1007/978-3-031-21595-7_3) A Novel Diagnostic Model for Early Detection of Alzheimer’s Disease Based on Clinical and Neuroimaging Features

---

## Project Overview
This repository contains the implementation of a novel diagnostic framework for the **early detection of Alzheimer’s Disease**. The study demonstrates how multimodal data—clinical and neuroimaging—can be leveraged to improve predictive accuracy and reliability. The framework integrates **Deep Neural Networks** for feature extraction and **Random Forest** classifiers for final predictions, achieving state-of-the-art results in balanced accuracy and AUC.

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

### Other Files:
- **`requirements.txt`**: List of required Python packages and dependencies.
- **`README.md`**: Documentation of the project (this file).
- **`LICENSE`**: License for the project.

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

---

## References
If you use this work in your research, please cite:
```plaintext
@InProceedings{10.1007/978-3-031-21595-7_3,
author="Gad, Eyad
and Gamal, Aya
and Elattar, Mustafa
and Selim, Sahar",
editor="Fournier-Viger, Philippe
and Hassan, Ahmed
and Bellatreche, Ladjel",
title="A Novel Diagnostic Model for Early Detection of Alzheimer's Disease Based on Clinical and Neuroimaging Features",
booktitle="Model and Data Engineering",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="26--39",
abstract="Alzheimer's Disease (AD) is a dangerous disease that is known for its characteristics of eroding memory and destroying the brain. The classification of Alzheimer's disease is an important topic that has recently been addressed by many studies using Machine Learning (ML) and Deep Learning (DL) methods. Most research papers tackling early diagnosis of AD use these methods as a feature extractor for neuroimaging data. In our research paper, the proposed algorithm is to optimize the performance of the prediction of early diagnosis from the multimodal dataset by a multi-step framework that uses a Deep Neural Network (DNN) as an optimization technique to extract features and train these features by Random Forest (RF) classifier. The results of the proposed algorithm showed that using only demographic and clinical data results in a balanced accuracy of 88{\%} and an area under the curve (AUC) of 94.6. Ultimately, combining clinical and neuroimaging features, prediction results improved further to a balanced accuracy of 92{\%} and an AUC of 97{\%}. This study successfully outperformed other studies for both clinical and the combination of clinical and neuroimaging data, proving that multimodal data is efficient in the early diagnosis of AD.",
isbn="978-3-031-21595-7"
}
```
