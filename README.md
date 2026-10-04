# 🚢 Comparative Study of Classification Evaluation Metrics on the Titanic Dataset

An SVM classifier predicts Titanic passenger survival, and the model is evaluated with **multiple metrics beyond accuracy**: confusion matrix, precision, recall and F1-score. The project shows why a single number like accuracy can hide weaknesses in a model.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Metric Comparison](#-metric-comparison)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔎 Overview

The goal is to predict whether a passenger survived (`1`) or not (`0`) and then answer a bigger question: **how do we judge a classifier properly?**

The study covers:

- What accuracy, precision, recall and F1-score each measure
- When each metric matters most (e.g. spam detection vs. medical diagnosis)
- Why accuracy alone can be misleading on imbalanced data
- Which metric is most meaningful for this dataset

---

## 📂 Dataset

- **Source:** [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)
- **Size:** 891 passengers
- **Target:** `Survived` (0 = No, 1 = Yes)
- **Class balance:** about 62% did not survive, 38% survived

| Feature | Description |
|---|---|
| `Pclass` | Ticket class (1, 2, 3) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket fare |
| `Embarked` | Port of embarkation (C, Q, S) |

---

## ⚙️ Project Workflow

1. **Load data** from `Titanic-Dataset.csv`
2. **Drop columns** with high cardinality or many missing values: `Name`, `Ticket`, `Cabin`
3. **Handle missing values**
   - `Age` filled with the median
   - `Embarked` filled with the mode
4. **Encode features**
   - `Sex` mapped to binary (male = 0, female = 1)
   - `Embarked` one-hot encoded (`Embarked_C`, `Embarked_Q`, `Embarked_S`)
5. **Split data**: 80% train / 20% test (`random_state=42`)
6. **Scale features** with `StandardScaler`
7. **Train model**: SVM with RBF kernel (`C=1.0`, `gamma='scale'`)
8. **Evaluate** with accuracy, confusion matrix and classification report

---

## 📊 Results

### Overall Accuracy

| Metric | Value |
|---|---|
| **Accuracy** | **0.81** (0.8101) |

### Confusion Matrix

|  | Predicted: Not Survived | Predicted: Survived |
|---|:---:|:---:|
| **Actual: Not Survived** | 93 (TN) | 12 (FP) |
| **Actual: Survived** | 22 (FN) | 52 (TP) |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|:---:|:---:|:---:|:---:|
| 0 (Not Survived) | 0.81 | 0.89 | 0.85 | 105 |
| 1 (Survived) | 0.81 | 0.70 | 0.75 | 74 |
| **Macro avg** | 0.81 | 0.79 | 0.80 | 179 |
| **Weighted avg** | 0.81 | 0.81 | 0.81 | 179 |

---

## 💡 Key Insights

- The model reaches **81% accuracy**, well above the ~62% a "predict everyone died" baseline would get.
- **Recall for survivors (0.70) is much lower than for non-survivors (0.89).** The model misses roughly 30% of actual survivors, which accuracy alone does not reveal.
- Precision for survivors is **0.81**: when the model predicts survival, it is right about 81% of the time.
- For this dataset, the **F1-score of the positive class (survived)** is the most meaningful single metric, since it balances precision and recall on the minority class.

---

## ⚖️ Metric Comparison

| Metric | What it answers | Most important when |
|---|---|---|
| **Accuracy** | How often is the model correct overall? | Classes are balanced and errors cost the same |
| **Precision** | Of all predicted positives, how many were truly positive? | False positives are costly (e.g. spam detection) |
| **Recall** | Of all actual positives, how many did we find? | False negatives are costly (e.g. medical diagnosis) |
| **F1-score** | What is the balance between precision and recall? | Classes are uneven and both error types matter |

**Precision vs. Recall trade-off:** lowering the decision threshold catches more survivors (higher recall) but produces more false alarms (lower precision), and vice versa.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook Comparative_Study_of_Classification_Evaluation_Metrics.ipynb
```

Or open it directly in **Google Colab** and upload `Titanic-Dataset.csv`.

> **Note:** the notebook reads the data from `/content/Titanic-Dataset.csv` (the Colab path). If you run it locally, change the path in the `pd.read_csv(...)` line to match where the CSV is saved, for example `'Titanic-Dataset.csv'`.

---

## 📁 Repository Structure

```
├── Comparative_Study_of_Classification_Evaluation_Metrics.ipynb   # Code and outputs
├── Comparative_Study_of_Classification_Evaluation_Metrics.pdf     # Full written report
├── Titanic-Dataset.csv                                            # Dataset
├── requirements.txt                                               # Dependencies
└── README.md
```

### `requirements.txt`

```
pandas
numpy
scikit-learn
jupyter
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn
- **Model:** Support Vector Machine (RBF kernel)
- **Environment:** Google Colab / Jupyter Notebook

---

## 🔮 Future Improvements

- Tune hyperparameters (`C`, `gamma`) with `GridSearchCV`
- Use cross-validation for more reliable estimates
- Compare against other models (Logistic Regression, Random Forest, XGBoost)
- Add ROC curve, AUC and precision-recall curve
- Engineer new features such as titles from `Name` and family size
- Explore decision-threshold tuning to improve survivor recall

---

## 👤 Author

**Your Name**

- LinkedIn: [your-linkedin-profile](https://www.linkedin.com/in/your-profile)
- GitHub: [@your-username](https://github.com/your-username)

---

⭐ If you found this project useful, consider giving it a star!
