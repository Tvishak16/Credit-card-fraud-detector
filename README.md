

# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning algorithms. The dataset used contains transactions made by credit cards in September 2013 by European cardholders.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, clone the repository and install the required packages.

```bash
git clone 
https://github.com/Tvishak16/Credit-card-fraud-detector.git

cd Credit-card-fraud-detector
```

### Requirements

Before running the notebook, you need to have the following packages installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Open the Jupyter Notebook:

```bash
jupyter notebook Credit_card.ipynb
```

Run the cells in the notebook sequentially to reproduce the results. The notebook includes data loading, preprocessing, model training, evaluation, and visualization steps.

## Project Overview

The notebook performs the following steps:

1. **Data Loading**: Load the credit card transaction data.
2. **Data Preprocessing**: Clean and preprocess the data.
3. **Exploratory Data Analysis**: Visualize the data to understand patterns and correlations.
4. **Model Training**: Train various machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine (SVM).
5. **Model Evaluation**: Evaluate the performance of the models using metrics such as accuracy, confusion matrix, ROC AUC score, etc.
6. **Dimensionality Reduction**: Use PCA for dimensionality reduction.
7. **Model Comparison**: Compare the performance of different models.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

## Requirements

The notebook imports the following libraries, so ensure these are installed:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
```

