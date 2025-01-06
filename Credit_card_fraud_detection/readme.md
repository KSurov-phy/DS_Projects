# Credit Card Fraud Detection

## Project Overview
This project aims to detect fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, with fraudulent transactions being a small minority. To address this imbalance, we applied SMOTE (Synthetic Minority Oversampling Technique) to balance the classes, enabling effective model training.

### Key Objectives:
- Analyze the dataset and visualize transaction patterns.
- Address class imbalance with SMOTE.
- Train and evaluate machine learning models.
- Compare models based on performance metrics like AUC and training time.

## Dataset
The dataset used in this project is the **Credit Card Fraud Detection Dataset** available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Features:
- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset.
- **Amount**: Transaction amount.
- **V1 to V28**: Principal Component Analysis (PCA)-transformed features.
- **Class**: Target variable (1 = Fraud, 0 = Legitimate).

## Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Visualized transaction times and class distributions.
   - Removed duplicates and aggregated transaction data by hour.

2. **Preprocessing**:
   - Standardized features using `StandardScaler`.
   - Split the data into training and testing sets.
   - Applied SMOTE to handle class imbalance.

3. **Modeling**:
   - Trained and evaluated the following models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - XGBoost
   - Compared models using ROC-AUC scores and training times.

4. **Evaluation**:
   - Metrics used: Confusion Matrix, Classification Report, ROC-AUC.
   - Visualized ROC curves for comparison.

## Results
- **Model Performances**:
  - Logistic Regression: AUC = 0.98, Training Time = 0.94 seconds
  - Decision Tree: AUC = 0.93, Training Time = 39.25 seconds
  - Random Forest: AUC = 0.97, Training Time = 309.22 seconds
  - XGBoost: AUC = 0.99, Training Time = 2.01 seconds

- **Best Model**: XGBoost achieved the highest AUC score while maintaining a reasonable training time, making it the best model for this task.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd credit-card-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run the `credit_card_fraud_detection.ipynb` notebook.

## Acknowledgments
- Dataset: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `xgboost`.

## Contact
For questions or suggestions, feel free to contact [Your Name](mailto:your.email@example.com).

