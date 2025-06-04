# Red Wine Quality Prediction 🍷✨

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--learn-0.24%2B-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-1.1%2B-green?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-1.19%2B-blueviolet?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-3.3%2B-yellowgreen?logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-0.11%2B-purple?logo=seaborn&logoColor=white" alt="Seaborn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter Notebook">
</p>

This project aims to predict the quality of red wine based on its physicochemical attributes. Using a dataset from the UCI Machine Learning Repository, we explore the data, preprocess it, build classification models, and evaluate their performance to determine which wine is 'good' qualité versus 'not good'.

## 📋 Table of Contents
1.  [Project Domain](#-project-domain)
2.  [Business Understanding](#-business-understanding)
3.  [Data Understanding](#-data-understanding)
4.  [Data Preparation](#-data-preparation)
5.  [Modeling](#-modeling)
6.  [Evaluation](#-evaluation)
7.  [How to Run](#-how-to-run)
8.  [File Structure](#-file-structure)
9. [Technologies Used](#-technologies-used)
10. [References](#-references)

## 🌍 Project Domain
The wine industry is a significant global market where quality is a key determinant of price and consumer satisfaction. Traditional wine quality assessment is often subjective and costly. This project explores using machine learning to provide an objective, data-driven approach to quality assessment based on standard laboratory tests.

## 🎯 Business Understanding

### Problem Statements
* How can we objectively classify red wine quality using physicochemical test data?
* What are the most significant chemical properties influencing red wine quality?
* Can a machine learning model efficiently and accurately distinguish between 'good' and 'not good' quality red wines?

### Goals
* To develop a classification model that predicts red wine quality (e.g., 'good' vs. 'not good') from its features.
* To identify key physicochemical drivers of wine quality.
* To achieve high accuracy and F1-score for reliable predictions.

### Solution Statement
* Implement and compare two classification algorithms: Logistic Regression (as a baseline) and Random Forest Classifier.
* Optimize the best-performing model using hyperparameter tuning (GridSearchCV).
* Evaluate models using metrics like Accuracy, Precision, Recall, and F1-score.

## 📊 Data Understanding
The project utilizes the **Red Wine Quality dataset** from the UCI Machine Learning Repository.
* **Source:** [Wine Quality Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* **Instances:** 1599
* **Features:** 11 physicochemical input features and 1 quality output variable (score from 3 to 8).
    1.  `fixed acidity`
    2.  `volatile acidity`
    3.  `citric acid`
    4.  `residual sugar`
    5.  `chlorides`
    6.  `free sulfur dioxide`
    7.  `total sulfur dioxide`
    8.  `density`
    9.  `pH`
    10. `sulphates`
    11. `alcohol`
    12. `quality` (Target Variable: Transformed into binary 0 for 'not good' [<=5] and 1 for 'good' [>5])

Exploratory Data Analysis (EDA) was performed to understand feature distributions, correlations, and the target variable.

## 🛠️ Data Preparation
The following steps were taken to prepare the data for modeling:
1.  **Loading Data:** Loaded the `winequality-red.csv` dataset.
2.  **Handling Missing Values:** Checked for and confirmed no missing values.
3.  **Target Variable Transformation:** Converted the `quality` score (3-8) into a binary target `quality_category`:
    * `0` (Not Good): if `quality` <= 5
    * `1` (Good): if `quality` > 5
4.  **Train-Test Split:** Divided the dataset into training (80%) and testing (20%) sets, stratified by the target variable.
5.  **Feature Scaling:** Standardized numerical features using `StandardScaler` (fit on training data, transformed on training and testing data).

## 🤖 Modeling
Two classification models were developed:

1.  **Logistic Regression:**
    * Used as a baseline model.
    * Simple, interpretable, and efficient.
2.  **Random Forest Classifier:**
    * An ensemble learning method known for high accuracy and robustness.
    * Capable of capturing non-linear relationships and providing feature importance.

**Hyperparameter Tuning:**
The Random Forest model was further optimized using `GridSearchCV` to find the best combination of parameters (e.g., `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`), aiming to maximize the F1-score (weighted).

## 📈 Evaluation
Model performance was assessed using the following metrics on the test set:
* **Accuracy:** Overall correctness of predictions. `(TP + TN) / (TP + TN + FP + FN)`
* **Precision:** Ability of the classifier not to label as positive a sample that is negative. `TP / (TP + FP)`
* **Recall (Sensitivity):** Ability of the classifier to find all the positive samples. `TP / (TP + FN)`
* **F1-score:** Weighted average of Precision and Recall. `2 * (Precision * Recall) / (Precision + Recall)`
    * (Weighted averages for Precision, Recall, and F1-score were used to account for class distribution.)

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/red-wine-quality-predictor.git](https://github.com/your-username/red-wine-quality-predictor.git)
    cd red-wine-quality-predictor
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment after installing all necessary libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter).*
4.  **Download the dataset:**
    Ensure `winequality-red.csv` is in the project's root directory or update the path in the notebook. You can download it from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv).
5.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
6.  Open and run the `.ipynb` notebook (e.g., `Red_Wine_Quality_Prediction.ipynb`).

## 📂 File Structure
