# Red Wine Quality Prediction

A machine learning project focused on predicting the quality of red wine based on its physicochemical properties. This project implements and compares different classification models to achieve a reliable prediction system.

## Project Overview

The primary goal of this project is to develop a machine learning model that can classify red wine into quality categories (e.g., "good" vs. "not good") using a publicly available dataset. This involves data exploration, preprocessing, model training, hyperparameter tuning, and evaluation.

## Dataset

The project utilizes the **Red Wine Quality dataset** from the UCI Machine Learning Repository.
- **Source:** [Wine Quality Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Citation:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547-553.

The dataset contains 1599 instances of red wine samples with 11 physicochemical input variables and one output variable (quality score from 3 to 8). For this project, the quality score is transformed into a binary target variable (e.g., 0 for "not good" quality, 1 for "good" quality).

## Problem Statement & Goals

- **Problem:** To objectively assess red wine quality based on its physicochemical properties, reducing subjectivity and a_nd_inconsistency associated with human tasters.
- **Goals:**
    1. To develop a machine learning model capable of accurately classifying red wine quality.
    2. To identify key physicochemical features influencing red wine quality.
    3. To compare the performance of different classification algorithms (Logistic Regression and Random Forest).
    4. To optimize the best-performing model using hyperparameter tuning.

## Methodology

1.  **Data Understanding & EDA:** Loading the dataset, exploring feature distributions, visualizing correlations, and understanding basic statistics.
2.  **Data Preparation:**
    * Handling any missing values (if present, though typically none in this dataset).
    * Transforming the multi-class `quality` variable into a binary target variable.
    * Splitting the data into training and testing sets.
    * Applying feature scaling (StandardScaler) to the input features.
3.  **Modeling:**
    * Training a **Logistic Regression** model as a baseline.
    * Training a **Random Forest Classifier** model.
    * Comparing their performance using metrics like Accuracy, Precision, Recall, and F1-score.
4.  **Hyperparameter Tuning:**
    * Optimizing the Random Forest model (or the best-performing model) using `GridSearchCV` to find the best set of hyperparameters.
5.  **Evaluation:**
    * Evaluating the final tuned model on the test set.
    * Analyzing the classification report and confusion matrix.
    * Discussing feature importances derived from the Random Forest model.

## File Structure

-   `wine-quality-prediction.ipynb`: Jupyter Notebook containing the complete Python code for data loading, preprocessing, EDA, modeling, and evaluation.
-   `format_laporan_submission_1.md` (or your report's filename, e.g., `Project_Report.md`): Detailed project report in Markdown format.
-   `main.py` (or similar): A supplementary Python script (if required for submission structure, may simply point to the notebook).
-   `winequality-red.csv`: The dataset file (ensure this is present or provide instructions on how to obtain it).
-   `README.md`: This file, providing an overview of the project.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/red-wine-quality-prediction.git](https://github.com/your-username/red-wine-quality-prediction.git)
    cd red-wine-quality-prediction
    ```
2.  **Ensure you have Python installed.** (Python 3.7+ recommended)
3.  **Install necessary libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
4.  **Dataset:** Make sure the `winequality-red.csv` dataset file is in the project's root directory. If not, download it from the [UCI link](https://archive.ics.uci.edu/ml/datasets/wine+quality) and place it there.
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook wine-quality-prediction.ipynb
    ```
    Execute the cells in the notebook sequentially.

## Results Summary

The project aims to achieve a robust classification model. Key performance metrics (Accuracy, Precision, Recall, F1-score) for Logistic Regression, default Random Forest, and tuned Random Forest are documented within the Jupyter Notebook and the project report. The tuned Random Forest model is expected to provide the best performance. Feature importance analysis also reveals which physicochemical properties are most influential in determining wine quality.

For detailed results and discussion, please refer to the `wine-quality-prediction.ipynb` notebook and the `Project_Report.md` (or your specific report file).
