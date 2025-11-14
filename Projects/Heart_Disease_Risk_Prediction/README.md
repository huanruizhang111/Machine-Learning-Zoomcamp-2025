# Heart Disease Risk Prediction

## Problem Description

This project aims to build a machine learning model to predict the 10-year risk of coronary heart disease (CHD) for patients based on their demographic, behavioral, and medical information.

### Dataset

The dataset used is the Framingham Heart Study dataset from Kaggle:
https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression?select=framingham.csv

The dataset contains 4,238 patient records with 16 features including:
- **Demographic features**: gender, age
- **Behavioral features**: education level, smoking status, daily cigarette consumption
- **Medical history**: blood pressure medication use, previous stroke, hypertension, diabetes
- **Current medical measurements**: total cholesterol, systolic/diastolic blood pressure, BMI, heart rate, glucose level
- **Target variable**: 10-year coronary heart disease risk (binary: 1 = Yes, 0 = No)

### Objective

Build and compare multiple classification models to accurately predict the probability of a patient developing coronary heart disease within 10 years, enabling early intervention and preventive care.

### Models Evaluated

1. **Logistic Regression** - Baseline model with hyperparameter tuning on C (regularization strength)
2. **Support Vector Machine (SVM)** - Tested with multiple kernels (rbf, linear, poly, sigmoid) and hyperparameters
3. **Decision Tree Classifier** - Evaluated with varying max_depth and min_samples_leaf
4. **Random Forest Classifier** - Ensemble model with tuning on n_estimators, max_depth, and min_samples_leaf
5. **XGBoost Classifier** - Gradient boosting model with optimization on learning_rate, max_depth, and reg_lambda

### Evaluation Metric

**ROC AUC (Area Under the Receiver Operating Characteristic Curve)** was used as the primary evaluation metric with 5-fold cross-validation to assess model performance. Since this dataset is imbalanced in its target variable, it is not suitable to use accuracy to be the evaluation metric. That's why I use ROC AUC in this project.

### Best Model

**Logistic Regression with C=0.9** achieved the highest validation AUC score and was selected for final deployment. The model was trained on the full training set and serialized using scikit-learn's Pipeline for production use.

### Key Steps

1. **Data Cleaning**: Handled missing values using mode (categorical) and median (numerical) imputation
2. **Feature Engineering**: Mapped categorical values to meaningful labels and renamed columns for readability
3. **Data Splitting**: 60% train, 20% validation, 20% test split using stratified sampling
4. **Feature Vectorization**: Used DictVectorizer to convert dictionary records into numerical feature matrices
5. **Model Training & Evaluation**: Trained multiple models with hyperparameter tuning using k-fold cross-validation
6. **Model Selection**: Selected Logistic Regression as the final model based on validation AUC
7. **Deployment**: Serialized the model as a pickle file and created a FastAPI service for predictions

### Output of The Model

The final model outputs the probability of a patient having 10-year CHD risk, which can be used for clinical decision-making and patient stratification.

## How to Run The Project

1. **Download** `Framingham Heart Study dataset` from Kaggle and put it in the folder your are going to used for reproducing this project:
https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression?select=framingham.csv
2. **Execute** `notebook.ipynb` and you can see the result. However, `the final model (model.bin)` used in this project will be generate in `train.py` script file. Therefore, please do not use the chd_rish_model.bin because it is only used to check if I can successfully save the model in pickle file and read it to use.
3. **Execute** `train.py` to train and save the final model. `model.bin` will be saved automatically in the save folder.
4. 