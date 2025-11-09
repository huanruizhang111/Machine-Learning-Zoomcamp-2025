#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

C = 10
max_iter = 1000
n_splits = 5

def load_data():
    print('Loading data ...')
    df = pd.read_csv('framingham.csv')

    df['education'] = df['education'].fillna(df['education'].mode()[0])

    df.loc[df['currentSmoker']==1, 'cigsPerDay'] = df.loc[df['currentSmoker']==1, 'cigsPerDay'].fillna(df.loc[df['currentSmoker']==1, 'cigsPerDay'].mode()[0])
    df.loc[df['currentSmoker']==0, 'cigsPerDay'] = df.loc[df['currentSmoker']==0, 'cigsPerDay'].fillna(0)


    df['BPMeds'] = df['BPMeds'].fillna(0)
    df['totChol'] = df['totChol'].fillna(df['totChol'].median())
    df['BMI'] = df['BMI'].fillna(df['BMI'].mode()[0])
    df['heartRate'] = df['heartRate'].fillna(df['heartRate'].median())

    df.loc[df['diabetes']==1, 'glucose'] = df.loc[df['diabetes']==1, 'glucose'].fillna(df.loc[df['diabetes']==1, 'glucose'].mode()[0])
    df.loc[df['diabetes']==0, 'glucose'] = df.loc[df['diabetes']==0, 'glucose'].fillna(0)



    df['male'] = df['male'].map({1:'male', 0:'female'})
    df['education'] = df['education'].map({1:'less_than_high_school', 2:'high_school', 3:'some_college_or_vocational_school', 4:'college_or_above'}) 
    df['currentSmoker'] = df['currentSmoker'].map({1:'smoker', 0:'non_smoker'})
    df['BPMeds'] = df['BPMeds'].map({1:'on_bp_meds', 0:'not_on_bp_meds'})
    df['diabetes'] = df['diabetes'].map({1:'diabetic', 0:'non_diabetic'})



    numericals_rename = {'age':'age',
                        'cigsPerDay':'cigarettes_per_day',
                        'totChol':'total_cholesterol',
                        'sysBP':'systolic_blood_pressure',
                        'diaBP':'diasolic_blood_pressure',
                        'BMI':'bmi',
                        'heartRate':'heart_rate',
                        'glucose':'glucose_level'
    }

    categoricals_rename = {'male':'gender',
                'education':'education_level',
                'currentSmoker':'smoker',
                'BPMeds':'blood_pressure_medication',
                'prevalentStroke':'had_a_stroke',
                'prevalentHyp':'hypertensive',
                'diabetes':'diabetes',
                'TenYearCHD':'10year_chd_risk'}

    numericals = list(numericals_rename.values())
    categoricals = list(categoricals_rename.values())
    categoricals.remove('10year_chd_risk')

    df = df.rename(columns=numericals_rename)
    df = df.rename(columns=categoricals_rename)
    
    print('Data loaded successfully!')

    return df, categoricals, numericals

def k_fold_auc(df_train_full, categoricals, numericals, c=1.0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    kf_train_auc = []
    kf_val_auc = []
    fold = 1
    for train_index, val_index in kf.split(df_train_full):
        print(f'Training in {fold} fold ...')
        # prepare train data
        df_train = df_train_full.iloc[train_index]
        train_dict = df_train[categoricals + numericals].to_dict(orient='records')
        dv = DictVectorizer(sparse=False)
        dv.fit(train_dict)

        # train the model
        X_train = dv.transform(train_dict)
        y_train = df_train['10year_chd_risk'].values
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train, y_train)
        
        # make predictions on train data
        y_train_pred = model.predict_proba(X_train)[:, 1]

        # prepare validation data
        df_val = df_train_full.iloc[val_index]
        val_dict = df_val[categoricals + numericals].to_dict(orient='records')
        X_val = dv.transform(val_dict)
        y_val = df_val['10year_chd_risk'].values

        # make predictions on validation data
        y_val_pred = model.predict_proba(X_val)[:, 1]

        # evaluate the model on train data and store the AUC score in a list
        kf_train_auc.append(round(roc_auc_score(y_train, y_train_pred), 3))
    
        # evaluate the model on validation data and store the AUC score in a list
        
        kf_val_auc.append(round(roc_auc_score(y_val, y_val_pred), 3))

        fold += 1
    
    print(f'Finsh K-fold: C={C} with AUC {round(np.mean(kf_val_auc), 3)} +- {round(np.std(kf_val_auc), 3)}')
    
    return kf_train_auc, kf_val_auc


def train_model(df):
    print('Training the model on the full dataset ...')
    y_train = df['10year_chd_risk'].values
    train_dict = df[categoricals + numericals].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear', C=C, max_iter=max_iter)
    )

    pipeline.fit(train_dict, y_train)

    print('Finish: Training the model on the full dataset ...')
    return pipeline

def save_model(pipeline, output_file):
    print(f'Saving the model to {output_file} ...')
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    print(f'Finish: Model saved to {output_file} ...')


## Main script
df, categoricals, numericals = load_data()

# calculate the standard deviation of the AUC scores
k_fold_auc(df, categoricals, numericals)

pipeline = train_model(df)
save_model(pipeline, 'model.bin')