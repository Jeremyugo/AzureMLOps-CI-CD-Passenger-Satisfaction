# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

TARGET_COL = "satisfaction"


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--base_output", type=str, help="Path of base model")
    parser.add_argument("--scaler", type=str, help="Path of feature engineering pipeline")

    # classifier specific arguments
    parser.add_argument('--classifier__n_estimators', type=int, default=100,
                        help='Number of trees')
    parser.add_argument('--classifier__bootstrap', type=int, default=True,
                        help='Method of selecting samples for training each tree')
    parser.add_argument('--classifier__max_depth', type=int, default=10,
                        help=' Maximum number of levels in tree')
    parser.add_argument('--classifier__max_features', type=str, default='sqrt',
                        help='Number of features to consider at every split')
    parser.add_argument('--classifier__min_samples_leaf', type=int, default=1,
                        help='Minimum number of samples required at each leaf node')
    parser.add_argument('--classifier__min_samples_split', type=int, default=2,
                        help='Minimum number of samples required to split a node')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data.drop(TARGET_COL, axis=1)

    # remapping target variables -> "{'satisfied': 1, 'neutral or dissatisfied': 0}"
    y_train = np.where(y_train == 'satisfied', 1, 0)
    
    # identifying numerical and categorical variables
    num_attribs = X_train.select_dtypes(exclude=object).columns
    cat_attribs = X_train.select_dtypes(include=object).columns


    # Transformer Class to handle dataframe selection
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attrib_names):
            self.attrib_names = attrib_names
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return X[self.attrib_names]


    # Transformer Class to handle categorical data encoding
    class Encode(BaseEstimator, TransformerMixin):
        def __init__(self, attrib_names):
            self.attrib_names = attrib_names
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X = X[self.attrib_names].copy()
            X['Customer Type'] = np.where(X['Customer Type'] == 'Loyal Customer', 1, 0)
            X['Type of Travel'] = np.where(X['Type of Travel'] == 'Business travel', 1, 0)
            X = pd.concat([X, X['Class'].str.get_dummies()], axis=1).drop('Class', axis=1)
            return X


    # creating a numerical processing pipeline
    num_pipeline = Pipeline([
        ('Select dataframe', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
    ])

    # creating a categorical processing pipeline 
    cat_pipeline = Pipeline([
        ('cat_encoder', Encode(cat_attribs))
    ])

    # full transformation pipeline
    full_pipeline = FeatureUnion(transformer_list = [
        ('numerical', num_pipeline),
        ('categorical', cat_pipeline)
    ])

    # applying transformation pieline
    X_train_scaled = full_pipeline.fit_transform(X_train)
    
        
    # logging transformation pipeline
    mlflow.sklearn.save_model(full_pipeline, path=args.scaler)

    # infering data signature
    signature = infer_signature(X_train, y_train)
    
    # Train a Random Forest Classification Model with the training set
    model = RandomForestClassifier(n_estimators = args.classifier__n_estimators,
                                bootstrap = args.classifier__bootstrap,
                                max_depth = args.classifier__max_depth,
                                max_features = args.classifier__max_features,
                                min_samples_leaf = args.classifier__min_samples_leaf,
                                min_samples_split = args.classifier__min_samples_split,
                                random_state=11)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", args.classifier__n_estimators)
    mlflow.log_param("bootstrap", args.classifier__bootstrap)
    mlflow.log_param("max_depth", args.classifier__max_depth)
    mlflow.log_param("max_features", args.classifier__max_features)
    mlflow.log_param("min_samples_leaf", args.classifier__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.classifier__min_samples_split)

    # Train model with the train set
    model.fit(X_train_scaled, y_train)

    # Predict using the Classification Model
    pred = model.predict(X_train_scaled)

    # Evaluate Classification performance with the train set
    f1score = f1_score(y_train, pred)
    pr_score = precision_score(y_train, pred)
    re_score = recall_score(y_train, pred)
    roc_score = roc_auc_score(y_train, pred)
    
    # log model performance metrics
    mlflow.log_metrics({
        "f1_score": f1score,
        "precision_score": pr_score,
        "recall_score": re_score,
        "roc_auc_score": roc_score
    })

    
    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.base_output, signature=signature)
    
    
if __name__ == "__main__":

    mlflow.start_run()
    
    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Base model path: {args.base_output}",
        f"Feature engineering path: {args.scaler}",
        f"n_estimators: {args.classifier__n_estimators}",
        f"bootstrap: {args.classifier__bootstrap}",
        f"max_depth: {args.classifier__max_depth}",
        f"max_features: {args.classifier__max_features}",
        f"min_samples_leaf: {args.classifier__min_samples_leaf}",
        f"min_samples_split: {args.classifier__min_samples_split}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
    
