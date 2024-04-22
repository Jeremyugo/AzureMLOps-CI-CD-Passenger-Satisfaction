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
import seaborn as sns

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
    parser.add_argument("--model_output", type=str, help="Path of output model")

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
    
    # creating mlflow experiment 
    mlflow.set_experiment("passenger-satisfaction-training")
       
    # starting mlflow run
    with mlflow.start_run(run_name="model-training") as run:
        run_id = run.info.run_id
        
        # logging transformation pipeline
        mlflow.sklearn.log_model(full_pipeline, artifact_path="feature_engineering")

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

        # Visualize results
        conf_matrix_pred = cross_val_predict(model, X_train_scaled, y_train, cv = 3)
        conf_matrix = confusion_matrix(conf_matrix_pred, y_train)
        conf_matrix = conf_matrix / conf_matrix.sum(axis=0)
        
        sns.heatmap(conf_matrix, annot = True, fmt = '.1%', cmap = 'gray_r', xticklabels = ['Predicted Negative','Predicted Positive'],
                yticklabels = ['Actual Negative','Actual Positive'], annot_kws = {'weight':'bold', 'size':12})
        plt.title('Confusion matrix of model predictions', fontsize = 16)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Save the model
        mlflow.sklearn.save_model(model, path="random_forest_model", signature=signature)
        
        
        class CustomPredict(mlflow.pyfunc.PythonModel):
            def __init__(self,):
                self.run_id = run_id
                self.class_names = np.array(["neutral or dissatisfied", "satisfied"])
                self.full_pipeline = full_pipeline
                
            def process_inference_data(self, model_input):
                model_input = self.full_pipeline.transform(model_input)
                return model_input
            
            def process_prediction(self, predictions):
                predictions = predictions.astype(int)
                class_names = self.class_names
                class_predictions = [class_names[pred] for pred in predictions]
                return class_predictions
            
            def load_context(self, context=None):
                self.model = mlflow.sklearn.load_model("random_forest_model") ###################
                return self.model
            
            def predict(self, context, model_input):
                model = self.load_context()
                model_input = self.process_inference_data(model_input)
                predictions = model.predict(model_input)
                return self.process_prediction(predictions)


        # logging the custom model
        mlflow.pyfunc.save_model(path=args.model_output, python_model=CustomPredict(), signature=signature)


if __name__ == "__main__":
    
    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
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
    