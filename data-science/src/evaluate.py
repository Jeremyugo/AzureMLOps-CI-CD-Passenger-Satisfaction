# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results and deploy flag.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

TARGET_COL = "satisfaction"


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--base_input", type=str, help="Path of base model")
    parser.add_argument("--scaler", type=str, help="Path of feature engineering pipeline")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")
    parser.add_argument("--runner", type=str, help="Local or Cloud Runner", default="CloudRunner")

    args = parser.parse_args()

    return args

def main(args):
    '''Read trained model and test dataset, evaluate model and save result'''

    # Load the test data
    test_data = pd.read_parquet(Path(args.test_data))

    # Split the data into inputs and outputs
    y_test = test_data[TARGET_COL]
    X_test = test_data.drop(TARGET_COL, axis=1)
    
    # remapping target variables -> "{'satisfied': 1, 'neutral or dissatisfied': 0}"
    y_test = np.where(y_test == 'satisfied', 1, 0)
    
    # Load transformation pipeline
    scaler = mlflow.sklearn.load_model(args.scaler)
    
    # Load the model from input port
    model =  mlflow.sklearn.load_model(args.base_input) 

    # ---------------- Model Evaluation ---------------- #
    yhat_test, score = model_evaluation(X_test, y_test, model, scaler, args.evaluation_output)

    # ----------------- Model Promotion ---------------- #
    if args.runner == "CloudRunner":
        predictions, deploy_flag = model_promotion(args.model_name, args.evaluation_output, X_test, y_test, yhat_test, score)



def model_evaluation(X_test, y_test, model, scaler, evaluation_output):

    # create a copy of test data
    output_data = X_test.copy()
    
    # transforming the test data
    X_test = scaler.transform(X_test)
    
    # Get predictions to y_test (y_test)
    yhat_test = model.predict(X_test)

    # Save the output data with feature columns, predicted cost, and actual cost in csv file
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv((Path(evaluation_output) / "predictions.csv"))

    # Evaluate Model performance with the test set
    f1score = f1_score(y_test, yhat_test)
    pr_score = precision_score(y_test, yhat_test)
    re_score = recall_score(y_test, yhat_test)
    roc_score = roc_auc_score(y_test, yhat_test)

    # Print score report to a text file
    (Path(evaluation_output) / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}"
    )
    with open((Path(evaluation_output) / "score.txt"), "a") as outfile:
        outfile.write("F1 score: {f1score.2f} \n")
        outfile.write("Precision score: {pr_score.2f} \n")
        outfile.write("Recall score: {re_score.2f} \n")
        outfile.write("Area under the curve score: {roc_score.2f} \n")

    mlflow.log_metrics({
            "f1_score": f1score,
            "precision_score": pr_score,
            "recall_score": re_score,
            "roc_auc_score": roc_score
        })


    return yhat_test, f1score

def model_promotion(model_name, evaluation_output, X_test, y_test, yhat_test, score):
    
    scores = {}
    predictions = {}

    client = MlflowClient()

    for model_run in client.search_model_versions(f"name='{model_name}'"):
        model_version = model_run.version
        mdl = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}")
        predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)
        scores[f"{model_name}:{model_version}"] = f1_score(
            y_test, predictions[f"{model_name}:{model_version}"])

    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {deploy_flag}")

    with open((Path(evaluation_output) / "deploy_flag"), 'w') as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # add current model score and predictions
    scores["current model"] = score
    predictions["currrent model"] = yhat_test

    perf_comparison_plot = pd.DataFrame(
        scores, index=["f1 score"]).plot(kind='bar', figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(Path(evaluation_output) / "perf_comparison.png")

    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")

    return predictions, deploy_flag

if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Base path: {args.base_input}",
        f"Feature engineering path: {args.scaler}",
        f"Test data path: {args.test_data}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
