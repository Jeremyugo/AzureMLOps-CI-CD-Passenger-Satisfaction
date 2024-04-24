# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers trained ML model if deploy flag is True.
"""

import argparse
from pathlib import Path
import pickle
import numpy as np
import mlflow

import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--scaler', type=str, help='Scaler directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args


def main(args):
    '''Loads model, registers it if deply flag is True'''

    with open((Path(args.evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())
        
    mlflow.log_metric("deploy flag", int(deploy_flag))
    deploy_flag=1
    if deploy_flag==1:

        print("Registering ", args.model_name)

        # load model
        model =  mlflow.sklearn.load_model(args.model_path) 
        
        # load scaler
        scaler = mlflow.sklearn.load_model(args.scaler)

        # log model using mlflow
        mlflow.sklearn.log_model(model, "base_model")
        mlflow.sklearn.log_model(scaler, "scaler")

        # register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        base_uri = f'runs:/{run_id}/base_model'
        scaler_uri = f'runs:/{run_id}/scaler'
        model_uri = f'runs:/{run_id}/{args.model_name}'
        
        # custom model
        class CustomPredict(mlflow.pyfunc.PythonModel):
            def __init__(self,):
                self.class_names = np.array(["neutral or dissatisfied", "satisfied"])
                self.full_pipeline = mlflow.sklearn.load_model(scaler_uri)
                
            def process_inference_data(self, model_input):
                model_input = self.full_pipeline.transform(model_input)
                return model_input
            
            def process_prediction(self, predictions):
                predictions = predictions.astype(int)
                class_names = self.class_names
                class_predictions = [class_names[pred] for pred in predictions]
                return class_predictions
            
            def load_context(self, context=None):
                self.model = mlflow.sklearn.load_model(base_uri)
                return self.model
            
            def predict(self, context, model_input):
                model = self.load_context()
                model_input = self.process_inference_data(model_input)
                predictions = model.predict(model_input)
                return self.process_prediction(predictions)


        # saving the custom model
        mlflow.pyfunc.log_model("custom_model", python_model=CustomPredict())
        
        # registering models
        mlflow_model = mlflow.register_model(scaler_uri, "scaler")
        mlflow_model = mlflow.register_model(base_uri, "base_model")
        mlflow_model = mlflow.register_model(model_uri, args.model_name)
        model_version = mlflow_model.version

        # write model info
        print("Writing JSON")
        dict = {"id": "{0}:{1}".format(args.model_name, model_version)}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(dict, fp=of)

    else:
        print("Model will not be registered!")

if __name__ == "__main__":

    mlflow.start_run()
    
    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Scaler path: {args.scaler}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
