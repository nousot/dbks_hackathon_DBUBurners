from datetime import datetime
from model_setup import ModelSetup
from model_trainer import ModelTrainer
from model_postprocessing import ModelPostprocessing

import mlflow
from mlflow.models import infer_signature

import pandas as pd
import os
from mlflow.artifacts import download_artifacts
from peft import PeftModel

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch

class QuickTrain:
    def __init__(self, base_model_catalog_name: str, base_model_name: str, base_model_version: int, data: pd.DataFrame, dbfs_tuned_model_output_dir: str = None, lora_config_dict = {}, training_args_dict = {}):
        self.base_model_catalog_name = base_model_catalog_name
        self.base_model_name = base_model_name
        self.base_model_version = base_model_version
        if dbfs_tuned_model_output_dir is None:
            now = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.dbfs_tuned_model_output_dir = f"/dbfs/tuned_models/{self.base_model_name}/{now}/"
        else:
            self.dbfs_tuned_model_output_dir = dbfs_tuned_model_output_dir
        self.data = data
        self.best_model_path = None
        self.lora_config_dict = lora_config_dict
        self.training_args_dict = training_args_dict
        self.base_model = None



    def get_model_from_catalog(self, catalog_name: str, model_name: str, version: int):
        mlflow.set_registry_uri('databricks-uc')

        model_mlflow_path = f"models:/{catalog_name}.models.{model_name}/{version}"
        model_local_path = f"/{model_name}/"

        path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)

        tokenizer_path = os.path.join(path, "components", "tokenizer")
        model_path = os.path.join(path, "model")

        return model_path, tokenizer_path

    def run_training(self, model_setup: ModelSetup, base_model_path: str):

        model_setup.mlflow_dir = f'mlflowruns/training{base_model_path}{model_setup.mlflow_experiment_id}'
        mlflow.set_tracking_uri(model_setup.mlflow_dir)
        
        with mlflow.start_run() as run:
            training_model = ModelTrainer(
                model=model_setup.model,
                tokenizer=model_setup.tokenizer,
                signature=model_setup.signature,
                train_dataset=model_setup.train_dataset,
                eval_dataset=model_setup.eval_dataset,
                mlflow_dir=model_setup.mlflow_dir,
                lora_config_dict=self.lora_config_dict,
                training_args_dict=self.training_args_dict,
            )
            
            mlflow.log_param("lora_config", training_model.lora_config_dict)
            mlflow.log_param("tokenizer_specs", training_model.tokenizer_specs)
            mlflow.log_param("training_args_dict", training_model.training_args_dict)

            training_model, output_time, best_model_path = training_model.train()
            self.best_model_path = best_model_path

            mlflow.log_param("output_time", output_time)
                
    def quick_train_model(self):
        base_model_path, base_tokenizer_path = self.get_model_from_catalog(
            catalog_name = self.base_model_catalog_name,
            model_name = self.base_model_name,
            version = self.base_model_version
        )

        model_setup = ModelSetup(
            model_name=base_model_path,
            tokenizer_path=base_tokenizer_path,
            raw_data=self.data
        )
        model_setup.quickstart()
        self.base_model = model_setup.base_model
        self.tokenizer = model_setup.tokenizer

        self.run_training(model_setup = model_setup, base_model_path = base_model_path)