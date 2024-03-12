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
import shutil


class QuickTrain:
    def __init__(self, base_model_catalog_name: str, base_model_name: str, base_model_version: int, data: pd.DataFrame, dbfs_tuned_adapter_dir: str = None, lora_config_dict = {}, training_args_dict = {}):
        self.base_model_catalog_name = base_model_catalog_name
        self.base_model_name = base_model_name
        self.base_model_version = base_model_version
        self.now = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


        if dbfs_tuned_adapter_dir is None:
            self.dbfs_tuned_adapter_dir = f"/dbfs/tuned_adapters/{self.base_model_name}/{self.now}/"
        else:
            self.dbfs_tuned_adapter_dir = dbfs_tuned_adapter_dir
        self.data = data
        self.best_adapter_path = None
        self.lora_config_dict = lora_config_dict
        self.training_args_dict = training_args_dict
        # self.base_model = None

    def get_model_from_catalog(self, catalog_name: str, model_name: str, version: int):
        mlflow.set_registry_uri('databricks-uc')

        model_mlflow_path = f"models:/{catalog_name}.models.{model_name}/{version}"
        model_local_path = f"/{model_name}/"

        path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)

        tokenizer_path = os.path.join(path, "components", "tokenizer")
        model_path = os.path.join(path, "model")

        return model_path, tokenizer_path

    def run_training(self, model_setup: ModelSetup, base_model_path: str):

        # model_setup.mlflow_dir = f'mlflowruns/training{base_model_path}{model_setup.mlflow_experiment_id}'
        # mlflow.set_tracking_uri(model_setup.mlflow_dir)

        # every time I set a custom run_id it would not go through, so turning off for now
        # with mlflow.start_run(base_model_name):
        with mlflow.start_run():
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

            training_model, output_time, best_adapter_path = training_model.train()
            self.best_adapter_path = best_adapter_path

    def upload_adapter_to_dbfs(self, best_adapter_path: str = None, dbfs_tuned_adapter_dir: str = None):
        if best_adapter_path is None:
            best_adapter_path = self.best_adapter_path
        if dbfs_tuned_adapter_dir is None:
            dbfs_tuned_adapter_dir = self.dbfs_tuned_adapter_dir
        
        shutil.copytree(best_adapter_path, dbfs_tuned_adapter_dir)

        print("You can find your tuned adapter in the following directory on the DBFS")
        print(dbfs_tuned_adapter_dir)
        
                
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
        self.tokenizer = model_setup.tokenizer

        self.run_training(model_setup = model_setup, base_model_path = base_model_path)
        self.upload_adapter_to_dbfs(best_adapter_path = self.best_adapter_path, dbfs_tuned_adapter_dir = self.dbfs_tuned_adapter_dir)