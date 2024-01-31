from datetime import datetime
from model_setup import ModelSetup
from model_trainer import ModelTrainer

import mlflow
from mlflow.models import infer_signature

import os
from mlflow.artifacts import download_artifacts
from peft import PeftModel

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch

class ModelPostprocessing:
    def __init__(self, base_model_catalog_name: str, base_model_name: str, base_model_version: int, best_model_path: str, dbfs_tuned_model_output_dir: str = None, base_model = None, tokenizer = None):
        self.base_model_catalog_name = base_model_catalog_name
        self.base_model_name = base_model_name
        self.base_model_version = base_model_version
        self.best_model_path = best_model_path
        self.base_model = base_model
        self.tokenizer = tokenizer

        if dbfs_tuned_model_output_dir is None:
            now = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.dbfs_tuned_model_output_dir = f"/dbfs/tuned_models/{self.base_model_name}/{now}/"
        else:
            self.dbfs_tuned_model_output_dir = dbfs_tuned_model_output_dir
            

    def upload_tuned_adapter_to_dbfs(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = False,
            bnb_4bit_quant_type = 'nf4',
            bnb_4bit_compute_dtype = torch.bfloat16,
        )

        if self.base_model is None:
            mlflow.set_registry_uri('databricks-uc')

            model_mlflow_path = f"models:/{self.base_model_catalog_name}.models.{self.base_model_name}/{self.base_model_version}"
            model_local_path = f"/{self.base_model_name}/"

            path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)
            local_base_tokenizer_path = os.path.join(path, "components", "tokenizer")
            local_base_model_path = os.path.join(path, "model")

            base_model = AutoModelForCausalLM.from_pretrained(local_base_model_path, quantization_config=quantization_config, load_in_4bit=True)

        else:
            base_model = self.base_model
        
        model_to_merge = PeftModel.from_pretrained(base_model, self.best_model_path)

        # merges in the adapters to the base model
        merged_model = model_to_merge.merge_and_unload()

        # this path has to be dbfs for this to work
        merged_model.save_pretrained(self.dbfs_tuned_model_output_dir)

        # ensuring the base tokenizer is brought in to allow for easier loading later on
        self.tokenizer.save_pretrained(self.dbfs_tuned_model_output_dir)

        return self.dbfs_tuned_model_output_dir