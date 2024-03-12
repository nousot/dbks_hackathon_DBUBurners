import pandas as pd
from utils import input_preprocessing, fix_key_names, format_training_data
import json
from delta.tables import DeltaTable
import re
import os
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from datasets import Dataset
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

import torch
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, PeftModel
from accelerate import accelerator

from time import perf_counter
import mlflow
from mlflow.artifacts import download_artifacts
import shutil

from defaults import get_default_generation_config




class TunedModel:
    def __init__(self, base_model_catalog: str, base_model_name: str, base_model_version: str, adapter_dbfs_path: str, gpu_or_cpu: str):
        self.base_model_catalog = base_model_catalog
        self.base_model_name = base_model_name
        self.base_model_version = base_model_version
        self.adapter_dbfs_path = adapter_dbfs_path
        self.gpu_or_cpu = gpu_or_cpu

        # load the files necessary and model to local immediately
        mlflow.set_registry_uri('databricks-uc')
        
        model_mlflow_path = f"models:/{base_model_catalog}.models.{base_model_name}/{base_model_version}"
        model_local_path = f"/{base_model_name}/"
        path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)
        tokenizer_path = os.path.join(path, "components", "tokenizer")
        model_path = os.path.join(path, "model")

        # # downloads the best adapter
        # adapter_local_path = os.path.join(path, "adapter")
        # shutil.copytree(adapter_dbfs_path, adapter_local_path)

        SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
        dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        if dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            print("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True, #enables 4bit quantization
            bnb_4bit_use_double_quant = False, #repeats quantization a second time if true
            bnb_4bit_quant_type = 'nf4', #`fp4` or `nf4`
            bnb_4bit_compute_dtype = dtype, #fp dtype, can be changed for speed up
        )

        # straight out of unsloth
        quantization_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = dtype,
        )

        model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto", torch_dtype=dtype, trust_remote_code=True), adapter_dbfs_path)
        
        self.prod_model = model_to_merge.merge_and_unload()

        # saves space
        del model_to_merge

        # self.prod_model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto", torch_dtype=dtype, trust_remote_code=True)

        # self.prod_model = PeftModel.from_pretrained(self.prod_model, adapter_dbfs_path)
        
        # self.prod_model = self.prod_model.merge_and_unload()


        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map="auto")
        tokenizer.pad_token = tokenizer.eos_token
        self.prod_tokenizer = tokenizer


    def generate(self, data: pd.DataFrame, target_schema: str, gpu_or_cpu: str = None, generation_config_args: dict = {}):
        if 'preprocessed_input' not in data.columns:
            target_schema_str = str(target_schema)
            data = format_training_data(data=data, target_schema_str=target_schema_str, model_name=self.base_model_name)

        for key, value in get_default_generation_config().items():
            if key not in generation_config_args.keys():
                generation_config_args[key] = value
    
        generation_config = GenerationConfig(
            penalty_alpha=generation_config_args["penalty_alpha"], 
            do_sample = generation_config_args["do_sample"], 
            top_k=generation_config_args["top_k"], 
            temperature=generation_config_args["temperature"], 
            repetition_penalty=generation_config_args["repetition_penalty"],
            max_new_tokens=generation_config_args["max_new_tokens"],
        )

        start_time = perf_counter()
        for index, row in data.iterrows():
            input_text = row['preprocessed_input']
            inputs = self.prod_tokenizer(input_text, return_tensors='pt')

            if gpu_or_cpu == 'gpu':
                inputs.to("cuda")

            outputs = self.prod_model.generate(**inputs, generation_config=generation_config, pad_token_id=self.prod_tokenizer.eos_token_id)
            output_text = self.prod_tokenizer.decode(outputs[0], skip_special_tokens=True)

            data.at[index,'generated_output'] = output_text

        end_time = perf_counter()
        output_time = end_time - start_time

        print(f"Time taken for inference: {round(output_time,2)} seconds")

        return data

        
        