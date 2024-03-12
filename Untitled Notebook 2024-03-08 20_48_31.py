# Databricks notebook source
!pip install rich peft optimum trl bitsandbytes accelerate mlflow -U transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from utils import input_preprocessing, fix_key_names, format_training_data, count_seq_len
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
from peft import prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM, PeftModelForFeatureExtraction, PeftConfig
from accelerate import accelerator

from time import perf_counter
import mlflow
from mlflow.artifacts import download_artifacts
import shutil

from defaults import get_default_generation_config
from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id

import numpy as np
import transformers

# COMMAND ----------


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

        # downloads the best adapter
        # adapter_local_path = os.path.join(path, "adapter")
        # shutil.copytree(adapter_dbfs_path, adapter_local_path)

        # config = PeftConfig.from_pretrained(adapter_local_path)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True, #enables 4bit quantization
            bnb_4bit_use_double_quant = False, #repeats quantization a second time if true
            bnb_4bit_quant_type = 'nf4', #`fp4` or `nf4`
            bnb_4bit_compute_dtype = torch.bfloat16, #fp dtype, can be changed for speed up
        )

        self.prod_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config), adapter_dbfs_path).merge_and_unload()
        
        # self.prod_model = self.prod_model.merge_and_unload()

        #saves space
        # del model_to_merge

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map="auto")
        tokenizer.pad_token = tokenizer.eos_token
        self.prod_tokenizer = tokenizer


    def generate(self, data: pd.DataFrame, target_schema: str, gpu_or_cpu: str = None, generation_config_args: dict = {}):

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

        if 'preprocessed_input' not in data.columns:
            target_schema_str = str(target_schema)
            data = format_training_data(data=data, target_schema_str=target_schema_str)
        
        data = data[["text", "label"]]

        max_length = count_seq_len(data)

        data_collator = transformers.DataCollatorWithPadding(tokenizer=self.prod_tokenizer)
        
        start_time = perf_counter()

        inputs = self.prod_tokenizer(data["text"])
        if gpu_or_cpu == 'gpu': 
            inputs.to("cuda") 
        
        raw_outputs = self.prod_model.generate(inputs, generation_config=generation_config, pad_token_id=self.prod_tokenizer.eos_token_id) 
        outputs = self.prod_tokenizer.batch_decode(raw_outputs, skip_special_tokens=True).cpu().detach().numpy() 

        for index, output in enumerate(outputs): 
            data.at[index,'generated_output'] = output 

        end_time = perf_counter() 
        output_time = end_time - start_time 
        
        print(f"Time taken for inference: {round(output_time,2)} seconds")

        return data

        
        

# COMMAND ----------

base_model_catalog = "databricks_dolly_v2_models"
base_model_name = "dolly_v2_12b"
base_model_version = 1
adapter_dbfs_path = f"/dbfs/tuned_adapters/{base_model_name}/2024-03-05 21:27:44"
gpu_or_cpu = "gpu"

# COMMAND ----------

data = pd.read_csv("osha_dataset.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

# COMMAND ----------

target_schema = {'summary_nr': 'int', 'Event Date': 'str', 'Event Description': 'str', 'Event Keywords': 'str', 'con_end': 'str', 'Construction End Use': 'str', 'build_stor': 'int', 'Building Stories': 'str', 'proj_cost': 'str', 'Project Cost': 'str', 'proj_type': 'str', 'Project Type': 'str', 'Degree of Injury': 'str', 'nature_of_inj': 'int', 'Nature of Injury': 'str', 'part_of_body': 'int', 'Part of Body': 'str', 'event_type': 'int', 'Event type': 'str', 'evn_factor': 'int', 'Environmental Factor': 'str', 'hum_factor': 'int', 'Human Factor': 'str', 'task_assigned': 'int', 'Task Assigned': 'str', 'hazsub': 'str', 'fat_cause': 'int', 'fall_ht': 'int'}

# COMMAND ----------

tuned_model = TunedModel(base_model_catalog=base_model_catalog, base_model_name=base_model_name, base_model_version=base_model_version, adapter_dbfs_path=adapter_dbfs_path, gpu_or_cpu=gpu_or_cpu)

# COMMAND ----------

data = tuned_model.generate(data=data, target_schema=target_schema, gpu_or_cpu=gpu_or_cpu)

# COMMAND ----------

data.head(5)
