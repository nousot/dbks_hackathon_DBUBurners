# Databricks notebook source
!pip install rich peft optimum trl bitsandbytes accelerate mlflow -U transformers

# COMMAND ----------

# this restarts the python kernel so the above installs go through
dbutils.library.restartPython()

# COMMAND ----------

# from tuned_model import TunedModel
import pandas as pd
import json

from typing import List, cast


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
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id
import numpy as np

from utils import input_preprocessing, fix_key_names, format_training_data

from typing import List, cast
import logging

# COMMAND ----------

base_model_catalog = "databricks_dolly_v2_models"
# base_model_name = "dolly_v2_12b"
base_model_name = "dolly_v2_3b"
base_model_version = 1
adapter_dbfs_path = f"/dbfs/tuned_adapters/{base_model_name}/2024-02-09 16:44:23"
gpu_or_cpu = "gpu"

# COMMAND ----------

data = pd.read_csv("osha_dataset.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

# COMMAND ----------

data.columns

# COMMAND ----------

target_schema = {'summary_nr': 'int', 'Event Date': 'str', 'Event Description': 'str', 'Event Keywords': 'str', 'con_end': 'str', 'Construction End Use': 'str', 'build_stor': 'int', 'Building Stories': 'str', 'proj_cost': 'str', 'Project Cost': 'str', 'proj_type': 'str', 'Project Type': 'str', 'Degree of Injury': 'str', 'nature_of_inj': 'int', 'Nature of Injury': 'str', 'part_of_body': 'int', 'Part of Body': 'str', 'event_type': 'int', 'Event type': 'str', 'evn_factor': 'int', 'Environmental Factor': 'str', 'hum_factor': 'int', 'Human Factor': 'str', 'task_assigned': 'int', 'Task Assigned': 'str', 'hazsub': 'str', 'fat_cause': 'int', 'fall_ht': 'int'}

# COMMAND ----------

# tuned_model = TunedModel(base_model_catalog=base_model_catalog, base_model_name=base_model_name, base_model_version=base_model_version, adapter_dbfs_path=adapter_dbfs_path, gpu_or_cpu=gpu_or_cpu)

# COMMAND ----------

default_args = get_default_generation_config()
generation_config = GenerationConfig(
    penalty_alpha=default_args["penalty_alpha"], 
    do_sample = default_args["do_sample"], 
    top_k=default_args["top_k"], 
    temperature=default_args["temperature"], 
    repetition_penalty=default_args["repetition_penalty"],
    max_new_tokens=default_args["max_new_tokens"],
)
target_schema_str = ''


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


quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, #enables 4bit quantization
    bnb_4bit_use_double_quant = False, #repeats quantization a second time if true
    bnb_4bit_quant_type = 'nf4', #`fp4` or `nf4`
    bnb_4bit_compute_dtype = torch.bfloat16, #fp dtype, can be changed for speed up
)

model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config), adapter_dbfs_path)

prod_model = model_to_merge.merge_and_unload()

# saves space
del model_to_merge

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
prod_tokenizer = tokenizer

del tokenizer

# COMMAND ----------

if 'preprocessed_input' not in data.columns:
    target_schema_str = str(target_schema)
    data = format_training_data(data=data, target_schema_str=target_schema_str)
data = data[["input", "output", "preprocessed_input", "text"]]
        
spark = SparkSession.builder.appName("TunedModel").getOrCreate()
spark_data = spark.createDataFrame(data=data, schema=["input", "output", "preprocessed_input", "text"])

# COMMAND ----------

def output_generation(data):
    # ideally we want more but ran into immediate space issues at 16 and much later space issues at 8
    # even on small model
    batch_size = 4


    def chunks(list_val, n):
        """Yield successive n-sized chunks from list_val."""
        logging.info(f"Using batch size of {batch_size}")
        for i in range(0, len(list_val), n):
            yield list_val[i:i + n]

    output_chunks = []
    list_val = data["preprocessed_input"].tolist()
    for chunk in chunks(list_val, batch_size):
        logging.info("Processing new chunk")
        formatted_chunk = cast(List[str], chunk)
        inputs = prod_tokenizer(formatted_chunk, return_tensors='pt', padding=True)
        if gpu_or_cpu == 'gpu':
            inputs.to("cuda")
        raw_outputs = prod_model.generate(**inputs, generation_config=generation_config, pad_token_id=prod_tokenizer.eos_token_id)
        outputs = raw_outputs[0].cpu().detach().numpy()
        output_chunks.append(outputs)

        del inputs
        del outputs
        torch.cuda.empty_cache()

    final_outputs = np.concatenate(output_chunks)

    return_data = (
        data[["input"]]
        .assign(output=list(final_outputs))
    )

    return return_data

# COMMAND ----------

# output_data = output_generation(data)

# COMMAND ----------

result_schema = StructType(
            [
                StructField("input", StringType(), True),
                StructField("generated_output", StringType(), True)
            ]
        )
spark_data = (
            spark_data
            .groupBy(spark_partition_id().alias("_pid"))
            .applyInPandas(output_generation, result_schema)
        )
# spark.udf.register("output_generation", output_generation)
# spark_data = spark_data.withColumn("output", output_generation(col("preprocessed_input")))

# COMMAND ----------

spark_data.select(["input","generated_output"]).show(5)

# COMMAND ----------

final_data = spark_data.toPandas()

# COMMAND ----------

final_data.to_csv("outputs.csv")

# COMMAND ----------

# data = tuned_model.generate(data=data, target_schema=target_schema, gpu_or_cpu=gpu_or_cpu)

# COMMAND ----------

# data.head(5)
