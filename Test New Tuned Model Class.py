# Databricks notebook source
!pip install rich peft optimum trl bitsandbytes accelerate mlflow -U transformers

# COMMAND ----------

# this restarts the python kernel so the above installs go through
dbutils.library.restartPython()

# COMMAND ----------

from tuned_model import TunedModel
import pandas as pd
import json

# COMMAND ----------

base_model_catalog = "databricks_mistral_models"
base_model_name = "mistral_7b_instruct_v0_2"
base_model_version = 1
adapter_run_time = "2024-03-12 16:37:13"
adapter_dbfs_path = f"/dbfs/tuned_adapters/{base_model_name}/{adapter_run_time}"
gpu_or_cpu = "gpu"

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

tuned_model = TunedModel(base_model_catalog=base_model_catalog, base_model_name=base_model_name, base_model_version=base_model_version, adapter_dbfs_path=adapter_dbfs_path, gpu_or_cpu=gpu_or_cpu)

# COMMAND ----------

data = pd.read_csv("osha_dataset.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

# COMMAND ----------

target_schema = {'summary_nr': 'int', 'Event Date': 'str', 'Event Description': 'str', 'Event Keywords': 'str', 'con_end': 'str', 'Construction End Use': 'str', 'build_stor': 'int', 'Building Stories': 'str', 'proj_cost': 'str', 'Project Cost': 'str', 'proj_type': 'str', 'Project Type': 'str', 'Degree of Injury': 'str', 'nature_of_inj': 'int', 'Nature of Injury': 'str', 'part_of_body': 'int', 'Part of Body': 'str', 'event_type': 'int', 'Event type': 'str', 'evn_factor': 'int', 'Environmental Factor': 'str', 'hum_factor': 'int', 'Human Factor': 'str', 'task_assigned': 'int', 'Task Assigned': 'str', 'hazsub': 'str', 'fat_cause': 'int', 'fall_ht': 'int'}

# COMMAND ----------

data = tuned_model.generate(data=data.head(5), target_schema=target_schema, gpu_or_cpu=gpu_or_cpu)

# COMMAND ----------

print(data.head(1))

# COMMAND ----------

print(data['generated_output'][1])

# COMMAND ----------

from utils import format_training_data
target_schema_str = str(target_schema)
cleaned_data = format_training_data(data=data, target_schema_str=target_schema_str, model_name=base_model_name)
cleaned_data['model_ready'][0]

# COMMAND ----------

print(cleaned_data['model_ready'][0])

# COMMAND ----------

print(cleaned_data['preprocessed_input'][0])

# COMMAND ----------

# data.to_csv("data_with_generations.csv")
