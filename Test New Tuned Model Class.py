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

base_model_catalog = "databricks_dolly_v2_models"
base_model_name = "dolly_v2_3b"
base_model_version = 1
adapter_dbfs_path = "/dbfs/tuned_adapters/dolly_v2_3b/2024-01-31 22:10:06"
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



# COMMAND ----------

data = tuned_model.generate(data=data, target_schema=target_schema, gpu_or_cpu=gpu_or_cpu)

# COMMAND ----------

data.head(5)
