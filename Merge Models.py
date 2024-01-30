# Databricks notebook source
### we'll need to make our own image and throw these in, rest is covered (so far)
!pip install rich peft optimum trl bitsandbytes accelerate -U transformers

# COMMAND ----------

# this restarts the python kernel so the above installs go through
dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
from mlflow.artifacts import download_artifacts
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
catalog_name = "databricks_dolly_v2_models" # Default catalog name when installing the model from Databricks Marketplace
model_name = "dolly_v2_3b"
version = 1
model_mlflow_path = f"models:/{catalog_name}.models.{model_name}/{version}"
model_local_path = f"/{model_name}/"
path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)
tokenizer_path = os.path.join(path, "components", "tokenizer")
model_path = os.path.join(path, "model")

## for restart post download of model
# import os
# model_name = "dolly_v2_3b"
# model_local_path = f"/{model_name}/"
# path=model_local_path
# tokenizer_path = os.path.join(path, "components", "tokenizer")
# model_path = os.path.join(path, "model")

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

best_model_path = "mlflowruns/training/dolly_v2_3b/model2024-01-29 16:41:23/outputs/tuned_model"

# COMMAND ----------

from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, #enables 4bit quantization
    bnb_4bit_use_double_quant = False, #repeats quantization a second time if true
    bnb_4bit_quant_type = 'nf4', #`fp4` or `nf4`
    bnb_4bit_compute_dtype = torch.bfloat16, #fp dtype, can be changed for speed up
)

model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, load_in_4bit=True), "mlflowruns/training/dolly_v2_3b/model2024-01-29 16:41:23/outputs/tuned_model")
merged_model = model_to_merge.merge_and_unload()

# this path has to be dbfs for this to work
merged_model.save_pretrained(f"/dbfs/DBUBurners_hackathon/tuned_models_jan_29_part_2/{model_name}")


# COMMAND ----------

# save tokenizer in same spot to make life easier
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.save_pretrained(f"/dbfs/DBUBurners_hackathon/tuned_models_jan_29_part_2/{model_name}")
