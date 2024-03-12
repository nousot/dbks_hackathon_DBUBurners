# Databricks notebook source
### we'll need to make our own image and throw these in, rest is covered (so far)
!pip install rich peft optimum trl bitsandbytes accelerate -U transformers

# COMMAND ----------

# this restarts the python kernel so the above installs go through
dbutils.library.restartPython()

# COMMAND ----------

# import logging
import pandas as pd

# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig
# # from datasets import Dataset
# # from sklearn.model_selection import train_test_split


# from time import perf_counter
# from rich import print

# import pprint
# import json
# from timeit import default_timer as timer

# from langchain.llms import LlamaCpp
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

# from trl import SFTTrainer

# from utils import fix_key_names, input_preprocessing

from model_setup import ModelSetup
from model_trainer import ModelTrainer
from quick_train import QuickTrain


# import mlflow
# import mlflow.sklearn
# from mlflow.models import infer_signature

# from datetime import datetime

# from model_postprocessing import ModelPostprocessing

# from defaults import get_default_LORA_config, get_default_training_args
# from delta.tables import DeltaTable
# from pyspark.sql.functions import col

# logger = logging.getLogger(__name__)
# global_config = None
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

data = pd.read_csv("osha_dataset.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

# COMMAND ----------

# how to manipulate training runs
# still need that dynamic sequence length detector

training_args_dict = {
        "per_device_train_batch_size": 1,
        # "gradient_accumulation_steps": 4,
        # "warmup_steps": 0,
        "max_steps": 50, # default is 300, 5 for just testing pipeline throughout
        "num_train_epochs": 10,
        "learning_rate": 4e-4, #4e-4 works well for mistral
        # "fp16": True,
        "logging_steps": 10,
        # "optim": "paged_adamw_8bit",
        # "save_strategy": "epoch",
        # "ddp_find_unused_parameters": False # this cannot be true for T4, crashes otherwise
        "push_to_hub": False
    }
lora_config_dict = {
        # "r": 16,
        # "lora_alpha": 32,
        # "lora_dropout": 0.05,
        # "bias": "none",
        "task_type": "CAUSAL_LM",
    }

# COMMAND ----------

# catalog_name = "databricks_llama_2_models"
# model_name = "llama_2_7b_hf"
# catalog_name = "databricks_dolly_v2_models"
# model_name = "dolly_v2_12b"
catalog_name = "databricks_mistral_models"
model_name = "mistral_7b_instruct_v0_2"

version = 1
quick_train = QuickTrain(
    base_model_catalog_name = catalog_name, 
    base_model_name = model_name,
    base_model_version = version,
    data = data,
    training_args_dict = training_args_dict, # optional, empty dict is also default, missing keys are filled
    lora_config_dict = lora_config_dict # optional, empty dict is also default, missing keys are filled
)
quick_train.quick_train_model()

# COMMAND ----------


