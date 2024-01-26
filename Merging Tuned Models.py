# Databricks notebook source
### we'll need to make our own image and throw these in, rest is covered (so far)
# !pip install rich peft trl auto-gptq optimum

# COMMAND ----------

### we'll need to make our own image and throw these in, rest is covered (so far)
!pip install rich peft optimum trl bitsandbytes -U transformers

# COMMAND ----------

# !pip show flash-attn

# COMMAND ----------

# !pip install cuda-python

# COMMAND ----------

# this takes a solid 20 min and does not yield any results
# !pip install packaging ninja einops flash-attn==v1.0.9

# COMMAND ----------

# !pip install transformers==4.33.1 --upgrade

# COMMAND ----------

!pip list

# COMMAND ----------

# !nvidia-smi

# COMMAND ----------

# run these too if you are on a small compute cluster
# !pip install langchain mlflow

# COMMAND ----------

# !export USE_FLASH_ATTENTION=True

# COMMAND ----------

# this restarts the python kernel so the above installs go through
dbutils.library.restartPython()

# COMMAND ----------

import logging
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig, BitsAndBytesConfig
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
import torch

from time import perf_counter
from rich import print

import pprint
import json
from timeit import default_timer as timer

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

from trl import SFTTrainer

from utils import fix_key_names, input_preprocessing

from model_setup import ModelSetup
from trainer_setup import ModelTrainer

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from datetime import datetime

from defaults import get_default_LORA_config, get_default_training_args
from delta.tables import DeltaTable
from pyspark.sql.functions import col

from peft import PeftModel

logger = logging.getLogger(__name__)
global_config = None
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

# model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# COMMAND ----------

data = pd.read_csv("arxiv_pdfs_data.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# # # This we would throw in the training, but with a function to save only the best checkpoint and then save that
# adapters_name = f"mlflowruns/training/dolly_v2_3b/model2024-01-26 19:16:39/outputs/checkpoint-30"
# model = PeftModel.from_pretrained(model_setup.model, adapters_name)
# model = model.merge_and_unload()
# model.save_pretrained("tuned_models/model3")

# COMMAND ----------

# you can see this is spiking both temp and space after model load
!nvidia-smi

# COMMAND ----------

tuned_model_path = "tuned_models/checkpoint-30"

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(tuned_model_path)

# COMMAND ----------

input_text = data['input'][0]
inputs = tokenizer(input_text, return_tensors='pt')
inputs.to("cuda")

# COMMAND ----------

if torch.cuda.is_available():
    device = torch.cuda.get_device_name()
    print(f"Using device: {device}")
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Memory allocated on GPU: {allocated_memory/1024} GB")
else:
    print("No GPU available. Using CPU.")

# COMMAND ----------

tuned_model = AutoModelForCausalLM.from_pretrained(tuned_model_path, torch_dtype=torch.float16)

# COMMAND ----------


start_time = perf_counter()

generation_config = GenerationConfig(
    penalty_alpha=0.6, 
    do_sample = True, 
    top_k=5, 
    temperature=0.1, 
    repetition_penalty=1.2,
    max_new_tokens=2048
)
outputs = tuned_model.generate(**inputs, generation_config=generation_config)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
end_time = perf_counter()
output_time = end_time - start_time
# with open('tuned_output.txt', 'w') as file:
#     file.write(output_text)
print(f"Time taken for original inference: {round(output_time,2)} seconds")

# COMMAND ----------

output_text
