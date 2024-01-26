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

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig
# from datasets import Dataset
# from sklearn.model_selection import train_test_split


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

import os
import mlflow
from mlflow.artifacts import download_artifacts
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

model_setup = ModelSetup(model_name=model_path, tokenizer_path=tokenizer_path, raw_data=data)
# model_setup.create_training_delta_table() # This works, just gets permissioning error
model_setup.quickstart() # This compiles all the individual functions that users can also call if they would prefer

# COMMAND ----------

model_setup.cleaned_data

# COMMAND ----------

# # Define the model class
# class training_model(mlflow.pyfunc.PythonModel):

#     def __init__(self, model, tokenizer, signature, train_dataset, eval_dataset, mlflow_dir, lora_config_dict={}, training_args_dict={}):
#         self.lora_config_dict = lora_config_dict
#         self.training_args_dict = training_args_dict
#         self.model = model
        
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "right"
#         self.tokenizer_specs = {
#             "pad_token": tokenizer.pad_token,
#             "eos_token": tokenizer.eos_token,
#             "padding_side": tokenizer.padding_side
#         }
#         self.tokenizer = tokenizer
        
#         self.signature = signature
#         self.train_dataset = train_dataset
#         self.eval_dataset = eval_dataset
#         self.mlflow_dir = mlflow_dir
#         self.augment_with_defaults()


#     def augment_with_defaults(self):
#         for key, value in get_default_LORA_config().items():
#             if key not in self.lora_config_dict.keys():
#                 self.lora_config_dict[key] = value

#         for key, value in get_default_training_args().items():
#             if key not in self.training_args_dict.keys():
#                 self.training_args_dict[key] = value
#         if 'output_dir' not in self.training_args_dict.keys():
#             self.training_args_dict['output_dir'] = "/".join([self.mlflow_dir, "outputs"])
                    


#     def predict(self):
#         config = LoraConfig(
#         r=self.lora_config_dict["r"],
#         lora_alpha=self.lora_config_dict["lora_alpha"],
#         target_modules=self.lora_config_dict["target_modules"],
#         lora_dropout=self.lora_config_dict["lora_dropout"],
#         bias=self.lora_config_dict["bias"],
#         task_type=self.lora_config_dict["task_type"]
#         )

#         model = get_peft_model(self.model, config)

#         ### more params for us to configure and play with
#         # https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/arguments.py
#         args=TrainingArguments(
#                 per_device_train_batch_size=self.training_args_dict['per_device_train_batch_size'],
#                 gradient_accumulation_steps=self.training_args_dict['gradient_accumulation_steps'],
#                 warmup_steps=self.training_args_dict['warmup_steps'],
#                 max_steps=self.training_args_dict['max_steps'],
#                 learning_rate=self.training_args_dict['learning_rate'],
#                 fp16=self.training_args_dict['fp16'],
#                 logging_steps=self.training_args_dict['logging_steps'],
#                 output_dir=self.training_args_dict['output_dir'],
#                 optim=self.training_args_dict['optim'],
#                 save_strategy=self.training_args_dict['save_strategy']
#         )

#         sft_trainer_args = {
#             "model": model,
#             "args": args,
#             "train_dataset": self.train_dataset,
#             "eval_dataset": self.eval_dataset,
#             "peft_config": config,
#             "dataset_text_field": "text", # field to tune on
#             "tokenizer": self.tokenizer,
#             "packing": False,
#             "max_seq_length": 4096
#         }

#         # link to SFTTrainer args (https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)
#         trainer = SFTTrainer(
#             model=sft_trainer_args["model"],
#             args=sft_trainer_args["args"],
#             train_dataset=sft_trainer_args["train_dataset"],
#             eval_dataset=sft_trainer_args["eval_dataset"],
#             peft_config=sft_trainer_args["peft_config"],
#             dataset_text_field=sft_trainer_args["dataset_text_field"],
#             tokenizer=sft_trainer_args["tokenizer"],
#             packing=sft_trainer_args["packing"],
#             max_seq_length=sft_trainer_args["max_seq_length"])


#         start_time = perf_counter()
#         trainer.train()
#         end_time = perf_counter()
#         output_time = end_time - start_time
#         output_time = output_time / 60
#         output_time = " ".join([str(round(output_time,2)), "minutes"])


#         return sft_trainer_args, output_time

#         ### this will automatically save a checkpoint in the specified output_dir
#         ### just like how our classic TFT had

# COMMAND ----------

training_args_dict = {
        # "per_device_train_batch_size": 1,
        # "gradient_accumulation_steps": 4,
        # "warmup_steps": 0,
        "max_steps": 5,
        # "learning_rate": 2e-5,
        # "fp16": True, # use mixed precision training
        # "logging_steps": 1,
        # "optim": "paged_adamw_8bit", #adamw kept coming up deprecated
        # "save_strategy": "epoch"
        "ddp_find_unused_parameters": False # defaults to true, not sure what the performance difference is, but can only get it working for us with false across any model
    }

# COMMAND ----------

# you can see this is spiking both temp and space after model load
!nvidia-smi

# COMMAND ----------


# model_setup.mlflow_experiment_id = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
model_setup.mlflow_dir = f'mlflowruns/training{model_path}{model_setup.mlflow_experiment_id}'
mlflow.set_tracking_uri(model_setup.mlflow_dir)

with mlflow.start_run() as run:
    training_model = ModelTrainer(
        model=model_setup.model,
        tokenizer=model_setup.tokenizer,
        signature=model_setup.signature,
        train_dataset=model_setup.train_dataset,
        eval_dataset=model_setup.eval_dataset,
        mlflow_dir=model_setup.mlflow_dir,
        # lora_config_dict=lora_config_dict,
        training_args_dict=training_args_dict,
    )
    
    mlflow.log_param("lora_config", training_model.lora_config_dict)
    mlflow.log_param("tokenizer_specs", training_model.tokenizer_specs)
    mlflow.log_param("training_args_dict", training_model.training_args_dict)

    sft_trainer_args, output_time = training_model.predict()
    
    # mlflow.log_param("sft_trainer_args", sft_trainer_args)
    mlflow.log_param("output_time", output_time)


# COMMAND ----------

from peft import PeftModel
    
adapters_name = f"mlflowruns/training/{model_name}/model{model_setup.mlflow_experiment_id}/outputs/checkpoint-30/"
model_to_merge = model_setup.model

# COMMAND ----------

model = PeftModel.from_pretrained(model_to_merge, adapters_name)

model = model.merge_and_unload()
model.save_pretrained("tuned_models/testing_safe_serial_model")

# COMMAND ----------

import torch
model = AutoModelForCausalLM.from_pretrained("tuned_models/testing_safe_serial_model")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# COMMAND ----------

inputs = tokenizer("A PAPER ABOUT TESTING: why tests matter. Written on July 10th", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
outputs

# COMMAND ----------

def testModel(input_data):
    model = PeftModel.from_pretrained()

# COMMAND ----------

mlflow.pyfunc.log_model(
        artifact_path="merged_adapters/",
        signature=model_setup.signature,
        python_model=model,
        registered_model_name=f"{model_name}_tuned_{model_setup.mlflow_experiment_id}"
    )

# COMMAND ----------

# model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_path).to("cuda"), adapters_name)

# COMMAND ----------

merged_model = model_to_merge.merge_and_unload()

# COMMAND ----------

# mlflow.pyfunc.save_model(
#     signature=model_setup.signature,
#     path=f"/dbfs/DBUBurners_hackathon/tuned_models/{model_name}/{model_setup.mlflow_experiment_id}",
#     python_model=training_model
# )

# COMMAND ----------

import inspect
inspect.signature(merged_model).parameters

# COMMAND ----------

mlflow.pyfunc.log_model(
        artifact_path="mlflowruns/",
        # signature=model_setup.signature,
        python_model=merged_model,
        registered_model_name=f"{model_name}_tuned_{model_setup.mlflow_experiment_id}"
    )
