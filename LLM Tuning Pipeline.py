# Databricks notebook source
### we'll need to make our own image and throw these in, rest is covered (so far)
!pip install rich peft trl auto-gptq optimum

# COMMAND ----------

!pip uninstall -y transformers
!pip install git+https://github.com/huggingface/transformers

# COMMAND ----------

# this restarts the python kernel so the above installs go through
dbutils.library.restartPython()

# COMMAND ----------

import logging
import logging
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split


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

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from datetime import datetime

from defaults import get_default_LORA_config, get_default_training_args


logger = logging.getLogger(__name__)
global_config = None
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

### config vars
model_name = "TheBloke/stable-code-3b-GPTQ"
model_path = "/".join(['models',model_name])
# raw_data_path = "PATH_TO_TEST_DATA"
# target_schema_path = we should dynamically generate this
mlflow_experiment_id = str(current_datetime)
mlflow_dir = f'mflowruns/training/{model_name}/{mlflow_experiment_id}'
mlflow.set_tracking_uri(mlflow_dir)

# COMMAND ----------

data = [
    {
        "input": "DUMMY DATA POINT 1",
        "output": {"DUMMY OUTPUT 2": """
                   {
    "KEY1": "7",
    "KEY2": "bye"
}
                   """}    },
    {
        "input": "DUMMY DATA POINT 2",
        "output": {"DUMMY OUTPUT 2": """
                   {
    "KEY1": "70000",
    "KEY2": "hello"
}
                   """}
    },
]

target_schema_str = """
{
    "KEY1": "int",
    "KEY2": "str"
}
"""

example_mapping = {
        "cheers": "cheers_again",
        "POINT": "testing"
    }


# COMMAND ----------

## function for cleaning and prepping our tuning data

## SKIPPING FOR NOW BECAUSE WE DO NOT HAVE DATA BUT WILL IMPLEMENT WHEN WE DO
# Load the relevant JSON files
# with open(target_schema_path, 'r') as file:
#     target_schema_str = file.read()
# with open(raw_data_path, 'r') as file:
#     data = json.load(file)

# process the data
for row in data:
    row['output'] = fix_key_names(dict = row['output'], mappings = example_mapping, direction = 'json_to_schema')
    row['output'] = str(row['output'])
    row = input_preprocessing(row, model_name, target_schema_str)
        
    # can probably pull this out into utils soon
    row['text'] = f"""
    INPUT:
    {row['preprocessed_input']}
    
    ----------------
    
    OUTPUT:
    {row['output']}
    """

### INSTEAD OF JSON FILE SAVE, LET'S MAKE THIS A TABLE
# Save the cleaned data to a new JSON file
with open('predicted-cleaned.jsonl', 'w') as file:
    json.dump(data, file, indent=4)

# COMMAND ----------

cleaned_data = pd.read_json('predicted-cleaned.jsonl')

# COMMAND ----------

### data setup
### cleaned data comes from above
### will need to adjust the signature call once we get the train test splitting implemented
signature = infer_signature(model_input=cleaned_data)
input_example = cleaned_data.head(5)

### convert cleaned data into format needed later on for LoRa, and split it
train, test = train_test_split(cleaned_data, test_size=0.3, random_state=1738)
dataset = Dataset.from_pandas(cleaned_data)
train_dataset = Dataset.from_pandas(train)
eval_dataset = Dataset.from_pandas(test)


# COMMAND ----------

quantization_config_loading = GPTQConfig(bits=4, disable_exllama=False)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quantization_config_loading,
                              device_map="cuda",)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.use_cache = False
model.config.pretraining_tp = 1

# COMMAND ----------

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# COMMAND ----------

# lora_config_dict = {
#     "r": 16, # attention heads
#     "lora_alpha": 32, # alpha scaling
#     "target_modules": ["k_proj","o_proj","q_proj","v_proj", "down_proj", "gate_proj", "up_proj"], # based on Lora paper, we want all linear layers
#     "lora_dropout": 0.05,
#     "bias": "none",
#     "task_type": "FEATURE_EXTRACTION",
# }

# training_args_dict = {
#     "per_device_train_batch_size": 1,
#     "gradient_accumulation_steps": 4,
#     "warmup_steps": 0, # CHANGE MADE HERE
#     "max_steps": 300, # I bumped this down to 100 for PoC, we'll want this up for actual training
#     "learning_rate": 2e-5, # CHANGE MADE HERE
#     "fp16": True, # use mixed precision training (note from og notebook)
#     "logging_steps": 1,
#     "output_dir": "/".join([mlflow_dir, "outputs"]),
#     "optim": "adamw_hf",
#     "save_strategy": "epoch"
# }

# COMMAND ----------

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# COMMAND ----------

print_trainable_parameters(model)
model

# COMMAND ----------

# Define the model class
class training_model(mlflow.pyfunc.PythonModel):

    def __init__(self, model, tokenizer, signature, train_dataset, eval_dataset, mlflow_dir, lora_config_dict={}, training_args_dict={}):
        self.lora_config_dict = lora_config_dict
        self.training_args_dict = training_args_dict
        self.model = model
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer_specs = {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "padding_side": tokenizer.padding_side
        }
        self.tokenizer = tokenizer
        
        self.signature = signature
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.mlflow_dir = mlflow_dir
        self.augment_with_defaults()


    def augment_with_defaults(self):
        for key, value in get_default_LORA_config().items():
            if key not in self.lora_config_dict.keys():
                self.lora_config_dict[key] = value

        for key, value in get_default_training_args().items():
            if key not in self.training_args_dict.keys():
                self.training_args_dict[key] = value
        if 'output_dir' not in self.training_args_dict.keys():
            self.training_args_dict['output_dir'] = "/".join([self.mlflow_dir, "outputs"])
                    


    def predict(self):
        config = LoraConfig(
        r=self.lora_config_dict["r"],
        lora_alpha=self.lora_config_dict["lora_alpha"],
        target_modules=self.lora_config_dict["target_modules"],
        lora_dropout=self.lora_config_dict["lora_dropout"],
        bias=self.lora_config_dict["bias"],
        task_type=self.lora_config_dict["task_type"]
        )

        model = get_peft_model(self.model, config)

        ### more params for us to configure and play with
        # https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/arguments.py
        args=TrainingArguments(
                per_device_train_batch_size=self.training_args_dict['per_device_train_batch_size'],
                gradient_accumulation_steps=self.training_args_dict['gradient_accumulation_steps'],
                warmup_steps=self.training_args_dict['warmup_steps'],
                max_steps=self.training_args_dict['max_steps'],
                learning_rate=self.training_args_dict['learning_rate'],
                fp16=self.training_args_dict['fp16'],
                logging_steps=self.training_args_dict['logging_steps'],
                output_dir=self.training_args_dict['output_dir'],
                optim=self.training_args_dict['optim'],
                save_strategy=self.training_args_dict['save_strategy']
        )

        sft_trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "peft_config": config,
            "dataset_text_field": "text", # field to tune on
            "tokenizer": self.tokenizer,
            "packing": False,
            "max_seq_length": 4096
        }

        # link to SFTTrainer args (https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)
        trainer = SFTTrainer(
            model=sft_trainer_args["model"],
            args=sft_trainer_args["args"],
            train_dataset=sft_trainer_args["train_dataset"],
            eval_dataset=sft_trainer_args["eval_dataset"],
            peft_config=sft_trainer_args["peft_config"],
            dataset_text_field=sft_trainer_args["dataset_text_field"],
            tokenizer=sft_trainer_args["tokenizer"],
            packing=sft_trainer_args["packing"],
            max_seq_length=sft_trainer_args["max_seq_length"])


        start_time = perf_counter()
        trainer.train()
        end_time = perf_counter()
        output_time = end_time - start_time
        output_time = output_time / 60
        output_time = " ".join([str(round(output_time,2)), "minutes"])


        return sft_trainer_args, output_time

        ### this will automatically save a checkpoint in the specified output_dir
        ### just like how our classic TFT had

# COMMAND ----------

with mlflow.start_run() as run:
    training_model = training_model(
        model=model,
        tokenizer=tokenizer,
        signature=signature,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        mlflow_dir=mlflow_dir,
        # lora_config_dict=lora_config_dict,
        # training_args_dict=training_args_dict,
    )
    
    mlflow.log_param("lora_config", training_model.lora_config_dict)
    mlflow.log_param("tokenizer_specs", training_model.tokenizer_specs)
    mlflow.log_param("training_args_dict", training_model.training_args_dict)

    sft_trainer_args, output_time = training_model.predict()
    
    mlflow.log_param("sft_trainer_args", sft_trainer_args)
    mlflow.log_param("output_time", output_time)
    
    mlflow.pyfunc.log_model(
        signature=signature,
        artifact_path="/".join([mlflow_dir, "logged_model"]),
        python_model=training_model
    )

    mlflow.pyfunc.save_model(
        signature=signature,
        path="/".join([mlflow_dir, "saved_model"]),
        python_model=training_model
   )

