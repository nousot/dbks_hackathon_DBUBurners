import logging
import logging
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig
# from datasets import Dataset
import torch
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
import re


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from datetime import datetime

from defaults import get_default_LORA_config, get_default_training_args
from delta.tables import DeltaTable
from pyspark.sql.functions import col


class ModelTrainer(mlflow.pyfunc.PythonModel):

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

        if "target_modules" not in self.lora_config_dict.keys():
            model_modules = str(self.model.modules)
            pattern = r'\((\w+)\): Linear'
            linear_layer_names = re.findall(pattern, model_modules)
            names = []
            for name in linear_layer_names:
                names.append(name)
            target_modules = list(set(names))
            self.lora_config_dict["target_modules"] = target_modules

        print("Using the following lora config:")
        print(str(self.lora_config_dict))

        for key, value in get_default_training_args().items():
            if key not in self.training_args_dict.keys():
                self.training_args_dict[key] = value
        if 'output_dir' not in self.training_args_dict.keys():
            self.training_args_dict['output_dir'] = "/".join([self.mlflow_dir, "outputs"])
        
        print("Using the following training_args:")
        print(str(self.training_args_dict))


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
        if torch.cuda.device_count() > 1: # If more than 1 GPU
            model.is_parallelizable = True
            model.model_parallel = True

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
                save_strategy=self.training_args_dict['save_strategy'],
                ddp_find_unused_parameters=self.training_args_dict['ddp_find_unused_parameters']
        )

        sft_trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "peft_config": config,
            "dataset_text_field": "text", # field to tune on
            "tokenizer": self.tokenizer,
            "packing": False, #unsure what this entails
            "max_seq_length": 100
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