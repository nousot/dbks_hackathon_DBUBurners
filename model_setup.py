import pandas as pd
from utils import input_preprocessing, fix_key_names, format_training_data
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
import tiktoken

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from accelerate import accelerator


class ModelSetup:
    def __init__(self, model_name: str, raw_data: pd.DataFrame, tokenizer_path: str = None, training_data_path: str = None, mlflow_dir: str = None, mlflow_experiment_id: str = None):
        self.model_name = model_name

        if tokenizer_path is None:
            self.tokenizer_path = model_name
        else:
            self.tokenizer_path = tokenizer_path

        if mlflow_experiment_id is None:
            self.mlflow_experiment_id = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            self.mflow_experiment_id = mlflow_experiment_id

        if mlflow_dir is None:
            self.mlflow_dir = f'mlflowruns/training/{model_name}/{mlflow_experiment_id}'
        else:
            self.mlflow_dir = "/".join([mlflow_dir, mlflow_experiment_id])

        self.mlflow_experiment_id = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self.raw_data = raw_data
        if training_data_path is None:
            current_dir = os.path.abspath(os.path.dirname(__file__))
            training_data_path = os.path.join(current_dir, "training_data")
            self.training_data_path = training_data_path
        else:
            self.training_data_path = training_data_path

        self.cleaned_data = None
        self.signature = None
        self.input_example = []
        self.train = None
        self.test = None
        self.dataset = None
        self.train_dataset = None
        self.eval_dataset = None

        self.base_model = None
        self.model = None
        self.tokenizer = None

    def create_training_delta_table(self, data: pd.DataFrame = None, target_mapping: dict[str:str] = {}, training_data_path: str = None):
        if data is None:
            data = self.raw_data
        if training_data_path is None:
            training_data_path = self.training_data_path

        data = format_training_data(data=data)
        spark = SparkSession.builder.appName("ModelSetup").getOrCreate()
        spark_df = spark.createDataFrame(data)
        spark_df = spark_df.select(col("input"), col("preprocessed_input"), col("output"), col("text"))
        spark_df.write.format("delta").mode("overwrite").save(training_data_path)
        spark.stop()
        self.cleaned_data = data
        return data

    def create_local_training_data(self, data: pd.DataFrame = None, target_mapping: dict[str:str] = {}, training_data_path: str = None):
        if data is None:
            data = self.raw_data
        if training_data_path is None:
            training_data_path = self.training_data_path

        data = format_training_data(data=data)
        self.cleaned_data = data
        return data

    def get_signature(self, cleaned_data: pd.DataFrame = None):
        if cleaned_data is None:
            cleaned_data = self.cleaned_data
        self.signature = infer_signature(model_input=cleaned_data)
        return self.signature

    def get_input_example(self, cleaned_data: pd.DataFrame = None):
        if cleaned_data is None:
            cleaned_data = self.cleaned_data
        self.input_example = cleaned_data.head(5)
        return cleaned_data.head(5)

    def count_seq_len(self):
        example_text = self.get_input_example.at[0, 'input']
        #rough estimate here
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = len(encoding.encode(example_text))
        return token_count

    def get_train_test_split(self, cleaned_data: pd.DataFrame = None, test_size: float = 0.3):
        if cleaned_data is None:
            cleaned_data = self.cleaned_data
        self.train, self.test = train_test_split(cleaned_data, test_size=test_size, random_state=1738)
        return self.train, self.test

    def get_all_data_as_datasets(self, cleaned_data: pd.DataFrame = None, train: pd.DataFrame = None, test: pd.DataFrame = None):
        if cleaned_data is None:
            cleaned_data = self.cleaned_data
        if train is None:
            train = self.train
        if test is None:
            test = self.test

        self.dataset = Dataset.from_pandas(cleaned_data)
        self.train_dataset = Dataset.from_pandas(train)
        self.eval_dataset = Dataset.from_pandas(test)
        return self.dataset, self.train_dataset, self.eval_dataset

    def prepare_model_and_tokenizer(self, model_name: str = None, tokenizer_path: str = None):
        if model_name is None:
            model_name = self.model_name

        if tokenizer_path is None:
            tokenizer_path = self.tokenizer_path

        # quantization_config = GPTQConfig(bits=4, disable_exllama=False)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True, #enables 4bit quantization
            bnb_4bit_use_double_quant = False, #repeats quantization a second time if true
            bnb_4bit_quant_type = 'nf4', #`fp4` or `nf4`
            bnb_4bit_compute_dtype = torch.bfloat16, #fp dtype, can be changed for speed up
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quantization_config, device_map="auto", load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.base_model = model

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def quickstart(self):
        self.cleaned_data = self.create_local_training_data()
        self.signature = self.get_signature()
        self.input_example = self.get_input_example()
        self.sft_seq_arg  = self.count_seq_len()
        self.train, self.test = self.get_train_test_split()
        self.dataset, self.train_dataset, self.eval_dataset = self.get_all_data_as_datasets()
        self.model, self.tokenizer = self.prepare_model_and_tokenizer()
