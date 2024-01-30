# Databricks notebook source
!pip install rich peft optimum trl bitsandbytes accelerate mlflow -U transformers

# COMMAND ----------

# this restarts the python kernel so the above installs go through
dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig, BitsAndBytesConfig
import torch
import os
import mlflow
from mlflow.artifacts import download_artifacts
mlflow.set_registry_uri('databricks-uc')


# COMMAND ----------

import os
import mlflow
from mlflow.artifacts import download_artifacts
model_name="dolly_v2_3b"
tuned_model_path = f"dbfs:/DBUBurners_hackathon/tuned_models_jan_29_part_2/{model_name}"
tuned_model_local_path = "/tuned_model/"
tuned_path = '/tuned_model/{model_name}'

# COMMAND ----------

tuned_path = download_artifacts(artifact_uri=tuned_model_path, dst_path=tuned_model_local_path)

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(tuned_path, local_files_only=True,  device_map="auto", torch_dtype=torch.float16, quantization_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=False,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat8
))
model.to_bettertransformer()

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(tuned_path, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

# COMMAND ----------

import pandas as pd
data = pd.read_csv("arxiv_pdfs_data.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

# COMMAND ----------

input_text = data['input'][0]
inputs = tokenizer(input_text, return_tensors='pt')
inputs.to("cuda")

# COMMAND ----------

if torch.cuda.is_available():
    device = torch.cuda.get_device_name()
    print(f"Using device: {device}")
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Memory allocated on GPU: {allocated_memory/1024/1024/1024} GB")
else:
    print("No GPU available. Using CPU.")

# COMMAND ----------

from time import perf_counter
from peft import prepare_model_for_kbit_training

start_time = perf_counter()

generation_config = GenerationConfig(
    penalty_alpha=0.6, 
    do_sample = True, 
    top_k=5, 
    temperature=0.1, 
    repetition_penalty=10.2,
    max_new_tokens=200,
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# COMMAND ----------

outputs = model.generate(**inputs, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
end_time = perf_counter()
output_time = end_time - start_time
# with open('tuned_output.txt', 'w') as file:
#     file.write(output_text)
print(f"Time taken for original inference: {round(output_time,2)} seconds")

# COMMAND ----------

output_text
