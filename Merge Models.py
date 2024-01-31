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

# downloads the base model
mlflow.set_registry_uri('databricks-uc')
catalog_name = "databricks_dolly_v2_models" # Default catalog name when installing the model from Databricks Marketplace
model_name = "dolly_v2_3b"
version = 1
model_mlflow_path = f"models:/{catalog_name}.models.{model_name}/{version}"
model_local_path = f"/{model_name}/"
path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)
tokenizer_path = os.path.join(path, "components", "tokenizer")
model_path = os.path.join(path, "model")

# downloads the best adapter
dbfs_path = ''
adapter_path = os.path.join(path, "adapter")
dbutils.fs.cp(dbfs_path, adapter_path)


## for restart post download of model
# import os
# model_name = "dolly_v2_3b"
# model_local_path = f"/{model_name}/"
# path=model_local_path
# tokenizer_path = os.path.join(path, "components", "tokenizer")
# model_path = os.path.join(path, "model")

# COMMAND ----------

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig, BitsAndBytesConfig


quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, #enables 4bit quantization
    bnb_4bit_use_double_quant = False, #repeats quantization a second time if true
    bnb_4bit_quant_type = 'nf4', #`fp4` or `nf4`
    bnb_4bit_compute_dtype = torch.bfloat16, #fp dtype, can be changed for speed up
)

tuned_model_path = "mlflowruns/training/dolly_v2_3b/model2024-01-31 19:46:13/outputs/tuned_model"

model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, load_in_4bit=True), tuned_model_path)
merged_model = model_to_merge.merge_and_unload()

# COMMAND ----------

del model_to_merge

# COMMAND ----------

# # save tokenizer in same spot to make life easier
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# tokenizer.save_pretrained(f"/dbfs/DBUBurners_hackathon/tuned_models_jan_29_part_2/{model_name}")

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

# COMMAND ----------

import pandas as pd
data = pd.read_csv("osha_dataset.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

# COMMAND ----------

import re
from utils import fix_key_names, input_preprocessing
import json

model_name = "dolly_v2_3b"
target_mapping = {}
for index, row in data.iterrows():

    row_output_string = str(row['output'])

    ### this needs a try except block and some cleaning maybe
    modified_string = re.sub(" 'S", "'S", row_output_string)
    modified_string = re.sub(" nan,", '"None",', modified_string)
    modified_string = re.sub(r"(?<!\w)'(?!')|(?<!')'(?!\w)", '"', modified_string)
    modified_string = re.sub(r"\n", ' ', modified_string)
    modified_string = re.sub(r"/", ' or ', modified_string)
    modified_string = re.sub(r'""', '"', modified_string)

    # temp_row_output = json.loads(modified_string)  # Modify data directly

    try:
        temp_row_output = json.loads(modified_string)  # Modify data directly
    except Exception:
        print("Check your data at the following index, as it is not JSON parsable")
        print(index)
        break
    temp_row_output = fix_key_names(dict=temp_row_output, mappings=target_mapping, direction='json_to_schema')
    target_schema_dict = {}
    for key, value in temp_row_output.items():
        target_schema_dict[key] = str(type(value).__name__)

    target_schema_str = str(target_schema_dict)

    # Assign the modified output as a string
    data.at[index, 'output'] = str(temp_row_output)
    row['output'] = str(temp_row_output)

    # Update the row with input preprocessing and concatenate text
    row = input_preprocessing(row, model_name, target_schema_str)
    data.at[index, 'preprocessed_input'] = row['preprocessed_input']
    data.at[index, 'text'] = f"""
    {row['preprocessed_input']}
    {row['output']}
    """

# COMMAND ----------

input_text = data['preprocessed_input'][0]
inputs = tokenizer(input_text, return_tensors='pt')
inputs.to("cuda")

# COMMAND ----------

from time import perf_counter

start_time = perf_counter()

generation_config = GenerationConfig(
    penalty_alpha=0.6, 
    do_sample = True, 
    top_k=5, 
    temperature=0.1, 
    repetition_penalty=10.2,
    max_new_tokens=200,
)


# COMMAND ----------

outputs = merged_model.generate(**inputs, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
end_time = perf_counter()
output_time = end_time - start_time
# with open('tuned_output.txt', 'w') as file:
#     file.write(output_text)
print(f"Time taken for original inference: {round(output_time,2)} seconds")

# COMMAND ----------

data['input'][0]

# COMMAND ----------

data['input'][2]

# COMMAND ----------

print(output_text)
