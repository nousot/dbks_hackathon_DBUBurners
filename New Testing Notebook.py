# Databricks notebook source
assert "gpu" in spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"), "THIS PACKAGE REQUIRES THAT A GPU MACHINE AND RUNTIME IS UTILIZED."

# COMMAND ----------

!pip install rich peft optimum trl bitsandbytes accelerate -U transformers
!pip install --upgrade huggingface_hub

# COMMAND ----------

### TURN ON FOLLOWING IF UTILIZING MULTIPLE GPUS (NEEDED FOR DEEPSPEED)
### DEEPSPEED IS SLOWER ON SINGLE GPU
# %sh
# mkdir -p /tmp/externals/cuda

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.50-1_amd64.deb -P /tmp/externals/cuda
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -P /tmp/externals/cuda
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -P /tmp/externals/cuda
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -P /tmp/externals/cuda

# dpkg -i /tmp/externals/cuda/libcurand-dev-11-7_10.2.10.50-1_amd64.deb
# dpkg -i /tmp/externals/cuda/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb
# dpkg -i /tmp/externals/cuda/libcublas-dev-11-7_11.10.1.25-1_amd64.deb
# dpkg -i /tmp/externals/cuda/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb

# COMMAND ----------

# %pip install deepspeed==0.9.1 py-cpuinfo==9.0.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tuning

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig, GenerationConfig, Trainer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from defaults import get_default_LORA_config, get_default_training_args
import tempfile

from utils import input_preprocessing, fix_key_names, format_training_data, count_seq_len
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import torch
from mlflow.artifacts import download_artifacts


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from time import perf_counter

import re

import datetime
import shutil

# COMMAND ----------

tmpdir = tempfile.TemporaryDirectory()
local_training_root = tmpdir.name

# COMMAND ----------

base_model_catalog = "databricks_t5_models"
base_model_name = "t5_3b"
base_model_version = 1
# adapter_dbfs_path = f"/dbfs/tuned_adapters/{base_model_name}/2024-02-09 16:44:23"
gpu_or_cpu = "gpu"

# COMMAND ----------

data = pd.read_csv("osha_dataset.csv")
data.head(5)
data["input"] = data["raw"]
data["output"] = data["structured"]
data = data.drop(columns=["raw", "structured"])

target_schema = {'summary_nr': 'int', 'Event Date': 'str', 'Event Description': 'str', 'Event Keywords': 'str', 'con_end': 'str', 'Construction End Use': 'str', 'build_stor': 'int', 'Building Stories': 'str', 'proj_cost': 'str', 'Project Cost': 'str', 'proj_type': 'str', 'Project Type': 'str', 'Degree of Injury': 'str', 'nature_of_inj': 'int', 'Nature of Injury': 'str', 'part_of_body': 'int', 'Part of Body': 'str', 'event_type': 'int', 'Event type': 'str', 'evn_factor': 'int', 'Environmental Factor': 'str', 'hum_factor': 'int', 'Human Factor': 'str', 'task_assigned': 'int', 'Task Assigned': 'str', 'hazsub': 'str', 'fat_cause': 'int', 'fall_ht': 'int'}

if 'preprocessed_input' not in data.columns:
    target_schema_str = str(target_schema)
    data = format_training_data(data=data, target_schema_str=target_schema_str)
data = data[["text", "label"]]

train, test = train_test_split(data, test_size=0.3, random_state=1738)
train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)
data = Dataset.from_pandas(data)

# COMMAND ----------

data[0].keys()

# COMMAND ----------

def get_model_from_catalog(catalog_name: str, model_name: str, version: int):
    mlflow.set_registry_uri('databricks-uc')

    model_mlflow_path = f"models:/{catalog_name}.models.{model_name}/{version}"
    model_local_path = f"/{model_name}/"

    path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)

    tokenizer_path = os.path.join(path, "components", "tokenizer")
    model_path = os.path.join(path, "model")

    return model_path, tokenizer_path

# COMMAND ----------

base_model_path, base_tokenizer_path = get_model_from_catalog(
    catalog_name = base_model_catalog,
    model_name = base_model_name,
    version = base_model_version
)

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path, device_map="auto")

# COMMAND ----------

max_length = count_seq_len(data)
def get_tokens(tokenizer) -> callable:

    def apply(x) -> transformers.tokenization_utils_base.BatchEncoding:
        """From a formatted dataset `x` a batch encoding `token_res` is created."""
        token_res = tokenizer(
            x["text"],
            text_target = x["label"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        return token_res
    
    return apply

dataset_tokens = get_tokens(tokenizer)
tokenized_data = data.map(dataset_tokens, batched=True, remove_columns=["text", "label"])
tokenized_train = train.map(dataset_tokens, batched=True, remove_columns=["text", "label"])
tokenized_test = test.map(dataset_tokens, batched=True, remove_columns=["text", "label"])



# COMMAND ----------

tokenized_test[0].keys()

# COMMAND ----------

if '__index_level_0__' in tokenized_data[0].keys():
    tokenized_data = tokenized_data.remove_columns('__index_level_0__')
if '__index_level_0__' in tokenized_train[0].keys():
    tokenized_train = tokenized_train.remove_columns('__index_level_0__')
if '__index_level_0__' in tokenized_test[0].keys():
    tokenized_test = tokenized_test.remove_columns('__index_level_0__')

# COMMAND ----------

checkpoint_name = f"trainer/{base_model_name}"
local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)
# training_args = TrainingArguments(
#     local_checkpoint_path,
#     num_train_epochs=1,  # default number of epochs to train is 3
#     per_device_train_batch_size=16,
#     optim="adamw_torch",
#     report_to=["tensorboard"],
# )

training_args_dict = get_default_training_args()

training_args_dict['max_steps'] = 2000

training_args = TrainingArguments(
                per_device_train_batch_size=training_args_dict['per_device_train_batch_size'],
                gradient_accumulation_steps=training_args_dict['gradient_accumulation_steps'],
                warmup_steps=training_args_dict['warmup_steps'],
                max_steps=training_args_dict['max_steps'],
                learning_rate=training_args_dict['learning_rate'],
                fp16=training_args_dict['fp16'],
                logging_steps=training_args_dict['logging_steps'],
                output_dir=local_checkpoint_path,
                optim=training_args_dict['optim'],
                save_strategy=training_args_dict['save_strategy'],
                ddp_find_unused_parameters=training_args_dict['ddp_find_unused_parameters'], 
                push_to_hub=training_args_dict["push_to_hub"],
        )

# COMMAND ----------

quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit = True, #enables 4bit quantization
            bnb_4bit_use_double_quant = False, #repeats quantization a second time if true
            bnb_4bit_quant_type = 'nf4', #`fp4` or `nf4`
            bnb_4bit_compute_dtype = torch.bfloat16, #fp dtype, can be changed for speed up
        )

# COMMAND ----------

if "t5" in base_model_name:
    # model specific architecture, most are causal lm though
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path,
        # quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True, 
        # quantization_config=quantization_config,
        device_map="auto"
    )

# COMMAND ----------

# more efficient data batching when collated
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# COMMAND ----------


## Default in tutorial
## NO PEFT ANYTHING, JUST THIS
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

lora_config_dict = get_default_LORA_config()
if "target_modules" not in lora_config_dict.keys():
    model_modules = str(model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)
    names = []
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))
    lora_config_dict["target_modules"] = target_modules

config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=lora_config_dict["task_type"]
)

args = training_args

trainer_specific_args = {
    "args": args,
    "train_dataset": train,
    "eval_dataset": test,
    "dataset_text_field": "preprocessed_input",
    "packing": False,
}

trainer = Trainer(
    model = model,
    args = args,
    train_dataset= tokenized_train,
    eval_dataset= tokenized_test,
    data_collator = data_collator,
    tokenizer = tokenizer
)
if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 - Train

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Before starting the training process, let's turn on Tensorboard. This will allow us to monitor the training process as checkpoint logs are created.

# COMMAND ----------

tensorboard_display_dir = f"{local_checkpoint_path}/runs"

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

# MAGIC %md
# MAGIC Start the fine-tuning process.

# COMMAND ----------

start_time = perf_counter()
trainer.train()
end_time = perf_counter()
output_time = end_time - start_time
output_time = output_time / 60
output_time = " ".join([str(round(output_time,2)), "minutes"])


# COMMAND ----------

best_model_path = f"{local_checkpoint_path}/tuned_model"
trainer.save_model(best_model_path)


# COMMAND ----------

def upload_adapter_to_dbfs(best_adapter_path: str = None, dbfs_tuned_adapter_dir: str = None):
    shutil.copytree(best_adapter_path, dbfs_tuned_adapter_dir)

    print("You can find your tuned adapter in the following directory on the DBFS")
    print(dbfs_tuned_adapter_dir)

# COMMAND ----------

now = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
dbfs_tuned_adapter_dir = f"/dbfs/tuned_adapters/{base_model_name}/{now}/"
upload_adapter_to_dbfs(best_model_path, dbfs_tuned_adapter_dir)

# COMMAND ----------

hello

# COMMAND ----------

# save model to the local checkpoint
trainer.save_model()
trainer.save_state()

# COMMAND ----------

# persist the fine-tuned model to DBFS
final_model_path = f"{DA.paths.working_dir}/llm04_fine_tuning/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 - Predict

# COMMAND ----------

fine_tuned_model = tr.AutoModelForSeq2SeqLM.from_pretrained(final_model_path)

# COMMAND ----------

reviews = [
    """
'Despicable Me' is a cute and funny movie, but the plot is predictable and the characters are not very well-developed. Overall, it's a good movie for kids, but adults might find it a bit boring.""",
    """ 'The Batman' is a dark and gritty take on the Caped Crusader, starring Robert Pattinson as Bruce Wayne. The film is a well-made crime thriller with strong performances and visuals, but it may be too slow-paced and violent for some viewers.
""",
    """
The Phantom Menace is a visually stunning film with some great action sequences, but the plot is slow-paced and the dialogue is often wooden. It is a mixed bag that will appeal to some fans of the Star Wars franchise, but may disappoint others.
""",
    """
I'm not sure if The Matrix and the two sequels were meant to have a tigh consistency but I don't think they quite fit together. They seem to have a reasonably solid arc but the features from the first aren't in the second and third as much, instead the second and third focus more on CGI battles and more visuals. I like them but for different reasons, so if I'm supposed to rate the trilogy I'm not sure what to say.
""",
]
inputs = tokenizer(reviews, return_tensors="pt", truncation=True, padding=True)
pred = fine_tuned_model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)

# COMMAND ----------

pdf = pd.DataFrame(
    zip(reviews, tokenizer.batch_decode(pred, skip_special_tokens=True)),
    columns=["review", "classification"],
)
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DeepSpeed
# MAGIC
# MAGIC As model architectures evolve and grow, they continually push the limits of available computational resources. For example, some large LLMs having hundreds of billions of parameters making them too large to fit, in some cases, in available GPU memory. Models of this scale therefore need to leverage distributed processing or high-end hardware, and sometimes even both, to support training efforts. This makes large model training a costly undertaking, and therefore accelerating the training process is highly desirable.
# MAGIC
# MAGIC As mentioned above, one such framework that can be leveraged to accelerate the model training process is Microsoft's [DeepSpeed](https://github.com/microsoft/DeepSpeed) [[paper]](https://arxiv.org/pdf/2207.00032.pdf). This framework provides advances in compression, distributed training, mixed precision, gradient accumulation, and checkpointing.
# MAGIC
# MAGIC It is worth noting that DeepSpeed is intended for large models that do not fit into device memory. The `t5-base` model we are using is not a large model, and therefore DeepSpeed is not expected to provide a benefit.
# MAGIC
# MAGIC ### !! Please do not attempt this in Vocareum as it will take more than 5 hours to run and exhaust your compute budget!!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Environment Setup
# MAGIC
# MAGIC The intended use for DeepSpeed is in a distributed compute environment. As such, each node of the environment is assigned a `rank` and `local_rank` in relation to the size of the distributed environment.
# MAGIC
# MAGIC Here, since we are testing with a single node/GPU environment we will set the `world_size` to 1, and both `ranks` to 0.

# COMMAND ----------

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration
# MAGIC
# MAGIC There are a number of [configuration options](https://www.deepspeed.ai/docs/config-json/) that can be set to enhance the training and inference processes. The [ZeRO optimization](https://www.deepspeed.ai/training/#memory-efficiency) settings target reducing the memory footprint allowing for larger models to be efficiently trained on limited resources. 
# MAGIC
# MAGIC The Hugging Face `TrainerArguments` accept the configuration either from a JSON file or a dictionary. Here, we will define the dictionary. 

# COMMAND ----------

zero_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}

# COMMAND ----------

model_checkpoint = "t5-base"
tokenizer = tr.AutoTokenizer.from_pretrained(
    model_checkpoint, cache_dir=DA.paths.datasets
)

imdb_to_tokens = to_tokens(tokenizer, imdb_label_lookup)
tokenized_dataset = imdb_ds.map(
    imdb_to_tokens, batched=True, remove_columns=["text", "label"]
)

model = tr.AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint, cache_dir=DA.paths.datasets
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC There are only two changes made to the training setup from above. The first is to set a new checkpoint name. The second is to add the `deepspeed` configuration to the `TrainingArguments`.
# MAGIC
# MAGIC Note: at this time the `deepspeed` argument is considered an experimental feature and may evolve in the future.

# COMMAND ----------

checkpoint_name = "test-trainer-deepspeed"
checkpoint_location = os.path.join(local_training_root, checkpoint_name)
training_args = tr.TrainingArguments(
    checkpoint_location,
    num_train_epochs=3,  # default number of epochs to train is 3
    per_device_train_batch_size=8,
    deepspeed=zero_config,  # add the deepspeed configuration
    report_to=["tensorboard"],
)

data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)
trainer = tr.Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# COMMAND ----------

tensorboard_display_dir = f"{checkpoint_location}/runs"

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

trainer.train()

trainer.save_model()
trainer.save_state()

# COMMAND ----------

# persist the fine-tuned model to DBFS
final_model_path = f"{DA.paths.working_dir}/llm04_fine_tuning/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

fine_tuned_model = tr.AutoModelForSeq2SeqLM.from_pretrained(final_model_path)

# COMMAND ----------

review = [
    """
           I'm not sure if The Matrix and the two sequels were meant to have a tight consistency but I don't think they quite fit together. They seem to have a reasonably solid arc but the features from the first aren't in the second and third as much, instead the second and third focus more on CGI battles and more visuals. I like them but for different reasons, so if I'm supposed to rate the trilogy I'm not sure what to say."""
]
inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)

pred = fine_tuned_model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)

# COMMAND ----------

pdf = pd.DataFrame(
    zip(review, tokenizer.batch_decode(pred, skip_special_tokens=True)),
    columns=["review", "classification"],
)
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

tmpdir.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
