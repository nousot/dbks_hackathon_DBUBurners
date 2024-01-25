def get_default_LORA_config():
    return {
        "r": 16, # attention heads
        "lora_alpha": 32, # alpha scaling
        "target_modules": ["k_proj","o_proj","q_proj","v_proj", "down_proj", "gate_proj", "up_proj"], # based on Lora paper, we want all linear layers
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION",
    }

def get_default_training_args():
    return {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 0,
        "max_steps": 300,
        "learning_rate": 2e-5,
        "fp16": True, # use mixed precision training
        "logging_steps": 1,
        "optim": "paged_adamw_8bit", #adamw kept coming up deprecated
        "save_strategy": "epoch"
    }