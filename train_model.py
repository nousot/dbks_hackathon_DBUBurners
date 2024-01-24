from datetime import datetime
from model_setup import ModelSetup
from model_trainer import ModelTrainer

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

class QuickTrain:
    def __init__(self, model_setup: ModelSetup = None, model_trainer: ModelTrainer = None, mlflow_dir: str = None, mlflow_experiment_id: str = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
        self.model_setup = model_setup
        self.model_trainer = model_trainer
        self.mlflow_experiment_id = mlflow_experiment_id

        if mlflow_dir is None:
            self.mlflow_dir = f'mlflowruns/training/{model_name}/{mlflow_experiment_id}'
        else:
            self.mlflow_dir = "/".join([mlflow_dir, mlflow_experiment_id])

    def run_training(self):
        if self.model_setup is None:
            model_setup = 
        training_model = self.training_model

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
                # training_args_dict=training_args_dict,
            )
            
            mlflow.log_param("lora_config", training_model.lora_config_dict)
            mlflow.log_param("tokenizer_specs", training_model.tokenizer_specs)
            mlflow.log_param("training_args_dict", training_model.training_args_dict)

            sft_trainer_args, output_time = training_model.predict()
            
            mlflow.log_param("sft_trainer_args", sft_trainer_args)
            mlflow.log_param("output_time", output_time)
            
            mlflow.pyfunc.log_model(
                signature=model_setup.signature,
                artifact_path="/".join([model_setup.mlflow_dir, "logged_model"]),
                python_model=training_model
            )

            mlflow.pyfunc.save_model(
                signature=model_setup.signature,
                path="/".join([model_setup.mlflow_dir, "saved_model"]),
                python_model=training_model
        )