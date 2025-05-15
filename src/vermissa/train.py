import argparse
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from data import create_dataset
from utils import load_config
import logging


def create_trainer(config):
  model_cfg = config["model"]
  dataset = create_dataset(config)
  model = AutoModelForCausalLM.from_pretrained(model_cfg["name"])
  trainer =  Trainer(
      model=model,
      args=TrainingArguments(
          output_dir=model_cfg["output_dir"],
          num_train_epochs=model_cfg["training"]["num_epochs"],
          per_device_train_batch_size=model_cfg["training"]["batch_size"],
          save_steps=model_cfg["training"]["save_steps"],
          logging_steps=model_cfg["training"]["logging_steps"],
          logging_dir=model_cfg["training"]["logging_dir"],
          report_to=[],  # disable WandB
      ),
      train_dataset=dataset
  )
  logging.info(f"creating trainer: {trainer.model}")
  return trainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tuning for Vermissa")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to the YAML config file",
        default="../../configs/config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = create_trainer(config)
    
    # Start fine-tuning
    # trainer.train()