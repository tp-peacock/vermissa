import argparse
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from vermissa.data import create_dataset
from vermissa.utils import load_config
import logging


def create_trainer(config, quantize=False):
  model_cfg = config["model"]
  dataset = create_dataset(config)
  model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], load_in_8bit=quantize)
  trainer =  Trainer(
      model=model,
      args=TrainingArguments(
          output_dir=model_cfg["output_dir"],
          **model_cfg["training"]
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