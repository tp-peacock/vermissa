from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from vermissa.utils import load_config
from pathlib import Path
import re

def find_last_checkpoint(root_path):
    pattern = re.compile(r"checkpoint-(\d+)")
    
    checkpoints = []
    for subdir in Path(root_path).iterdir():
        if subdir.is_dir():
            match = pattern.match(subdir.name)
            if match:
                checkpoints.append(int(match.group(1)))
    
    if not checkpoints:
        return None
    
    return str(max(checkpoints))


def generate(config, prompt, checkpoint=None):

    generate_cfg = config["model"]["generate"]

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token

    if not checkpoint:
        checkpoint = find_last_checkpoint(config['model']['output_dir'])

    model = AutoModelForCausalLM.from_pretrained(f"{config['model']['output_dir']}/checkpoint-{checkpoint}")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        **generate_cfg,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0])


def base_model_generate(config, prompt):

    generate_cfg = config["model"]["generate"]

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        **generate_cfg,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generator for Vermissa")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to the YAML config file",
        default="../../configs/config.yaml"
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        required=True,
        help="Prompt from which to generate model output",
    )    
    args = parser.parse_args()

    config = load_config(args.config)
    prompt = args.prompt

    print("Fine-tuned GPT-2:\n\t", generate(config, prompt))
    print("")
    print("Original GPT-2:\n\t", base_model_generate(config, prompt))



