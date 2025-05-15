import re
import argparse
import requests
import logging
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from vermissa.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class TextDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }


def extract_gutenberg_body(text):
    text = text.replace('\r\n', '\n')

    start_match = re.search(r"\*\*\* *START OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    end_match = re.search(r"\*\*\* *END OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)

    if not start_match or not end_match:
        raise ValueError("Could not find the START or END marker in the text.")

    start_idx = start_match.end()
    end_idx = end_match.start()

    return text[start_idx:end_idx].strip()


def download_raw_text(url, raw_file):
    logging.info("Downloading raw text...")
    response = requests.get(url)
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(response.text)


def clean_text(raw_file, processed_file):
    logging.info("Cleaning text...")
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = extract_gutenberg_body(raw_text)
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    with open(processed_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)


def chunk_tokens(input_ids, block_size):
    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    input_ids = input_ids.view(-1, block_size)
    return input_ids


def tokenize_and_chunk_dataset(clean_path, chunks_path, tokenizer, block_size):
    tokenizer.model_max_length = int(1e9)
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info("Creating dataset chunks...")
    with open(clean_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = tokenizer(raw_text, add_special_tokens=False, truncation=False, max_length=None)["input_ids"]
    tokens = torch.tensor(tokens, dtype=torch.long)
    input_chunks = chunk_tokens(tokens, block_size)

    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    torch.save(input_chunks, chunks_path)
    return input_chunks


def create_dataset(config):
    data_cfg = config["data"]
    model_cfg = config["model"]

    root_dir = data_cfg["root"]
    block_size = data_cfg["block_size"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    raw_path = Path(root_dir).joinpath("stage=raw/part-000.pt") # assumes data chunked into only one file
    clean_path = Path(root_dir).joinpath("stage=clean/part-000.pt") # assumes data chunked into only one file
    chunks_path = Path(root_dir).joinpath("stage=chunked/part-000.pt") # assumes data chunked into only one file
    
    # create chunks, if they do not exist
    if not chunks_path.exists():

        # clean raw data, if it does not exist
        if not clean_path.exists():

            # download raw data, if it does not exist
            if not raw_path.exists():
                url = data_cfg["download_url"]
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                download_raw_text(url, str(raw_path))

            clean_text(str(raw_path), str(clean_path))

        input_chunks = tokenize_and_chunk_dataset(clean_path, chunks_path, tokenizer, block_size)

    else:
        logging.info("Loading dataset chunks...")
        input_chunks = torch.load(chunks_path)
    
    return TextDataset(input_chunks)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data processing for Vermissa")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to the YAML config file",
        default=os.path.expanduser("../../configs/config.yaml")
    )

    args = parser.parse_args()
    config = load_config(args.config)
    dataset = create_dataset(config)