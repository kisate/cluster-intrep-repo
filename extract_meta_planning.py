import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import os
import json
from pathlib import Path

prompt_template = """{{ instruction }}"""


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

compute_dtype = torch.bfloat16
device   = 'cuda'
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation="sdpa", device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("dmitriihook/deepseek-r1-qwen-32b-planning-mystery")["train"]
tokenizer.chat_template = tokenizer.chat_template.replace("{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", "")


print(os.listdir("."))

def load_dataset_from_file(domain_name, task_name):
    prompt_dir = Path(f"./cot-planning/results/{domain_name}/deepseek-32b/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

domain_name = "blocksworld_mystery"
task_name = "plan_generation_po"
parsed_dataset = load_dataset_from_file(domain_name, task_name)["instances"]

metadata = []

for x in parsed_dataset:
    if "dataset_idx" not in x:
        continue

    dataset_idx = x["dataset_idx"]
    row = dataset[dataset_idx]

    generation = row["generation"]

    query = row["distilabel_metadata"]["raw_input_text_generation_0"][0]

    messages = [
        query,
        {"role": "assistant", "content": generation}
    ]
    chat    = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")


    think_pos = torch.where(chat[0] == 151649)[0]

    if len(think_pos) == 0:
        think_pos = None
    else:
        think_pos = think_pos.item()

    item_meta = {
        "dataset_idx": dataset_idx,
        "think_pos": think_pos,
        "total_length": chat[0].shape[0],
        "bench_item": x
    }

    metadata.append(item_meta)


with open("planning_metadata_mystery.json", 'w') as file:
    json.dump(metadata, file)