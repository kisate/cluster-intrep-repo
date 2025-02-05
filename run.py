import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

compute_dtype = torch.bfloat16
device   = 'cuda'
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation="sdpa", device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = tokenizer.chat_template.replace("{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", "")


dataset = load_dataset("dmitriihook/numina-deepseek-r1-qwen-7b")["train"]

activations = []

for i, row in enumerate(tqdm(dataset)):
    generation = row["generation"]
    problem = row["problem"]
    messages = [
        {"role":"user", "content": problem},
        {"role": "assistant", "content": generation}
    ]
    chat    = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
    with torch.no_grad():
        outputs = model(chat.to(device), output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1][0]

    activations.append(hidden_states.cpu())

torch.save(activations, "test_activations_7b.pt")
    
