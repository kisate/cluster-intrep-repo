import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset

prompt_template = """{{ instruction }}"""

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

compute_dtype = torch.bfloat16
device   = 'cuda'
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation="sdpa", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = tokenizer.chat_template.replace("{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", "")


dataset = load_dataset("dmitriihook/deepseek-r1-qwen-32b-planning-big")["train"]

activations = []

def activation_compression(x, comp_type="think_pos", **kwargs):
    if comp_type == "think_pos":
        think_pos = kwargs["think_pos"]
        window_left = kwargs["window_left"]
        window_right = kwargs["window_right"]

        return x[think_pos-window_left:think_pos+window_right].cpu()
    elif comp_type == "step":
        step = kwargs["step"]
        new_x = torch.zeros((x.shape[0]//step, x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(new_x.shape[0]):
            new_x[i] = x[i*step:(i+1)*step].sum(dim=0)

        return new_x
    else:
        raise ValueError(f"Unknown compression type {comp_type}")

    


for i, row in enumerate(tqdm(dataset)):
    generation = row["generation"]
    query = row["distilabel_metadata"]["raw_input_text_generation_0"][0]

    messages = [
        query,
        {"role": "assistant", "content": generation}
    ]
    chat    = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")


    think_pos = torch.where(chat[0] == 151649)[0]

    # print(chat[0].shape)

    if len(think_pos) == 0 or chat[0].shape[0] > 10000:
        hidden_states = None
    else:
        think_pos = think_pos[0]

        with torch.no_grad():
            outputs = model(chat.to(device), output_hidden_states=True)

        # hidden_states = outputs.hidden_states[-1][0][think_pos-600:think_pos+10].cpu()

        hidden_states = activation_compression(
            outputs.hidden_states[-1][0],
            comp_type="step",
            step=10
        )

    activations.append(hidden_states)

torch.save(activations, "planning_activations_32b_big_step.pt")
    
