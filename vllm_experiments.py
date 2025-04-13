import torch
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset
from utils import initialize_tokenizer, tokenize_blocksworld_generation, DOMAIN_PHRASES
from collections import OrderedDict


from vllm import TokensPrompt, SamplingParams, LLM
from pathlib import Path

compute_dtype = torch.float
device   = 'cuda'
model_id = "Qwen/QwQ-32B"



cur_dir = Path(".").absolute()


def load_dataset_from_file(domain_name, task_name):
    prompt_dir = cur_dir / Path(f"./cot-planning/results/{domain_name}/qwq-32b/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)


task_name = "plan_generation_po"
eval_results = [
    load_dataset_from_file(domain_name, task_name)["instances"] for domain_name in [
        "blocksworld_mystery_2",
    ]
]
eval_results = [{x["dataset_idx"]: x for x in er} for er in eval_results]

tokenizer = initialize_tokenizer(model_id)

dataset = load_dataset(f"dmitriihook/qwq-32b-planning-mystery-2-24k")["train"]


def extract_all_phrase_positions(tokens, phrase, tokenizer, cot_only=True):
    """Find end of the phrase token positions"""
    tokens = tokens.squeeze()

    phrase_tokens = [
        tokenizer.encode(" " + phrase),
        tokenizer.encode(" " + phrase.capitalize()),
        tokenizer.encode("\n" + phrase)[1:],
        tokenizer.encode("\n" + phrase.capitalize())[1:],
        tokenizer.encode("\n\n" + phrase)[1:],
        tokenizer.encode("\n\n" + phrase.capitalize())[1:],
    ]

    positions = set()

    if cot_only:
        start_pos = torch.where(tokens == 151667)[0]
        start_mask = torch.arange(tokens.shape[0]) >= start_pos

    for phts in phrase_tokens:
        presence_mask = torch.ones_like(tokens)
        if cot_only:
            presence_mask = presence_mask * start_mask

        for i, t in enumerate(phts):
            presence_mask = presence_mask * (tokens == t)[i:]
            presence_mask = presence_mask[:-1]

        for p in (torch.where(presence_mask)[0]).tolist():
            positions.add(
                tuple([p-1, p + len(phts)])
            )        
    
    return sorted(list(set(positions)))


llm = LLM(model=model_id, tensor_parallel_size=2, enforce_eager=True, max_seq_len_to_capture=20000, max_num_batched_tokens=4096)

row = dataset[3]

text = "\n\n".join(row["generation"].split("\n\n")[:40])
tokens = tokenize_blocksworld_generation(tokenizer, row, text)[:, :-2][0]

with open(
    "mean_reprs_mystery_2_full.json",
    'r'
) as f:
    reprs = json.load(f)


mean_reprs = {
    k: np.array(v) for k, v in reprs["mean_reprs"].items()
}


mean_actions = np.array(reprs["mean_actions"])
mean_predicates = np.array(reprs["mean_predicates"])


phrases = DOMAIN_PHRASES["mystery_2"]
phrases = list(phrases["actions"].values()) + list(phrases["predicates"].values())


phrase_positions = {
    phrase: extract_all_phrase_positions(tokens, phrase, tokenizer, cot_only=False)
    for phrase in phrases
}


pharse_masks = {
    phrase: np.zeros(tokens.shape[0])
    for phrase in phrases
}

for ip, phrase in enumerate(phrases):
    positions = phrase_positions[phrase]
    
    for start, end in positions:
        pharse_masks[phrase][start:end] = 1


masks_batch = {
    k: [v, v] for k, v in pharse_masks.items()
}

masks_batch_combined = {
    k: np.concatenate(v, axis=0) for k, v in masks_batch.items()
}

combined_len = masks_batch_combined[phrases[0]].shape[0]


masks_batch_combined["illuminate"].shape

block_size = 4096

def hook(module, input, output):
    meta = getattr(module, "_meta", {})
    meta["mask_offset"] = meta.get("mask_offset", 0)
    # print(
    #     meta["mask_offset"],
    # )
    if meta["mask_offset"] >= combined_len:
        return output
    
    mask_start = meta["mask_offset"]
    mask_end = mask_start + block_size
    
    meta["mask_offset"] = mask_end
    module._meta = meta
    
    hs, res = output
  
    
    for ip, phrase in enumerate(phrases):
        if ip < 4:
            adjustment = mean_actions
        else:
            adjustment = mean_predicates
        
        steering_mask = masks_batch_combined[phrase]
        
        steering_mask = steering_mask[mask_start:mask_end]
        steering_mask = np.concatenate([steering_mask, np.zeros(hs.shape[0] - steering_mask.shape[0])], axis=0)
        
        steering_vector = mean_reprs[phrase] - adjustment
        steering_vector = steering_mask[:, None] * steering_vector
        steering_vector = torch.tensor(steering_vector, dtype=hs.dtype, device=hs.device)
    
        hs += steering_vector
    return hs, res

    
    

def add_hook(module):
    module._forward_hooks = OrderedDict()
    module._meta = {}
    module.register_forward_hook(hook)

    


llm.apply_model(
    lambda x: add_hook(x.model.layers[47]),
)

sampling_params = SamplingParams(
    max_tokens=20000,
    temperature=0,
    top_k=1,
)


prompt = TokensPrompt(
    prompt_token_ids=tokens.tolist(),
)


res = llm.generate(
    [prompt, prompt], sampling_params=sampling_params
)
