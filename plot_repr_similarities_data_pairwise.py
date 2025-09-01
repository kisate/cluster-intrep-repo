import os
import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
from collections import defaultdict
from scipy.spatial.distance import cosine
import pandas as pd
from utils import initialize_tokenizer, tokenize_blocksworld_generation, THINK_TOKEN, THINK_START_TOKEN, DOMAIN_PHRASES
from tqdm import trange
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Constants
COMPUTE_DTYPE = torch.bfloat16
DEVICE = 'cuda'
MODEL_ID = "Qwen/QwQ-32B"
# MODEL_ID = "Qwen/Qwen2.5-32B"
LAYER = 44
N_ROWS = 40
CONT_SIZE = 100

# Initialize current directory
CUR_DIR = Path(".").absolute()

# Global variables for model and tokenizer
tokenizer = None
model = None

def initialize_model_and_tokenizer():
    """Initialize the model and tokenizer globally"""
    global tokenizer, model
    
    tokenizer = initialize_tokenizer(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=COMPUTE_DTYPE, 
        attn_implementation="sdpa", 
        device_map="auto"
    )
    
    return tokenizer, model

def load_dataset_from_file(domain_name, task_name):
    """Load dataset from a JSON file"""
    prompt_dir = CUR_DIR / Path(f"./cot-planning/results/{domain_name}/qwq-32b-greedy/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

def load_datasets(domain_names, dataset_types):
    """Load datasets and prepare evaluation results"""
    task_name = "plan_generation_po"
    
    # Load evaluation results
    eval_results = [
        load_dataset_from_file(domain_name, task_name)["instances"] 
        for domain_name in domain_names
    ]
    eval_results = [{x["dataset_idx"]: x for x in er} for er in eval_results]
    
    # Load datasets
    datasets = [
        load_dataset(f"dmitriihook/qwq-32b-planning-{dataset_type}-greedy")["train"]
        # load_dataset(f"dmitriihook/llama-3_3-nemotron-super-49b-v1-planning-{dataset_type}-greedy")["train"]
        for dataset_type in dataset_types
    ]
    
    # Load label datasets
    # label_datasets = [
    #     load_dataset(f"dmitriihook/{domain_name.replace('_', '-')}-qwq-reasoning-parts-exploration")["train"]
    #     for domain_name in domain_names
    # ]
    
    # label_datasets = [
    #     {x["index"]: x for x in ld} for ld in label_datasets
    # ]
    
    label_datasets = [
        None for _ in domain_names
    ]
    
    return eval_results, datasets, label_datasets

def extract_all_phrase_positions(tokens, phrase, cot_only=True):
    """Find all phrase positions in the tokens"""
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
        think_token_id = 151667  # This should be THINK_TOKEN or an appropriate constant
        start_pos = torch.where(tokens == think_token_id)[0]
        if len(start_pos) > 0:
            start_mask = torch.arange(tokens.shape[0]) >= start_pos[0]
        else:
            start_mask = torch.ones_like(tokens).bool()

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
    
    return sorted(list(positions))

def collect_correct_ids(eval_results):
    """Collect IDs of correct examples"""
    clean_ids = []
    for idx in range(len(eval_results)):
        if idx in eval_results and eval_results[idx]["llm_correct"]:
            clean_ids.append(idx)
        if len(clean_ids) == N_ROWS:
            break
    return clean_ids

def collect_hidden_states(ids, dataset, layers):
    """Collect hidden states for the given IDs and layers"""
    hidden_states = defaultdict(dict)
    for idx in tqdm(ids):
        row = dataset[idx]
        tokens = tokenize_blocksworld_generation(tokenizer, row)
        with torch.no_grad():
            hs = model(tokens.to(DEVICE), output_hidden_states=True).hidden_states
            for layer in layers:
                hidden_states[idx][layer] = hs[layer][0].cpu().to(torch.float16).numpy()
    return hidden_states

def extract_phrase_representations_batched(ids, dataset, positions_dataset, hidden_states, layers, phrases, steps, cont_size):
    """Extract representations for phrases at different steps and layers"""
    # First collect all positions and tokens
    extracted_positions = defaultdict(list) 
    tokens_list = []
    end_positions = []
    
    for idx in ids:
        row = dataset[idx]
        generation = row["generation"]
        text = generation.split("</think>")[0]
        tokens = tokenize_blocksworld_generation(tokenizer, row, text)[0]
        end_pos = len(tokens)
        
        for phrase in phrases:
            positions = extract_all_phrase_positions(tokens, phrase)
            positions = [p for p in positions if p[0] < end_pos]
            extracted_positions[phrase].append(positions)
            
        tokens_list.append(tokens)
        end_positions.append(end_pos)

    # Process each layer separately
    all_layer_reprs = {}
    for layer in layers:
        hs = []
        for idx in ids:
            hs.append(hidden_states[idx][layer])
            
        layer_reprs = defaultdict(list)
        
        for phrase in phrases:
            for step in steps:
                ph_positions = extracted_positions[phrase]
                step_reprs = []
                for i, positions in enumerate(ph_positions):
                    _hs = hs[i]
                    positions = [p for p in positions if p[0] > step - cont_size and p[1] < step]
                    if len(positions) == 0:
                        continue

                    for p in positions:
                        if p[1] - p[0] > 1 and p[1] < _hs.shape[0]:
                            step_reprs.append(_hs[p[0]:p[1]].mean(axis=0))
                
                print(f"Layer {layer}, {phrase} {step}: {len(step_reprs)} occurrences")
                
                if step_reprs:
                    layer_reprs[f"{phrase}_{step}"] = np.stack(step_reprs).mean(axis=0)
                    
        all_layer_reprs[layer] = layer_reprs
            
    return all_layer_reprs

def make_mean_reprs(reprs, phrases, steps, n_last=3, offset=1):
    """Make mean representations for phrases"""
    mean_reprs = defaultdict(list)
    for phrase in phrases:
        for step in steps[-n_last-offset:-offset]:
            name = f"{phrase}_{step}"
            if name not in reprs:
                continue
            mean_reprs[phrase].append(reprs[name])
            
    return {k: np.stack(v, axis=0).mean(axis=0) if v else None for k, v in mean_reprs.items()}


LAYERS = [30, 40]

initialize_model_and_tokenizer()

def compute_mean_reprs(mystery_num):
    """Compute hidden states and mean representations that can be reused across different avg_reprs"""
    domain_names = [f"blocksworld_mystery_{mystery_num}"]
    dataset_types = [f"mystery-{mystery_num}-24k"]
    domain_phrases = DOMAIN_PHRASES[f"mystery_{mystery_num}"]
    action_phrases = list(domain_phrases["actions"].values())
    predicate_phrases = list(domain_phrases["predicates"].values())
    phrases = action_phrases + predicate_phrases

    # Load datasets
    eval_results, datasets, label_datasets = load_datasets(domain_names, dataset_types)
    label_positions = [None for _ in datasets]
    correct_ids = [collect_correct_ids(er) for er in eval_results]
    steps = list(range(1000, 10000, 200))

    # Compute hidden states
    hidden_states = collect_hidden_states(correct_ids[0], datasets[0], LAYERS)
    
    # Compute representations
    reprs = extract_phrase_representations_batched(
        correct_ids[0][:N_ROWS], 
        datasets[0], 
        label_positions[0], 
        hidden_states, 
        LAYERS, 
        phrases, 
        steps, 
        CONT_SIZE
    )

    # Compute mean representations
    last_steps = steps[-3:]
    mean_reprs = {}
    for layer in LAYERS:
        layer_reprs = reprs[layer]
        all_reprs = []
        for phrase in phrases:
            phrase_step_reprs = []
            for step in last_steps:
                key = f"{phrase}_{step}"
                if key in layer_reprs:
                    phrase_step_reprs.append(layer_reprs[key])
            if phrase_step_reprs:
                all_reprs.append(np.mean(phrase_step_reprs, axis=0))
        mean_all = np.mean(all_reprs, axis=0) if all_reprs else None
        mean_reprs[layer] = mean_all

    return mean_reprs, reprs


def process_domain(mystery_num, mean_reprs, reprs, other_num, output_dir="charts"):
    """Process domain using precomputed hidden states and mean representations"""
    os.makedirs(output_dir, exist_ok=True)

    # Extract precomputed data
    domain_phrases = DOMAIN_PHRASES[f"mystery_{mystery_num}"]
    action_phrases = list(domain_phrases["actions"].values())
    predicate_phrases = list(domain_phrases["predicates"].values())
    phrases = action_phrases + predicate_phrases
    steps = list(range(1000, 10000, 200))

    other_domain_phrases = DOMAIN_PHRASES[f"mystery_{other_num}"]

    # Load avg_reprs (this is what can vary between calls)
    repr_file = f"multilayer_representations/multilayer_7k/mystery_{other_num}/mean_reprs_mystery_{other_num}_multi_layer.json"
    with open(repr_file, 'r') as f:
        avg_reprs = json.load(f)

    avg_means = {
        int(k): {
            "mean_actions": v["mean_actions"],
            "mean_predicates": v["mean_predicates"],
            "mean_domain": v["mean_domain"]
        }
        for k, v in avg_reprs.items()
    }
    avg_reprs = {
        int(k): v["mean_reprs"]
        for k, v in avg_reprs.items()
    }

    map_to_other = {
        domain_phrases["actions"][k]: v 
        for k, v in other_domain_phrases["actions"].items()
    }
    map_to_other.update({
        domain_phrases["predicates"][k]: v
        for k, v in other_domain_phrases["predicates"].items()
    })    

    for layer in LAYERS:
        _reprs = reprs[layer]
        _avg_reprs = avg_reprs[layer]
        mean_domain_avg = np.array(avg_means[layer]["mean_domain"])
        action_own_sims = []
        pred_own_sims = []
        action_other_action_sims = []
        pred_other_pred_sims = []
        action_pred_sims = []
        pred_action_sims = []
        for step in steps:
            action_own = []
            pred_own = []
            action_other_action = []
            pred_other_pred = []
            action_pred = []
            pred_action = []
            for phrase in phrases:
                is_action = phrase in action_phrases
                is_predicate = phrase in predicate_phrases
                repr_key = f"{phrase}_{step}"
                if repr_key not in _reprs:
                    continue
                repr_vector = _reprs[repr_key] - mean_reprs[layer]

                avg_phrase = map_to_other[phrase]

                base_avg_vector = np.array(_avg_reprs[avg_phrase]) - mean_domain_avg
                own_sim = 1 - cosine(repr_vector, base_avg_vector)
                if is_action:
                    action_own.append(own_sim)
                else:
                    pred_own.append(own_sim)
                if is_action:
                    other_actions = [p for p in action_phrases if p != phrase]
                    sims = []
                    for other in other_actions:
                        other_avg_phrase = map_to_other[other]
                        other_avg = np.array(_avg_reprs[other_avg_phrase]) - mean_domain_avg
                        sims.append(1 - cosine(repr_vector, other_avg))
                        print(f"Step {step}, {phrase} - {other}: {sims[-1]}")
                    if sims:
                        action_other_action.append(np.mean(sims))
                    sims = []
                    for other in predicate_phrases:
                        other_avg_phrase = map_to_other[other]
                        other_avg = np.array(_avg_reprs[other_avg_phrase]) - mean_domain_avg    
                        sims.append(1 - cosine(repr_vector, other_avg))
                    if sims:
                        action_pred.append(np.mean(sims))
                else:
                    other_preds = [p for p in predicate_phrases if p != phrase]
                    sims = []
                    for other in other_preds:
                        other_avg_phrase = map_to_other[other]
                        other_avg = np.array(_avg_reprs[other_avg_phrase]) - mean_domain_avg
                        sims.append(1 - cosine(repr_vector, other_avg))
                    if sims:
                        pred_other_pred.append(np.mean(sims))
                    sims = []
                    for other in action_phrases:
                        other_avg_phrase = map_to_other[other]
                        other_avg = np.array(_avg_reprs[other_avg_phrase]) - mean_domain_avg
                        sims.append(1 - cosine(repr_vector, other_avg))
                    if sims:
                        pred_action.append(np.mean(sims))
            action_own_sims.append(np.mean(action_own) if action_own else np.nan)
            pred_own_sims.append(np.mean(pred_own) if pred_own else np.nan)
            action_other_action_sims.append(np.mean(action_other_action) if action_other_action else np.nan)
            pred_other_pred_sims.append(np.mean(pred_other_pred) if pred_other_pred else np.nan)
            action_pred_sims.append(np.mean(action_pred) if action_pred else np.nan)
            pred_action_sims.append(np.mean(pred_action) if pred_action else np.nan)

        df = pd.DataFrame({
            "step": steps,
            "action_own": action_own_sims,
            "pred_own": pred_own_sims,
            "action_other_action": action_other_action_sims,
            "pred_other_pred": pred_other_pred_sims,
            "action_pred": action_pred_sims,
            "pred_action": pred_action_sims,
        })
        df.to_csv(os.path.join(output_dir, f"avg_similarities_layer_{layer}_mystery_{other_num}_test.csv"), index=False)

if __name__ == "__main__":
    for mystery_num in trange(1, 16):
        mean_reprs, reprs = compute_mean_reprs(mystery_num)
        out_dir = os.path.join("charts_data", "10k", f"mystery_{mystery_num}")
        for other_num in [2]:
            process_domain(mystery_num, mean_reprs, reprs, other_num, output_dir=out_dir)






