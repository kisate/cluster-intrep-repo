import os
import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import re
from utils import initialize_tokenizer, tokenize_blocksworld_generation, THINK_TOKEN, THINK_START_TOKEN, DOMAIN_PHRASES

# Configure environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Constants
COMPUTE_DTYPE = torch.bfloat16
DEVICE = 'cuda'
MODEL_ID = "Qwen/QwQ-32B"
MODEL_ID = "Qwen/Qwen2.5-32B"
model_type = "qwq"
model_name = "qwq-32b"
LAYERS = list(range(50))
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
    prompt_dir = CUR_DIR / Path(f"./cot-planning/results/{domain_name}/{model_name}/")
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
        load_dataset(f"dmitriihook/{model_name}-planning-{dataset_type}")["train"]
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
        tokens = tokenize_blocksworld_generation(tokenizer, row, model_type=model_type)
        with torch.no_grad():
            hs = model(tokens.to(DEVICE), output_hidden_states=True).hidden_states
            for layer in layers:
                hidden_states[idx][layer] = hs[layer][0].cpu().to(torch.float16).numpy()
    return hidden_states

def extract_phrase_representations_batched(ids, dataset, positions_dataset, hidden_states, layers, phrases, steps, cont_size):
    """Extract representations for phrases at different steps"""
    extracted_positions = defaultdict(list)
    hs = []
    
    for idx in ids:
        row = dataset[idx]
        generation = row["generation"]
        text = generation.split("</think>")[0]
        tokens = tokenize_blocksworld_generation(tokenizer, row, text, model_type=model_type)[0]
        end_pos = len(tokens)
        print(
            f"Tokens: {end_pos}"
        )

        for phrase in phrases:
            positions = extract_all_phrase_positions(tokens, phrase)
            positions = [p for p in positions if p[0] < end_pos]
            extracted_positions[phrase].append(positions)

        hs.append(hidden_states[idx])

    layer_reprs = {}

    for layer in tqdm(layers):
        reprs = defaultdict(list)
        for phrase in phrases:
            for step in steps:
                ph_positions = extracted_positions[phrase]
                step_reprs = []
                for i, positions in enumerate(ph_positions):
                    _hs = hs[i][layer]
                    positions = [p for p in positions if p[0] > step - cont_size and p[1] < step]
                    if len(positions) == 0:
                        continue

                    for p in positions:
                        if p[1] - p[0] > 1 and p[1] < _hs.shape[0]:
                            step_reprs.append(_hs[p[0]:p[1]].mean(axis=0))
                
                # print(f"{phrase} {step}: {len(step_reprs)} occurrences")
                
                if step_reprs:
                    reprs[f"{phrase}_{step}"] = np.stack(step_reprs).mean(axis=0)

        layer_reprs[layer] = reprs
            
    return layer_reprs

def make_mean_reprs(reprs, phrases, steps, n_last=3, offset=1):
    """Make mean representations for phrases"""
    mean_reprs = defaultdict(lambda: defaultdict(list))

    for layer, _reprs in reprs.items():
        for phrase in phrases:
            for step in steps[-n_last-offset:-offset]:
                name = f"{phrase}_{step}"
                if name not in _reprs:
                    continue
                mean_reprs[layer][phrase].append(_reprs[name])
            
    return {l: {k: np.stack(v, axis=0).mean(axis=0) for k, v in lr.items()} for l, lr in mean_reprs.items()}


def save_representations(mean_reprs, domain_name, output_dir):
    """Save representations to a JSON file"""

    data_to_save = {}

    domain_phrases = DOMAIN_PHRASES[domain_name]
    action_phrases = list(domain_phrases["actions"].values())
    predicate_phrases= []
    
    for layer, layer_reprs in mean_reprs.items():
        mean_domain = np.stack(list(layer_reprs.values())).mean(0).tolist()

        action_reprs = [layer_reprs[a] for a in action_phrases]
        predicate_reprs = [layer_reprs[p] for p in predicate_phrases]

        mean_actions = np.stack(action_reprs).mean(0).tolist()
        mean_predicates = np.zeros_like(mean_actions).tolist()

        data_to_save[layer] = {
            "mean_domain": mean_domain,
            "mean_actions": mean_actions,
            "mean_predicates": mean_predicates,
            "mean_reprs": {k: v.tolist() for k, v in layer_reprs.items()}
        }
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"mean_reprs_{domain_name}.json"
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(data_to_save, f)
    
    print(f"Saved representations for domain {domain_name}")
    
def main():
    # Initialize model and tokenizer
    initialize_model_and_tokenizer()
    
    # Create output directory
    output_dir = "./multilayer_representations_base/clean_2k"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set domain names and dataset types for clean domain
    domain_names = ["blocksworld_4_blocks"]
    dataset_types = ["4-blocks"]
    
    # Get domain-specific phrases
    clean_domain_phrases = DOMAIN_PHRASES["clean"]
    
    # Extract action and predicate phrases
    clean_action_phrases = list(clean_domain_phrases["actions"].values())
    clean_predicate_phrases = []
    clean_phrases = clean_action_phrases + clean_predicate_phrases
    
    try:
        # Load datasets
        eval_results, datasets, label_datasets = load_datasets(domain_names, dataset_types)
        
        # Find label positions
        # label_positions = [
        #     find_label_positions(ld, dataset) 
        #     for ld, dataset in zip(label_datasets, datasets)
        # ]
        
        label_positions = [
            None for _ in datasets
        ]
        
        # Collect correct IDs
        correct_ids = [
            collect_correct_ids(er) for er in eval_results
        ]
        
        # Set steps for phrase extraction
        steps_clean = list(range(1000, 2000, 100))
        
        # Collect hidden states for clean domain
        hidden_states_clean = collect_hidden_states(correct_ids[0], datasets[0], LAYERS)
        
        # Extract phrase representations
        reprs_clean = extract_phrase_representations_batched(
            correct_ids[0][:N_ROWS], 
            datasets[0], 
            label_positions[0], 
            hidden_states_clean, 
            LAYERS, 
            clean_phrases, 
            steps_clean, 
            CONT_SIZE
        )
        
        # Make mean representations
        clean_reprs = make_mean_reprs(reprs_clean, clean_phrases, steps_clean)
    
        # Save clean representations
        save_representations(clean_reprs, "clean", output_dir)
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        raise e

if __name__ == "__main__":
    main()