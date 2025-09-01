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
LAYER = 47
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
        # load_dataset(f"dmitriihook/qwq-32b-planning-{dataset_type}-greedy")["train"]
        load_dataset(f"dmitriihook/llama-3_3-nemotron-super-49b-v1-planning-{dataset_type}-greedy")["train"]
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
            hs = model(tokens.to(DEVICE)[:, :11000], output_hidden_states=True).hidden_states
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


def process_domain(mystery_num, output_dir="charts"):
    os.makedirs(output_dir, exist_ok=True)

    domain_names = [f"blocksworld_mystery_{mystery_num}"]
    dataset_types = [f"mystery-{mystery_num}-24k"]
    domain_phrases = DOMAIN_PHRASES[f"mystery_{mystery_num}"]
    action_phrases = list(domain_phrases["actions"].values())
    predicate_phrases = list(domain_phrases["predicates"].values())
    phrases = action_phrases

    # Load datasets
    eval_results, datasets, label_datasets = load_datasets(domain_names, dataset_types)
    label_positions = [None for _ in datasets]
    correct_ids = [collect_correct_ids(er) for er in eval_results]
    steps = list(range(1000, 10000, 200))

    repr_files = {
        "clean_2k": f"multilayer_representations/clean_2k/mean_reprs_clean.json",
        "clean_3k": f"multilayer_representations/clean_3k/mean_reprs_clean.json",
        "clean_4k": f"multilayer_representations/clean_4k/mean_reprs_clean.json"
    }

    hidden_states = collect_hidden_states(correct_ids[0], datasets[0], LAYERS)
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

    reversed_phrases = {v: k for k, v in domain_phrases["actions"].items()}
    reversed_phrases.update({v: k for k, v in domain_phrases["predicates"].items()})
    original_phrases = {
        "attack": "pick up",
        "succumb": "put down",
        "overcome": "stack",
        "feast": "unstack",
        "planet": "on table",
        "province": "clear",
        "harmony": "hand empty",
        "craves": "on",
        "pain": "holding"
    }
    unique_phrases = phrases
    # _reprs = reprs[selected_layer]
    # _avg_reprs = avg_reprs[selected_layer]

    # fig = make_subplots(rows=len(unique_phrases), cols=1, 
    #                     subplot_titles=[f"Similarities with Average Representation of '{map_to_original[phrase]}'" 
    #                                   for phrase in unique_phrases])
    # for i, base_phrase in enumerate(unique_phrases):
    #     base_avg_vector = np.array(_avg_reprs[base_phrase]) - np.array(avg_means[selected_layer]["mean_domain"])
    #     similarities = {phrase: [] for phrase in unique_phrases}
    #     for compare_phrase in unique_phrases:
    #         for step in steps:
    #             repr_key = f"{compare_phrase}_{step}"
    #             if repr_key in _reprs:
    #                 repr_vector = _reprs[repr_key] - mean_reprs[selected_layer]
    #                 similarity = 1 - cosine(repr_vector, base_avg_vector)
    #                 similarities[compare_phrase].append(similarity)
    #     legend_entries = set()
    #     for phrase in unique_phrases:
    #         is_base = phrase == base_phrase
    #         is_action = phrase in action_phrases
    #         is_base_action = base_phrase in action_phrases
    #         if is_base:
    #             line_style = 'solid'
    #             line_width = 3
    #             line_color = 'black'
    #             legend_name = 'Current Phrase'
    #         elif is_action == is_base_action:
    #             line_style = 'solid'
    #             line_width = 1
    #             line_color = 'rgba(128, 128, 128, 0.6)'
    #             legend_name = 'Same-type Phrases'
    #         else:
    #             line_style = 'dash'
    #             line_width = 1
    #             line_color = 'rgba(128, 128, 128, 0.6)'
    #             legend_name = 'Different-type Phrases'
    #         show_legend = (i == 0) and (legend_name not in legend_entries)
    #         if show_legend:
    #             legend_entries.add(legend_name)
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=steps,
    #                 y=similarities[phrase],
    #                 name=legend_name,
    #                 line=dict(
    #                     color=line_color,
    #                     width=line_width,
    #                     dash=line_style
    #                 ),
    #                 showlegend=show_legend
    #             ),
    #             row=i+1,
    #             col=1
    #         )
    # fig.update_layout(
    #     height=300*len(unique_phrases),
    #     title_text=f"Cosine Similarities with Average Representations (Layer {selected_layer})",
    #     showlegend=True
    # )
    # for i in range(len(unique_phrases)):
    #     fig.update_yaxes(title_text="Cosine Similarity", row=i+1, col=1)
    # fig.update_xaxes(title_text="Steps", row=len(unique_phrases), col=1)
    # fig.write_image(os.path.join(output_dir, f"phrase_similarities_layer_{selected_layer}.png"), scale=2)

    for repr_type, repr_file in repr_files.items():
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

        for layer in LAYERS:
            _reprs = reprs[layer]
            _avg_reprs = avg_reprs[layer]
            mean_domain = np.array(avg_means[layer]["mean_domain"])
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
                for phrase in unique_phrases:
                    is_action = phrase in action_phrases
                    is_predicate = phrase in predicate_phrases

                    original_phrase = original_phrases[reversed_phrases[phrase]]

                    repr_key = f"{phrase}_{step}"
                    if repr_key not in _reprs:
                        continue
                    repr_vector = _reprs[repr_key] - mean_reprs[layer]
                    base_avg_vector = np.array(_avg_reprs[original_phrase]) - mean_domain
                    own_sim = 1 - cosine(repr_vector, base_avg_vector)
                    if is_action:
                        action_own.append(own_sim)
                    else:
                        pred_own.append(own_sim)
                    if is_action:
                        other_actions = [p for p in action_phrases if p != phrase]
                        sims = []
                        for other in other_actions:
                            other_original_phrase = original_phrases[reversed_phrases[other]]
                            other_avg = np.array(_avg_reprs[other_original_phrase]) - mean_domain
                            sims.append(1 - cosine(repr_vector, other_avg))
                        if sims:
                            action_other_action.append(np.mean(sims))
                        # sims = []
                        # for other in predicate_phrases:
                        #     other_avg = np.array(_avg_reprs[original_phrases[other]]) - np.array(mean_domain)
                        #     sims.append(1 - cosine(repr_vector, other_avg))
                        # if sims:
                        #     action_pred.append(np.mean(sims))
                    else:
                        other_preds = [p for p in predicate_phrases if p != phrase]
                        sims = []
                        for other in other_preds:
                            other_avg = np.array(_avg_reprs[original_phrase]) - mean_domain
                            sims.append(1 - cosine(repr_vector, other_avg))
                        if sims:
                            pred_other_pred.append(np.mean(sims))
                        sims = []
                        for other in action_phrases:
                            other_avg = np.array(_avg_reprs[original_phrase]) - mean_domain
                            sims.append(1 - cosine(repr_vector, other_avg))
                        if sims:
                            pred_action.append(np.mean(sims))
                action_own_sims.append(np.mean(action_own) if action_own else np.nan)
                pred_own_sims.append(np.mean(pred_own) if pred_own else np.nan)
                action_other_action_sims.append(np.mean(action_other_action) if action_other_action else np.nan)
                pred_other_pred_sims.append(np.mean(pred_other_pred) if pred_other_pred else np.nan)
                action_pred_sims.append(np.mean(action_pred) if action_pred else np.nan)
                pred_action_sims.append(np.mean(pred_action) if pred_action else np.nan)
        # fig2 = go.Figure()
        # fig2.add_trace(go.Scatter(x=steps, y=action_own_sims, name="Actions - Own Avg"))
        # fig2.add_trace(go.Scatter(x=steps, y=pred_own_sims, name="Predicates - Own Avg"))
        # fig2.add_trace(go.Scatter(x=steps, y=action_other_action_sims, name="Actions - Other Actions"))
        # fig2.add_trace(go.Scatter(x=steps, y=pred_other_pred_sims, name="Predicates - Other Predicates"))
        # fig2.add_trace(go.Scatter(x=steps, y=action_pred_sims, name="Actions - Predicates"))
        # fig2.add_trace(go.Scatter(x=steps, y=pred_action_sims, name="Predicates - Actions"))
        # fig2.update_layout(
        #     title=f"Layer {layer} - Average Similarities",
        #     xaxis_title="Steps",
        #     yaxis_title="Average Cosine Similarity"
        # )
        # fig2.write_image(os.path.join(output_dir, f"avg_similarities_layer_{layer}.png"), scale=2)

            df = pd.DataFrame({
                "step": steps,
                "action_own": action_own_sims,
                # "pred_own": pred_own_sims,
                "action_other_action": action_other_action_sims,
                # "pred_other_pred": pred_other_pred_sims,
                # "action_pred": action_pred_sims,
                # "pred_action": pred_action_sims,
            })
            df.to_csv(os.path.join(output_dir, f"avg_similarities_layer_{layer}_{repr_type}_nemo_traces.csv"), index=False)

if __name__ == "__main__":
    for mystery_num in trange(1, 16):
        out_dir = os.path.join("charts_data", "clean", f"mystery_{mystery_num}")
        process_domain(mystery_num, output_dir=out_dir)





