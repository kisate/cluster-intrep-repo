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
# MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_type = "qwq"
model_name = "qwq-32b"
# model_name = "llama-3_3-nemotron-super-49b-v1"
# model_name = "deepseek-qwen-32b"
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
    prompt_dir = CUR_DIR / Path(f"./cot-planning/results/{domain_name}/{model_name}-greedy")
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
        # load_dataset(f"dmitriihook/{model_name}-planning-{dataset_type}-greedy")["train"]
        # load_dataset(f"dmitriihook/llama-3_3-nemotron-super-49b-v1-planning-{dataset_type}-greedy")["train"]
        load_dataset(f"dmitriihook/deepseek-qwen-32b-planning-{dataset_type}-greedy")["train"]
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

def make_phrase_tokens(phrase):
    if "deepseek" in MODEL_ID.lower():
        return [
            tokenizer.encode(" " + phrase, add_special_tokens=False),
            tokenizer.encode(" " + phrase.capitalize(), add_special_tokens=False),
            tokenizer.encode("\n" + phrase, add_special_tokens=False)[1:],
            tokenizer.encode("\n" + phrase.capitalize(), add_special_tokens=False)[1:],
            tokenizer.encode("\n\n" + phrase, add_special_tokens=False)[1:],
            tokenizer.encode("\n\n" + phrase.capitalize(), add_special_tokens=False)[1:],
        ]
    else:
        return [
            tokenizer.encode(" " + phrase),
            tokenizer.encode(" " + phrase.capitalize()),
            tokenizer.encode("\n" + phrase)[1:],
            tokenizer.encode("\n" + phrase.capitalize())[1:],
            tokenizer.encode("\n\n" + phrase)[1:],
            tokenizer.encode("\n\n" + phrase.capitalize())[1:],
        ]


def extract_all_phrase_positions(tokens, phrase, cot_only=False):
    """Find all phrase positions in the tokens"""
    tokens = tokens.squeeze()

    

    phrase_tokens = make_phrase_tokens(phrase)

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
    return list(range(N_ROWS))
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

def plot_pca(reprs, phrases, title, output_dir):
    """Plot PCA of phrase representations"""
    valid_phrases = [p for p in phrases if p in reprs and reprs[p] is not None]
    if not valid_phrases:
        print(f"No valid phrases found for PCA plot: {title}")
        return
    
    # Extract domain from title
    domain_name = None
    for domain in DOMAIN_PHRASES:
        if domain in title.lower():
            domain_name = domain
            break
    
    # Get mapping of phrases to actions/predicates for better labeling
    phrase_to_concept = {}
    if domain_name:
        domain_phrases = DOMAIN_PHRASES[domain_name]
        # Combine actions and predicates
        for concept_type in ["actions", "predicates"]:
            for concept, phrase in domain_phrases[concept_type].items():
                phrase_to_concept[phrase] = f"{concept_type[:-1]}:{concept}"
    
    X = np.stack([reprs[p] for p in valid_phrases], axis=0)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = go.Figure()

    for i, phrase in enumerate(valid_phrases):
        # Use concept name as label if available
        display_text = phrase_to_concept.get(phrase, phrase) if domain_name else phrase
        
        fig.add_trace(go.Scatter(
            x=[X_pca[i, 0]],
            y=[X_pca[i, 1]],
            mode='markers+text',
            name=phrase,
            text=[display_text],
            textposition="top center",
            marker=dict(size=10)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=800,
        height=600
    )
    
    # Save the figure
    filename = f"{title.replace(' ', '_').lower()}.png"
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, filename))
    
    return fig

def plot_combined_pca(reprs_clean, phrases_clean, reprs_mystery, phrases_mystery, title, output_dir):
    """Plot combined PCA of clean and mystery phrases"""
    valid_phrases_clean = [p for p in phrases_clean if p in reprs_clean and reprs_clean[p] is not None]
    valid_phrases_mystery = [p for p in phrases_mystery if p in reprs_mystery and reprs_mystery[p] is not None]
    
    if not valid_phrases_clean or not valid_phrases_mystery:
        print(f"Not enough valid phrases for combined PCA: {title}")
        return
    
    # Extract domain names from title
    mystery_domain = None
    for domain in DOMAIN_PHRASES:
        if domain in title.lower():
            mystery_domain = domain
            break
    
    # Create mappings from phrases to their conceptual meanings
    clean_phrase_to_concept = {}
    mystery_phrase_to_concept = {}
    
    # Set up clean domain phrase mapping
    clean_domain = "mystery_2"
    if clean_domain in DOMAIN_PHRASES:
        for concept_type in ["actions", "predicates"]:
            for concept, phrase in DOMAIN_PHRASES[clean_domain][concept_type].items():
                clean_phrase_to_concept[phrase] = f"{concept}"
    
    # Set up mystery domain phrase mapping
    if mystery_domain in DOMAIN_PHRASES:
        for concept_type in ["actions", "predicates"]:
            for concept, phrase in DOMAIN_PHRASES[mystery_domain][concept_type].items():
                mystery_phrase_to_concept[phrase] = f"{concept}"
    
    X_1 = np.stack([reprs_clean[p] for p in valid_phrases_clean], axis=0)
    X_1 = X_1 - X_1.mean(0)

    X_2 = np.stack([reprs_mystery[p] for p in valid_phrases_mystery], axis=0)
    X_2 = X_2 - X_2.mean(0)

    X = np.concatenate([X_1, X_2], axis=0)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = go.Figure()

    # Add clean representations
    for i, phrase in enumerate(valid_phrases_clean):
        # Use concept name if available
        display_text = f"{clean_phrase_to_concept.get(phrase, phrase)} (clean)"
        
        fig.add_trace(go.Scatter(
            x=[X_pca[i, 0]],
            y=[X_pca[i, 1]],
            mode='markers+text',
            name=phrase + " (clean)",
            text=[display_text],
            textposition="top center",
            marker=dict(size=10, color='blue')
        ))
    
    # Add mystery representations
    n_clean = len(valid_phrases_clean)
    for i, phrase in enumerate(valid_phrases_mystery):
        # Use concept name if available
        display_text = f"{mystery_phrase_to_concept.get(phrase, phrase)} (mystery)"
        
        fig.add_trace(go.Scatter(
            x=[X_pca[n_clean + i, 0]],
            y=[X_pca[n_clean + i, 1]],
            mode='markers+text',
            name=phrase + " (mystery)",
            text=[display_text],
            textposition="top center",
            marker=dict(size=10, color='red')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=1000,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save the figure
    filename = f"{title.replace(' ', '_').lower()}.png"
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, filename))
    
    return fig

def plot_phrases_comparison(reprs, action_phrases, predicate_phrases, clean_reprs, mystery_domain_means, clean_domain_means, output_dir, domain_name):
    """Plot comparison of phrase representations"""
    from plotly.colors import qualitative
    colors = qualitative.Plotly
    
    # Get domain-specific clean phrases
    clean_domain_phrases = DOMAIN_PHRASES["mystery_2"]
    clean_action_names = list(clean_domain_phrases["actions"].keys())  # These are the general action names
    clean_predicate_names = list(clean_domain_phrases["predicates"].keys())  # These are the general predicate names
    
    clean_phrases = clean_action_names + clean_predicate_names
    
    phrases = action_phrases + predicate_phrases
    
    all_phrases = list(clean_reprs.keys())
    
    # Create color map with meaningful action names
    color_map = {phrase: colors[i % len(colors)] for i, phrase in enumerate(all_phrases)}

    # Create subplots
    fig = make_subplots(rows=len(phrases), cols=1)

    for j, action in enumerate(phrases):
        action_reprs = []

        for key in reprs:
            if action in key:
                action_reprs.append(reprs[key] - mystery_domain_means[action])

        if not action_reprs:
            print(f"No representations found for action {action} in domain {domain_name}")
            continue
            
        action_reprs = np.stack(action_reprs)
        
        action_reprs = action_reprs / np.linalg.norm(action_reprs, axis=1, keepdims=True)

        csims_action = []
        for phrase in all_phrases:
            if clean_reprs[phrase] is not None:
                clean_repr = clean_reprs[phrase] - clean_domain_means[phrase]
                clean_repr = clean_repr / np.linalg.norm(clean_repr)
                csim = action_reprs @ clean_repr
                csims_action.append(csim)

        if not csims_action:
            continue
            
        csims_action = np.stack(csims_action, axis=0)

        for i, phrase in enumerate(all_phrases):
            if i < len(csims_action):
                fig.add_trace(go.Scatter(
                    x=np.arange(csims_action.shape[1]),
                    y=csims_action[i],
                    mode='lines',
                    name=phrase,
                    line=dict(color=color_map[phrase]),
                    showlegend=(j == 0)  # Show legend only for the first subplot
                ), row=j + 1, col=1)

    # Update layout for better viewing
    fig.update_layout(
        height=200 * len(phrases),
        title_text=f"Action Representations Comparison - {domain_name}",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add y-axis titles
    for j, action in enumerate(phrases):
        fig.update_yaxes(title_text=f"{action} ({clean_phrases[j]})", row=j+1, col=1)
    
    # Save the figure
    filename = f"action_comparison_{domain_name}.png"
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, filename))
    
    return fig

def save_representations(mean_reprs, domain_name, output_dir):
    """Save representations to a JSON file"""

    data_to_save = {}

    domain_phrases = DOMAIN_PHRASES[domain_name]
    action_phrases = list(domain_phrases["actions"].values())
    predicate_phrases= list(domain_phrases["predicates"].values())
    
    for layer, layer_reprs in mean_reprs.items():
        mean_domain = np.stack(list(layer_reprs.values())).mean(0).tolist()

        action_reprs = [layer_reprs[a] for a in action_phrases]
        predicate_reprs = [layer_reprs[p] for p in predicate_phrases]

        mean_actions = np.stack(action_reprs).mean(0).tolist()
        mean_predicates = np.stack(predicate_reprs).mean(0).tolist()

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
    

def plot_in_domain_similarity_heatmap(reprs, phrases, domain_name, mean_reprs, domain_means, output_dir):
    """
    Plot a heatmap of in-domain similarities with the domain means.
    
    Args:
        reprs: Dictionary of phrase representations at different steps
        phrases: List of phrases to include in the analysis
        domain_name: Name of the domain for the title and filename
        steps: List of steps used for representation extraction
        output_dir: Directory to save the output image
    """
    import plotly.express as px
    import numpy as np
    import os
        
    # Calculate relative representations (subtract domain mean)
    repr_diffs = {p: mean_reprs[p] - domain_means[p] for p in phrases}
    repr_diffs = {p: v / np.linalg.norm(v) for p, v in repr_diffs.items() if v is not None}
    
    all_keys = list(reprs.keys())
    all_reprs = np.stack([reprs[k] - domain_means[k.split("_")[0]] for k in all_keys], dtype=np.float32)
    
    # Normalize
    all_reprs = all_reprs / np.linalg.norm(all_reprs, axis=1, keepdims=True)
    
    csims = [
        all_reprs @ r for r in repr_diffs.values()
    ]
    
    # Stack similarities
    csim_matrix = np.stack(csims, axis=0)
    
    # Generate more human-readable x-axis labels
    x_labels = []
    for key in all_keys:
        # Split by underscore and get phrase and step
        parts = key.split('_')
        if len(parts) >= 2:
            phrase = '_'.join(parts[:-1])
            step = parts[-1]
            x_labels.append(f"{phrase}_{step}")
        else:
            x_labels.append(key)
    
    # Create heatmap
    fig = px.imshow(
        csim_matrix,
        x=x_labels,
        y=list(repr_diffs.keys()),
        labels=dict(x="Step Representations", y="Mean Representations", color="Cosine Similarity"),
        title=f"In-Domain Similarity Heatmap - {domain_name}",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    # Update layout for better viewing
    fig.update_layout(
        width=max(1000, len(all_keys) * 30),  # Scale width based on number of columns
        height=max(600, len(phrases) * 50),  # Scale height based on number of rows
        xaxis=dict(tickangle=45)
    )
    
    # Save the figure
    filename = f"in_domain_similarity_{domain_name}.png"
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, filename))
    
    return fig

def plot_cross_domain_similarity_heatmap(mystery_reprs, clean_phrases, mystery_phrases, clean_mean_reprs, clean_domain_means, mystery_domain_means, domain_name, output_dir):
    """
    Plot a heatmap of cross-domain similarities between clean and mystery domain phrases.
    
    Args:
        clean_reprs: Dictionary of clean domain phrase representations
        mystery_reprs: Dictionary of mystery domain phrase representations
        clean_phrases: List of clean domain phrases
        mystery_phrases: List of mystery domain phrases
        domain_name: Name of the mystery domain for the title and filename
        output_dir: Directory to save the output image
    """
    import plotly.express as px
    import numpy as np
    import os
    
    clean_repr_diffs = {p: clean_mean_reprs[p] - clean_domain_means[p] for p in clean_phrases}
    clean_repr_diffs = {p: v / np.linalg.norm(v) for p, v in clean_repr_diffs.items()}
    
    all_keys = list(mystery_reprs.keys())
    all_reprs = np.stack([mystery_reprs[k] - mystery_domain_means[k.split("_")[0]] for k in all_keys], dtype=np.float32)
    
    # Normalize
    all_reprs = all_reprs / np.linalg.norm(all_reprs, axis=1, keepdims=True)
    
    csims = [
        all_reprs @ r for r in clean_repr_diffs.values()
    ]
    
    csim_matrix = np.stack(csims, axis=0)

    # Create heatmap
    fig = px.imshow(
        csim_matrix,
        x=all_keys,
        y=mystery_phrases,
        labels=dict(x="Clean Domain", y="Mystery Domain", color="Cosine Similarity"),
        title=f"Cross-Domain Similarity Heatmap - Clean vs {domain_name}",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    # Update layout for better viewing
    fig.update_layout(
        width=max(800, len(all_keys) * 100),
        height=max(600, len(clean_phrases) * 100)
    )
    
    # Save the figure
    filename = f"cross_domain_similarity_{domain_name}.png"
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, filename))
    
    return fig

def process_representations(reprs_mystery, mystery_phrases, mystery_action_phrases, mystery_predicate_phrases, clean_phrases, clean_action_phrases, clean_predicate_phrases, steps_mystery, output_dir, mystery_num, mean_clean_actions, mean_clean_predicates, mean_reprs_clean, domain_dir):
    """Process representations for a given domain"""
    # Make mean representations
    mean_reprs_mystery = make_mean_reprs(reprs_mystery, mystery_phrases, steps_mystery)
    
    mean_mystery_actions = {
        l: np.stack([layer_reprs[a] for a in mystery_action_phrases], axis=0).mean(0)
        for l, layer_reprs in mean_reprs_mystery.items()
    }
    mean_mystery_predicates = {
        l: np.stack([layer_reprs[p] for p in mystery_predicate_phrases], axis=0).mean(0)
        for l, layer_reprs in mean_reprs_mystery.items()
    }
    
    print(f"Plotting mystery {mystery_num} representations")

    for layer in tqdm(LAYERS):

        clean_domain_means = {
            k: mean_clean_actions[layer] if k in clean_action_phrases else mean_clean_predicates[layer]
            for k in clean_phrases
        }
        
        mystery_domain_means = {
            k: mean_mystery_actions[layer] if k in mystery_action_phrases else mean_mystery_predicates[layer]
            for k in mystery_phrases
        }


        # Plot action comparison
        plot_phrases_comparison(
            reprs_mystery[layer],
            mystery_action_phrases,
            mystery_predicate_phrases,
            mean_reprs_clean[layer],
            mystery_domain_means,
            clean_domain_means,
            domain_dir,
            f"mystery_{mystery_num}_{layer}"
        )
    
        # Plot in-domain similarity heatmap
        plot_in_domain_similarity_heatmap(
            reprs_mystery[layer], 
            mystery_phrases, 
            f"mystery_{mystery_num}_{layer}",
            mean_reprs_mystery[layer],
            mystery_domain_means,
            domain_dir
        )
        
        # Plot cross-domain similarity heatmap
        plot_cross_domain_similarity_heatmap(
            reprs_mystery[layer],
            clean_phrases,
            mystery_phrases,
            mean_reprs_clean[layer],
            clean_domain_means,
            mystery_domain_means,
            f"mystery_{mystery_num}_{layer}",
            domain_dir
        )
    
    # Save representations
    save_representations(mean_reprs_mystery, f"mystery_{mystery_num}", domain_dir)

def process_mystery_domain(mystery_num, mean_reprs_clean, mean_clean_actions, mean_clean_predicates, output_dir):
    """Process a specific mystery domain"""
    print(f"\n=== Processing Mystery {mystery_num} ===")
    
    # Set domain names and dataset types
    domain_names = [f"blocksworld_mystery_{2}", f"blocksworld_mystery_{mystery_num}"]
    dataset_types = [f"mystery-{2}-16k", f"mystery-{mystery_num}-16k"]
    
    # Get domain-specific phrases from the phrases definitions
    # This assumes the phrases are defined in the code - we'll add these definitions
    clean_domain_phrases = DOMAIN_PHRASES["mystery_2"]
    mystery_domain_phrases = DOMAIN_PHRASES[f"mystery_{mystery_num}"]
    
    # Extract action and predicate phrases
    clean_action_phrases = list(clean_domain_phrases["actions"].values())
    clean_predicate_phrases = list(clean_domain_phrases["predicates"].values())
    clean_phrases = clean_action_phrases + clean_predicate_phrases
    
    mystery_action_phrases = list(mystery_domain_phrases["actions"].values())
    mystery_predicate_phrases = list(mystery_domain_phrases["predicates"].values())
    mystery_phrases = mystery_action_phrases + mystery_predicate_phrases
    
    # Create subdirectory for this mystery domain
    domain_dir = os.path.join(output_dir, f"mystery_{mystery_num}")
    os.makedirs(domain_dir, exist_ok=True)
    
    # Load datasets
    eval_results, datasets, label_datasets = load_datasets(domain_names, dataset_types)
    
    label_positions = [
        None for _ in datasets
    ]
    
    # Collect correct IDs
    correct_ids = [
        collect_correct_ids(er) for er in eval_results
    ]

    print(len(correct_ids[1]))
    
    steps_mystery = list(range(1000, 7000, 200))
    
    # Collect hidden states for mystery domain
    hidden_states_mystery = collect_hidden_states(correct_ids[1], datasets[1], LAYERS)
    
    # Extract phrase representations
    reprs_mystery = extract_phrase_representations_batched(
        correct_ids[1][:N_ROWS], 
        datasets[1], 
        label_positions[1], 
        hidden_states_mystery, 
        LAYERS, 
        mystery_phrases, 
        steps_mystery, 
        CONT_SIZE
    )

    process_representations(reprs_mystery, mystery_phrases, mystery_action_phrases, mystery_predicate_phrases, clean_phrases, clean_action_phrases, clean_predicate_phrases, steps_mystery, output_dir, mystery_num, mean_clean_actions, mean_clean_predicates, mean_reprs_clean, domain_dir)
    

def main():
    """Main function to process all mystery domains"""
    # Initialize model and tokenizer
    initialize_model_and_tokenizer()
    
    # Create output directory
    output_dir = "./multilayer_representations_ds_ds_traces/multilayer_7k"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process mystery domain 2 as clean representations first
    print("=== Processing Mystery 2 (Clean) ===")
    
    # Set domain names and dataset types for clean domain
    domain_names = ["blocksworld_mystery_2"]
    dataset_types = ["mystery-2-16k"]
    
    # Get domain-specific phrases
    clean_domain_phrases = DOMAIN_PHRASES["mystery_2"]
    
    # Extract action and predicate phrases
    clean_action_phrases = list(clean_domain_phrases["actions"].values())
    clean_predicate_phrases = list(clean_domain_phrases["predicates"].values())
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
        steps_clean = list(range(1000, 7000, 200))
        
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

        mean_clean_actions = {
            l: np.stack([layer_reprs[a] for a in clean_action_phrases], axis=0).mean(0) 
            for l, layer_reprs in clean_reprs.items()
        }
        mean_clean_predicates = {
            l: np.stack([layer_reprs[p] for p in clean_predicate_phrases], axis=0).mean(0)
            for l, layer_reprs in clean_reprs.items()
        }
        
    
        # Save clean representations
        save_representations(clean_reprs, "mystery_2", output_dir)

        # Process other mystery domains
        for mystery_num in range(1, 16):
            process_mystery_domain(mystery_num, clean_reprs, mean_clean_actions, mean_clean_predicates, output_dir)
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        raise e

if __name__ == "__main__":
    main()