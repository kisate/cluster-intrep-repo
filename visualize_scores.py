import gradio as gr
import json
import numpy as np

# Load JSON file
def load_json(filepath="wait_probe_preds.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)  # Expecting a list of objects with "outputs" and "tokens"
    return data

# Normalize scores to [0, 1]
def normalize_scores(scores):
    scores = [max(-1, min(1, s)) for s in scores]  # Clip to [-1, 1]
    min_score, max_score = min(scores), max(scores)
    return [(s - min_score) / (max_score - min_score) if max_score > min_score else 0.5 for s in scores]

# Generate colored HTML for tokens
def generate_colored_text(tokens, scores):
    html = "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>"
    for token, score in zip(tokens, scores):
        alpha = score  # Use score as transparency level (0 = fully transparent, 1 = fully opaque)
        alpha = alpha - 0.5
        alpha = max(0, alpha) * 2  # Scale to [0, 1]
        color = f"rgba(200, 0, 100, {alpha:.2f})"  # Red-purple with variable transparency
        html += f"<span style='background-color: {color}; padding: 5px; border-radius: 5px;'>{token}</span> "
    html += "</div>"
    return html

# Update display based on selected instance
def display_tokens(instance_index):
    data = load_json()  # Load fresh data
    instance = data[int(instance_index)]
    tokens = instance["tokens"]
    scores = instance["outputs"]
    
    normalized_scores = normalize_scores(scores)
    return generate_colored_text(tokens, normalized_scores)

# Load data and prepare instance selector
data = load_json()
instance_options = [f"{i}" for i in range(len(data))]

# Gradio Interface
iface = gr.Interface(
    fn=display_tokens,
    inputs=gr.Dropdown(choices=instance_options, label="Select an Instance", value="Instance 0"),
    outputs="html",
    live=True
)

iface.launch()
