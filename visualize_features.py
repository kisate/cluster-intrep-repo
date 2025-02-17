import gradio as gr
import numpy as np
from pathlib import Path
from utils import initialize_tokenizer
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

tokenizer = initialize_tokenizer(model_id)

data_path = Path("saved_itda")

# feature_acts = np.load(data_path / "feature_acts.npz")
indices = np.load(data_path / "indices_128.npz")["arr_0"]
weights = np.load(data_path / "weights_128.npz")["arr_0"]
tokens = np.load(data_path / "tokens_128.npz")["arr_0"]

# Display non-zero features in the selected token range
def update_features_dropdown(token_offset_start, token_offset_end):
    _indices = indices[token_offset_start:token_offset_end]
    _weights = weights[token_offset_start:token_offset_end]
    
    non_zero = _indices[_weights > 1]
    feature_indices = np.unique(non_zero)   

    return gr.Dropdown([f"{i}" for i in feature_indices])


# Clip to [-5, 100] and normalize to [-1, 1]
def normalize_weights(weights):
    min_weight = -1
    max_weight = 3    
    weights = np.clip(weights, min_weight, max_weight)
    
    pos_mask = weights > 0
    new_weights = np.zeros_like(weights)

    new_weights[pos_mask] = weights[pos_mask] / max_weight
    new_weights[~pos_mask] = -weights[~pos_mask] / min_weight

    return new_weights

def generate_colored_html(tokens, weights):
    weights = normalize_weights(weights)
    html = "<div style='display: flex; flex-wrap: wrap; gap: 5px;'>"
    for token, weight in zip(tokens, weights):
        alpha = abs(weight)
        color = f"rgba(0, 200, 0, {alpha:.2f})"
        if weight < 0:
            color = f"rgba(200, 0, 0, {alpha:.2f})"

        html += f"<span style='background-color: {color}; min-height: 30px'>{token}</span> "
    html += "</div>"
    return html

def display_features(feature_index, token_offset_start, token_offset_end):
    _indices = indices[token_offset_start:token_offset_end]
    _weights = weights[token_offset_start:token_offset_end]

    _tokens = tokens[token_offset_start:token_offset_end] 
    token_weights = np.zeros_like(_tokens, dtype=float)

    feature_index = int(feature_index)
    feature_mask = _indices == feature_index

    _weights = _weights * feature_mask
    _weights = _weights.sum(axis=1)

    token_mask = feature_mask.any(axis=1)
    token_weights[token_mask] = _weights[token_mask]

    _tokens = [tokenizer.decode([t]) for t in _tokens]

    return gr.HTML(generate_colored_html(_tokens, token_weights))
    

with gr.Blocks() as demo:
    token_offset_start = gr.Number(minimum=0, maximum=len(tokens) - 1, label="Token Offset Start")
    token_offset_end = gr.Number(minimum=0, maximum=len(tokens) - 1, label="Token Offset End")

    features_dropdown = gr.Dropdown(choices=[], label="Select a Feature")

    html = gr.HTML()

    features_dropdown.change(
        fn=display_features, 
        inputs=[features_dropdown, token_offset_start, token_offset_end],
        outputs=[html]
    )

    token_offset_start.change(
        fn=update_features_dropdown,
        inputs=[token_offset_start, token_offset_end],
        outputs=[features_dropdown]
    )

    token_offset_end.change(
        fn=update_features_dropdown,
        inputs=[token_offset_start, token_offset_end],
        outputs=[features_dropdown]
    )
    

demo.launch()
    


