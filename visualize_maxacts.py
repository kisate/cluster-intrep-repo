import gradio as gr
import numpy as np
from pathlib import Path
from utils import initialize_tokenizer
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

tokenizer = initialize_tokenizer(model_id)

data_path = Path("saved_itda")

# feature_acts = np.load(data_path / "feature_acts.npz")
tokens = np.load(data_path / "tokens_128.npz")["arr_0"]
feature_acts = np.load(data_path / "feature_acts_128.npz")
combined_weights = np.load(data_path / "weights_128.npz")["arr_0"]
combined_indices = np.load(data_path / "indices_128.npz")["arr_0"]

print(feature_acts.keys())

dropdown = gr.Dropdown([f"{i}" for i in feature_acts])

def normalize_weights(weights):
    min_weight = -1
    max_weight = 3    


    # print(weights)

    weights = np.clip(weights, min_weight, max_weight)
    
    pos_mask = weights > 0
    new_weights = np.zeros_like(weights)

    new_weights[pos_mask] = weights[pos_mask] / max_weight
    new_weights[~pos_mask] = -weights[~pos_mask] / min_weight

    return new_weights

# Display feature activation around token
def display_features(feature_index, window_size=20, topk=500):
    offsets, weights = feature_acts[feature_index]

    _tokens = []
    _weights = []

    feature_index = int(feature_index)
    
    for ofs, w in zip(offsets[:topk], weights[:topk]):
        ofs = int(ofs)
        start = max(0, ofs - window_size)
        end = min(len(tokens), ofs + window_size + 1)
        _tokens.append([tokenizer.decode(t) for t in tokens[start:end]])

        weights_window = combined_weights[start:end]
        indices_window = combined_indices[start:end]

        _indices = (weights_window != 0) & (indices_window == feature_index)

        _feature_present_mask = _indices.any(axis=1)
        weights_window = weights_window * _indices
        weights_window = weights_window.sum(axis=1)

        _token_weights = np.zeros_like(tokens[start:end], dtype=float)
        _token_weights[_feature_present_mask] = weights_window[_feature_present_mask]

        _weights.append(_token_weights)

    return gr.HTML(generate_colored_html(_tokens, _weights))

def generate_colored_html(tokens, weights):
    html = """
    <div style='
        display: block;
        overflow-x: auto;
        white-space: nowrap;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        max-width: 100%;
        min-height: 50px;
    '>
    """
    
    for _tokens, _weights in zip(tokens, weights):
        _weights = normalize_weights(_weights)
        html += "<div style='display: flex; align-items: center; gap: 2px; margin-bottom: 5px;'>"
        
        for token, weight in zip(_tokens, _weights):
            alpha = abs(weight)
            color = f"rgba(0, 200, 0, {alpha:.2f})" if weight >= 0 else f"rgba(200, 0, 0, {alpha:.2f})"
            
            html += f"""
            <span style='
                background-color: {color};
                padding: 0px 0px;
                border-radius: 0px;
                display: inline-block;
                min-height: 20px;
                line-height: 20px;
                text-align: center;
            '>
                {token}
            </span>
            """

        html += "</div>"

    html += "</div>"
    return html




iface = gr.Interface(
    fn=display_features,
    inputs=[
        dropdown,
        # gr.Slider(1, 10, 1, 5, label="Window Size"),
        # gr.Slider(1, 50, 1, 20, label="Top-K")
    ],
    outputs="html",
    live=True
)


iface.launch()