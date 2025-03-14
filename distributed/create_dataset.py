import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import ray
import time
from transformers import AutoModelForCausalLM
from utils import initialize_tokenizer, tokenize_blocksworld_generation

from stacks_utils import *
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

# Removed vLLM import

# Initialize Ray cluster
ray.init(address="auto", namespace="blocksworld")

compute_dtype = torch.bfloat16
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
n_blocks = 6
dataset_actor_name = "dataset_actor"
blocksworld_type = {
    4: "4-blocks",
    6: "6-blocks-big",
}

@ray.remote
class HiddenStatesDataset:
    def __init__(self):
        self.hidden_states = defaultdict(dict)

    def get_hidden_states(self):
        return self.hidden_states

    def put_hidden_states(self, hidden_states: dict):
        for idx in hidden_states:
            for layer in hidden_states[idx]:
                self.hidden_states[idx][layer] = ray.put(hidden_states[idx][layer])

dataset_handle = HiddenStatesDataset.options(name=dataset_actor_name).remote()

# Ray task for collecting hidden states
@ray.remote(num_gpus=1)
def collect_hidden_states_ray(row_ids: list[int], max_tokens: int, layers: list[int], n_dim: int, dataset_handle):
    # Get the GPU ID assigned by Ray
    import os
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    device = f"cuda:0"  # With Ray's GPU isolation, this will be the correct device
    
    tokenizer = initialize_tokenizer(model_id)
    dataset = load_dataset(
        f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type[n_blocks]}")["train"]

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=device, 
        torch_dtype=compute_dtype, 
        attn_implementation="sdpa"
    )

    hidden_states = defaultdict(dict)

    for idx in tqdm(row_ids):
        row = dataset[idx]
        tokens = tokenize_blocksworld_generation(tokenizer, row)[0]
        tokens = tokens[:max_tokens]

        with torch.no_grad():
            _hidden_states = model(tokens.unsqueeze(0).to(device), output_hidden_states=True).hidden_states
            for layer in layers:
                hidden_states[idx][layer] = _hidden_states[layer][0].cpu().to(torch.float16).numpy()

    hidden_states = {idx: hidden_states[idx] for idx in row_ids}

    dataset_handle.put_hidden_states.remote(hidden_states)
        
    return hidden_states


def main():
    n_gpus = int(ray.cluster_resources().get("GPU", 1))
    rows_start = 0
    rows_end = 5000
    n_dim = 5120
    max_tokens = 4000
    layers = [63]

    n_rows = rows_end - rows_start
    
    # Divide work by available GPUs
    rows_per_gpu = n_rows // n_gpus
    row_ids = [list(range(rows_start + i * rows_per_gpu, 
                          rows_start + (i + 1) * rows_per_gpu if i < n_gpus - 1 else rows_end)) 
              for i in range(n_gpus)]

    print(f"Starting hidden state collection with {n_gpus} GPUs")
    
    # Launch Ray tasks
    futures = []
    
    for i in range(n_gpus):
        futures.append(collect_hidden_states_ray.remote(
            row_ids[i], max_tokens, layers, n_dim, dataset_handle
        ))
    
    # Wait for all tasks to complete and gather results
    print("Waiting for tasks to complete...")
    results = ray.get(futures)
    
    # Combine results
    hidden_states = {}
    for result in results:
        hidden_states.update(result)
    
if __name__ == "__main__":
    print("Collecting hidden states")
    main()

    print("Dataset is ready and registered. Keeping the script alive...")
    print("Press Ctrl+C to exit.")
    try:
        # Keep the script running so the Ray actor remains available
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        ray.shutdown()