import sys
import os
import argparse
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

# Initialize Ray cluster
ray.init(address="auto", namespace="blocksworld")

compute_dtype = torch.bfloat16
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_id = "Qwen/QwQ-32B"
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

# Ray task for collecting hidden states - now takes gpus_per_model as parameter
def collect_hidden_states_ray_factory(gpus_per_model):
    @ray.remote(num_gpus=gpus_per_model)
    def collect_hidden_states_ray(row_ids: list[int], max_tokens: int, layers: list[int], n_dim: int, dataset_handle):
        # Get the GPU IDs assigned by Ray
        import os
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        
        # Set up device mapping based on number of GPUs
        if gpus_per_model == 1:
            device_map = "cuda:0"
        else:
            # Use all available GPUs with automatic mapping
            device_map = "auto"
        
        print(f"Worker using {len(gpu_ids)} GPUs: {gpu_ids}")
        
        tokenizer = initialize_tokenizer(model_id)
        # dataset = load_dataset(
        #     f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type[n_blocks]}")["train"]
        
        dataset = load_dataset(
            f"dmitriihook/qwq-32b-planning-6-blocks")["train"]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device_map, 
            torch_dtype=compute_dtype, 
            attn_implementation="sdpa"
        )

        hidden_states = defaultdict(dict)

        for idx in tqdm(row_ids):
            row = dataset[idx]
            tokens = tokenize_blocksworld_generation(tokenizer, row)[0]
            tokens = tokens[:max_tokens]

            # Determine the correct device for input tokens based on model's device map
            input_device = model.device if hasattr(model, 'device') else "cuda:0"
            
            with torch.no_grad():
                _hidden_states = model(tokens.unsqueeze(0).to(input_device), output_hidden_states=True).hidden_states
                for layer in layers:
                    hidden_states[idx][layer] = _hidden_states[layer][0].cpu().to(torch.float16).numpy()

        hidden_states = {idx: hidden_states[idx] for idx in row_ids}

        dataset_handle.put_hidden_states.remote(hidden_states)
            
        return hidden_states
    
    return collect_hidden_states_ray


def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Collect hidden states from language models")
    parser.add_argument("--gpus_per_model", type=int, default=2, 
                        help="Number of GPUs to allocate per model instance")
    parser.add_argument("--rows_start", type=int, default=0,
                        help="Starting row index")
    parser.add_argument("--rows_end", type=int, default=2000,
                        help="Ending row index (exclusive)")
    parser.add_argument("--max_tokens", type=int, default=7000,
                        help="Maximum number of tokens to process")
    parser.add_argument("--layers", type=int, nargs="+", default=[39],
                        help="Layer indices to extract hidden states from")
    
    args = parser.parse_args()
    
    # Get parameters from arguments
    gpus_per_model = args.gpus_per_model
    rows_start = args.rows_start
    rows_end = args.rows_end
    max_tokens = args.max_tokens
    layers = args.layers
    n_dim = 5120

    n_rows = rows_end - rows_start
    
    # Get available GPUs and calculate number of workers
    total_gpus = int(ray.cluster_resources().get("GPU", 1))
    n_workers = total_gpus // gpus_per_model
    
    if n_workers == 0:
        raise ValueError(f"Not enough GPUs. Requested {gpus_per_model} GPUs per model, but only {total_gpus} available.")
    
    print(f"Starting hidden state collection with {total_gpus} total GPUs")
    print(f"Using {gpus_per_model} GPUs per model across {n_workers} workers")
    
    # Create task function with specified GPU allocation
    collect_task = collect_hidden_states_ray_factory(gpus_per_model)
    
    # Divide work by available workers
    rows_per_worker = n_rows // n_workers
    row_ids = [list(range(rows_start + i * rows_per_worker, 
                         rows_start + (i + 1) * rows_per_worker if i < n_workers - 1 else rows_end)) 
             for i in range(n_workers)]

    # Launch Ray tasks
    futures = []
    
    for i in range(n_workers):
        futures.append(collect_task.remote(
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