import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict

import torch
import ray
from transformers import AutoTokenizer, AutoModelForCausalLM, CompileConfig
from tqdm import tqdm, trange
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import initialize_tokenizer, tokenize_blocksworld_generation, THINK_TOKEN, THINK_START_TOKEN

# Initialize Ray cluster
ray.init(address="auto", namespace="blocksworld_steering")

compute_dtype = torch.bfloat16
model_id = "Qwen/QwQ-32B"  # Model ID for generation
device = 'cuda'  # Default device

# Phrases to steer with
DOMAIN_PHRASES = {
    "mystery_2": {
        "actions": {
            "attack": "illuminate",
            "succumb": "silence",
            "overcome": "distill",
            "feast": "divest"
        },
        "predicates": {
            "planet": "aura",
            "province": "essence",
            "harmony": "nexus",
            "craves": "harmonizes",
            "pain": "pulse"
        }
    },
    "mystery_3": {
        "actions": {
            "attack": "tltezi",
            "succumb": "jchntg",
            "overcome": "deesdu",
            "feast": "xavirm"
        },
        "predicates": {
            "planet": "oxtslo",
            "province": "adohre",
            "harmony": "jqlyol",
            "craves": "gszswg",
            "pain": "ivbmyg"
        }
    },
    "mystery_4": {
        "actions": {
            "attack": "swim",
            "succumb": "fire",
            "overcome": "deduct",
            "feast": "respond"
        },
        "predicates": {
            "planet": "fever",
            "province": "marble",
            "harmony": "craving",
            "craves": "mines",
            "pain": "shadow"
        }
    },
    "mystery_5": {
        "actions": {
            "attack": "whisper",
            "succumb": "calculate",
            "overcome": "orbit",
            "feast": "navigate"
        },
        "predicates": {
            "planet": "crystal",
            "province": "fountain",
            "harmony": "autumn",
            "craves": "illuminates",
            "pain": "legend"
        }
    },
    "mystery_6": {
        "actions": {
            "attack": "decode",
            "succumb": "hibernate",
            "overcome": "thunder",
            "feast": "quench"
        },
        "predicates": {
            "planet": "prism",
            "province": "hollow",
            "harmony": "zenith",
            "craves": "echoes",
            "pain": "emblem"
        }
    },
    "mystery_7": {
        "actions": {
            "attack": "explore",
            "succumb": "ripen",
            "overcome": "weave",
            "feast": "bloom"
        },
        "predicates": {
            "planet": "fossil",
            "province": "dialect",
            "harmony": "equinox",
            "craves": "fractures",
            "pain": "symphony"
        }
    },
    "mystery_8": {
        "actions": {
            "attack": "harvest",
            "succumb": "ignite",
            "overcome": "carve",
            "feast": "suspend"
        },
        "predicates": {
            "planet": "nebula",
            "province": "labyrinth",
            "harmony": "mirage",
            "craves": "captivates",
            "pain": "cascade"
        }
    },
    "mystery_9": {
        "actions": {
            "attack": "construct",
            "succumb": "demolish",
            "overcome": "reinforce",
            "feast": "collapse"
        },
        "predicates": {
            "planet": "eclipse",
            "province": "vintage",
            "harmony": "paradox",
            "craves": "resonates",
            "pain": "twilight"
        }
    },
    "mystery_10": {
        "actions": {
            "attack": "plant",
            "succumb": "harvest",
            "overcome": "nurture",
            "feast": "prune"
        },
        "predicates": {
            "planet": "crystal",
            "province": "puzzle",
            "harmony": "vortex",
            "craves": "whispers",
            "pain": "cipher"
        }
    },
    "mystery_11": {
        "actions": {
            "attack": "prosecute",
            "succumb": "acquit",
            "overcome": "testify",
            "feast": "appeal"
        },
        "predicates": {
            "planet": "nebula",
            "province": "molecule",
            "harmony": "anthem",
            "craves": "silhouettes",
            "pain": "voltage"
        }
    },
    "mystery_12": {
        "actions": {
            "attack": "broadcast",
            "succumb": "receive",
            "overcome": "encrypt",
            "feast": "decode"
        },
        "predicates": {
            "planet": "horizon",
            "province": "compass",
            "harmony": "solstice",
            "craves": "orbits",
            "pain": "quantum"
        }
    },
    "mystery_13": {
        "actions": {
            "attack": "whisper",
            "succumb": "banish",
            "overcome": "entangle",
            "feast": "unmask"
        },
        "predicates": {
            "planet": "tethered",
            "province": "unburdened",
            "harmony": "hollow",
            "craves": "shrouds",
            "pain": "consuming"
        }
    },
    "mystery_14": {
        "actions": {
            "attack": "question",
            "succumb": "resolve",
            "overcome": "interweave",
            "feast": "liberate"
        },
        "predicates": {
            "planet": "echoing",
            "province": "sovereign",
            "harmony": "potential",
            "craves": "obscures",
            "pain": "contemplating"
        }
    },
    "mystery_15": {
        "actions": {
            "attack": "summon",
            "succumb": "dismiss",
            "overcome": "fold",
            "feast": "unravel"
        },
        "predicates": {
            "planet": "suspended",
            "province": "timeless",
            "harmony": "interval",
            "craves": "transcends",
            "pain": "enveloping"
        }
    },
    "mystery_16": {
        "actions": {
            "attack": "illuminate",
            "succumb": "silence",
            "overcome": "distill",
            "feast": "divest"
        },
        "predicates": {
            "planet": "aura",
            "province": "essence",
            "harmony": "nexus",
            "craves": "harmonizes",
            "pain": "pulse"
        }
    }
}
# Load representation data from file
def load_representations(file_path):
    with open(file_path, 'r') as f:
        return {
            k: np.array(v) for k, v in json.load(f).items()
        }

# Extract phrase positions from tokens
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
        start_pos = torch.where(tokens == 151667)[0]  # THINK_TOKEN ID
        if len(start_pos) > 0:
            start_mask = torch.arange(tokens.shape[0]) >= start_pos[0]
        else:
            start_mask = torch.ones_like(tokens).bool()

    for phts in phrase_tokens:
        presence_mask = torch.ones_like(tokens).bool()
        if cot_only:
            presence_mask = presence_mask & start_mask

        for i, t in enumerate(phts):
            shifted_tokens = tokens[i:] if i > 0 else tokens
            curr_mask = (shifted_tokens == t)
            if i > 0:
                # Shift the mask back to align with original tokens
                padded_mask = torch.zeros_like(tokens).bool()
                padded_mask[i:] = curr_mask
                curr_mask = padded_mask
            
            presence_mask = presence_mask & curr_mask

        position_indices = torch.where(presence_mask)[0].tolist()
        for p in position_indices:
            positions.add(
                tuple([p, p + len(phts)])
            )
    
    return sorted(list(positions))

# Ray remote actor to store and share representations
@ray.remote
class RepresentationStore:
    def __init__(self):
        self.representations = {}
    
    def set_representations(self, domain, representations):
        self.representations[domain] = representations
        
    def get_representations(self, domain):
        return self.representations.get(domain, {})

# Function to save intermediate results to a file
def save_intermediate_results(results, output_dir, worker_id):
    """Save intermediate results to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"intermediate_results_worker_{worker_id}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return output_file

def load_intermediate_results(output_dir, worker_id):
    """Load intermediate results from a JSON file"""
    output_file = os.path.join(output_dir, f"intermediate_results_worker_{worker_id}.json")
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    else:
        return []

# Ray task for steered generation with batching
def steered_generation_ray_factory(gpus_per_worker):
    @ray.remote(num_gpus=gpus_per_worker)
    def steered_generation_ray(worker_id, row_ids, dataset_name, domain, layer, representation_store, 
                             output_dir, save_frequency=5, max_new_tokens=5000, batch_size=4):
        # Get the GPU IDs assigned by Ray
        import os
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        
        print(f"Worker {worker_id} using {len(gpu_ids)} GPUs: {gpu_ids}")
        
        # Load tokenizer and model
        tokenizer = initialize_tokenizer(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="balanced",
            torch_dtype=compute_dtype, 
            attn_implementation="sdpa",
        )
        
        # Load dataset
        dataset = load_dataset(dataset_name)["train"]
        
        # Get representations from store
        representations = ray.get(representation_store.get_representations.remote(domain))
        mean_reprs = representations["mean_reprs"]
        mean_domain = representations["mean_domain"]
        
        mean_actions = representations.get("mean_actions", mean_domain)
        mean_predicates = representations.get("mean_predicates", mean_domain)
        
        phrases = DOMAIN_PHRASES[domain]
        phrases = list(phrases["actions"].values()) + list(phrases["predicates"].values())
        
        results = load_intermediate_results(output_dir, worker_id)
        
        existing_ids = set(result["idx"] for result in results)
        
        # Filter out already processed row IDs
        row_ids = [idx for idx in row_ids if idx not in existing_ids]
        
        if not row_ids:
            print(f"Worker {worker_id}: All assigned indices already processed.")
            return results
            
        # Process in batches
        for batch_start in trange(0, len(row_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(row_ids))
            batch_ids = row_ids[batch_start:batch_end]
            
            # Prepare batch data
            batch_data = []
            for idx in batch_ids:
                row = dataset[idx]
    
                text = "\n\n".join(row["generation"].split("\n\n")[:40])
                tokens = tokenize_blocksworld_generation(tokenizer, row, text)[:, :-2]

                # Extract phrase positions
                phrase_positions = {
                    phrase: extract_all_phrase_positions(tokens, phrase, tokenizer, cot_only=False)
                    for phrase in phrases
                }
                
                batch_data.append({
                    "idx": idx,
                    "row": row,
                    "tokens": tokens,
                    "phrase_positions": phrase_positions
                })
            
            # Define forward hook for batched steering
            def forward_hook(module, input, output):
                """Replace output with the steered representation for batch processing"""
                output = output[0]  # Get the tensor from the tuple
                
                # If we're only processing a single token, don't apply steering
                if output.shape[1] == 1:
                    return (output,)
                
                # Create a copy since we'll be modifying it
                output_copy = output.clone()
                batch_size = output.shape[0]
                
                # Apply steering for each batch item
                for batch_idx in range(batch_size):
                    # Only process if we're in range of our actual batch data
                    # (Important for handling partial batches at the end)
                    if batch_idx < len(batch_data):
                        batch_item = batch_data[batch_idx]
                        phrase_positions = batch_item["phrase_positions"]
                        
                        # Calculate padding offset once for this batch item
                        orig_len = batch_item["tokens"].shape[1]
                        pad_len = output.shape[1] - orig_len
                        
                        # Apply steering for each phrase
                        for ip, phrase in enumerate(phrases):
                            positions = phrase_positions[phrase]
                            if ip < 4:
                                adjustment = mean_actions
                            else:
                                adjustment = mean_predicates
                            
                            # Only apply steering if we have positions and representations
                            if positions and phrase in mean_reprs:
                                # Calculate steering vector
                                r = torch.tensor(
                                    mean_reprs[phrase] - adjustment,
                                    dtype=output.dtype,
                                    device=output.device
                                )
                                
                                # Apply steering at all phrase positions
                                for sp, ep in positions:
                                    # Adjust positions to account for left padding
                                    adjusted_sp = sp + pad_len
                                    adjusted_ep = ep + pad_len
                                    
                                    # Make sure we don't go out of bounds with the current sequence
                                    if adjusted_sp < output.shape[1] and adjusted_ep <= output.shape[1]:
                                        output_copy[batch_idx, adjusted_sp:adjusted_ep] += r
                
                return (output_copy,)
            
            # Register hook to specified layer
            for m in model.modules():
                m._forward_hooks = OrderedDict()
            
            model.model.layers[layer].register_forward_hook(forward_hook)
            
            # Process batch for generation with left padding (correct for causal models)
            input_device = next(model.parameters()).device
            
            # Find maximum sequence length in this batch
            max_seq_len = max([item["tokens"].shape[1] for item in batch_data])
            
            # Create padded token tensors and attention masks
            padded_tokens = []
            attention_masks = []
            
            # Use pad_token_id if available, otherwise eos_token_id
            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            for item in batch_data:
                tokens = item["tokens"]
                seq_len = tokens.shape[1]
                
                # Create attention mask (1 for tokens, 0 for padding)
                mask = torch.ones(max_seq_len, dtype=torch.long)
                
                # For generation tasks, we use left padding (causal models)
                if seq_len < max_seq_len:
                    # Calculate padding amount
                    pad_len = max_seq_len - seq_len
                    
                    # Create left padding
                    padding = torch.full((1, pad_len), pad_token, dtype=tokens.dtype)
                    padded = torch.cat([padding, tokens], dim=1)
                    
                    # Update mask for padding positions (first pad_len positions)
                    mask[:pad_len] = 0
                else:
                    padded = tokens
                
                padded_tokens.append(padded)
                attention_masks.append(mask.unsqueeze(0))
            
            # Combine into batch
            batch_tokens = torch.cat(padded_tokens, dim=0).to(input_device)
            batch_attention_masks = torch.cat(attention_masks, dim=0).to(input_device)
            
            
            gen_tokens = model.generate(
                batch_tokens,
                attention_mask=batch_attention_masks,
                do_sample=False, 
                max_new_tokens=max_new_tokens, 
                temperature=None, 
                top_p=None, 
                top_k=None, 
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            
            # Process each result individually
            for i, (batch_item, gen_token_seq) in enumerate(zip(batch_data, gen_tokens)):
                # Decode generated text
                generated_text = tokenizer.decode(gen_token_seq, skip_special_tokens=True)
                
                results.append({
                    "idx": batch_item["idx"],
                    "original_input": batch_item["row"]["generation"],
                    "steered_generation": generated_text,
                    "token_length": len(gen_token_seq)
                })
            
            # Save intermediate results at specified frequency
            batch_idx = batch_start + batch_size
            if batch_idx % save_frequency == 0 or batch_end == len(row_ids):
                saved_file = save_intermediate_results(results, output_dir, worker_id)
                print(f"Worker {worker_id}: Saved intermediate results to {saved_file} after processing {batch_end}/{len(row_ids)} items")
        
        # Final save in case the last batch wasn't a multiple of save_frequency
        final_file = save_intermediate_results(results, output_dir, worker_id)
        print(f"Worker {worker_id}: Completed all {len(row_ids)} items. Final results saved to {final_file}")
        
        return results
    
    return steered_generation_ray

def main():
    parser = argparse.ArgumentParser(description="Parallel steering generation")
    parser.add_argument("--gpus_per_worker", type=int, default=2, 
                        help="Number of GPUs to allocate per worker")
    parser.add_argument("--rows_start", type=int, default=0,
                        help="Starting row index")
    parser.add_argument("--rows_end", type=int, default=100,
                        help="Ending row index (exclusive)")
    parser.add_argument("--dataset", type=str, default="dmitriihook/qwq-32b-planning-mystery-4-24k",
                        help="Dataset to use for generation")
    parser.add_argument("--domain", type=str, default="mystery",
                        help="Domain for steering phrases")
    parser.add_argument("--layer", type=int, default=47,
                        help="Layer to apply steering")
    parser.add_argument("--repr_path", type=str, required=True,
                        help="Path to JSON file with representations")
    parser.add_argument("--output_path", type=str, default="steered_generations.json",
                        help="Path to save generated results")
    parser.add_argument("--max_new_tokens", type=int, default=20000,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--intermediate_dir", type=str, default="intermediate_results",
                        help="Directory to save intermediate results")
    parser.add_argument("--save_frequency", type=int, default=1,
                        help="Frequency to save intermediate results (number of items)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    
    args = parser.parse_args()
    
    # Get parameters from arguments
    gpus_per_worker = args.gpus_per_worker
    rows_start = args.rows_start
    rows_end = args.rows_end
    dataset_name = args.dataset
    domain = args.domain
    layer = args.layer
    repr_path = args.repr_path
    output_path = args.output_path
    max_new_tokens = args.max_new_tokens
    intermediate_dir = args.intermediate_dir
    save_frequency = args.save_frequency
    batch_size = args.batch_size
    
    n_rows = rows_end - rows_start
    
    # Get available GPUs and calculate number of workers
    total_gpus = int(ray.cluster_resources().get("GPU", 1))
    n_workers = total_gpus // gpus_per_worker
    
    if n_workers == 0:
        raise ValueError(f"Not enough GPUs. Requested {gpus_per_worker} GPUs per worker, but only {total_gpus} available.")
    
    print(f"Starting steered generation with {total_gpus} total GPUs")
    print(f"Using {gpus_per_worker} GPUs per worker across {n_workers} workers")
    print(f"Processing in batches of {batch_size}")
    print(f"Saving intermediate results every {save_frequency} items to {intermediate_dir}")
    
    # Create intermediate results directory
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Load representations
    representations = load_representations(repr_path)
    
    # Create representation store
    repr_store = RepresentationStore.remote()
    repr_store.set_representations.remote(domain, representations)
    
    # Create task function with specified GPU allocation
    generation_task = steered_generation_ray_factory(gpus_per_worker)
    
    # Divide work by available workers
    rows_per_worker = max(1, n_rows // n_workers)
    row_ids = [list(range(rows_start + i * rows_per_worker, 
                         min(rows_start + (i + 1) * rows_per_worker, rows_end))) 
             for i in range(n_workers)]
    
    # Launch Ray tasks
    futures = []
    
    for i in range(n_workers):
        if row_ids[i]:  # Only launch if there are rows to process
            futures.append(generation_task.remote(
                i, row_ids[i], dataset_name, domain, layer, repr_store, 
                intermediate_dir, save_frequency, max_new_tokens, batch_size
            ))
    
    # Wait for all tasks to complete and gather results
    print("Waiting for tasks to complete...")
    results = ray.get(futures)
    
    # Combine results
    all_generations = []
    for result in results:
        all_generations.extend(result)
    
    # Sort by original index
    all_generations.sort(key=lambda x: x["idx"])
    
    # Save final results
    with open(output_path, 'w') as f:
        json.dump(all_generations, f, indent=2)
    
    print(f"Generated {len(all_generations)} steered texts. Results saved to {output_path}")
    
    # Check for and collect any intermediate results that might not be in the final output
    # (This can happen if a worker crashed but saved some intermediate results)
    try:
        print("Checking for any missing intermediate results...")
        intermediate_files = [f for f in os.listdir(intermediate_dir) if f.startswith("intermediate_results_worker_")]
        
        all_processed_ids = set(gen["idx"] for gen in all_generations)
        recovered_generations = []
        
        for filename in intermediate_files:
            file_path = os.path.join(intermediate_dir, filename)
            with open(file_path, 'r') as f:
                worker_results = json.load(f)
                
                for result in worker_results:
                    if result["idx"] not in all_processed_ids:
                        recovered_generations.append(result)
                        all_processed_ids.add(result["idx"])
        
        if recovered_generations:
            # Save recovered results
            recovered_path = output_path.replace(".json", "_recovered.json")
            all_generations.extend(recovered_generations)
            all_generations.sort(key=lambda x: x["idx"])
            
            with open(recovered_path, 'w') as f:
                json.dump(all_generations, f, indent=2)
            
            print(f"Recovered {len(recovered_generations)} additional results from intermediate files.")
            print(f"Combined results saved to {recovered_path}")
    except Exception as e:
        print(f"Error while checking for missing intermediate results: {e}")

if __name__ == "__main__":
    print("Starting parallel steering generation with batching")
    main()
    
    # Shut down Ray
    ray.shutdown()