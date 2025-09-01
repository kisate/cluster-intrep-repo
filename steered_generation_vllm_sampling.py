import os
# os.environ["VLLM_USE_V1"] = "1"
import torch
import json
import numpy as np
import argparse
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset
from utils import initialize_tokenizer, tokenize_blocksworld_generation, DOMAIN_PHRASES
from collections import OrderedDict
from vllm import TokensPrompt, SamplingParams, LLM
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLLM on specified mystery domain")
    parser.add_argument("--domain_number", type=int, required=True, help="Mystery domain number (e.g., 2)")
    parser.add_argument("--num_rows", type=int, default=100, help="Number of rows to process")
    parser.add_argument("--model_id", type=str, default="Qwen/QwQ-32B", help="Model ID")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path")
    parser.add_argument("--n_workers", type=int, default=2, help="Number of workers to use")
    parser.add_argument("--target_layer", type=int, default=47, help="Target layer to apply the hook")
    parser.add_argument("--scale", type=float, default=1.0, help="Steering scale")
    parser.add_argument("--actions", action="store_true", help="Use actions only")
    parser.add_argument("--predicates", action="store_true", help="Use predicates only")
    parser.add_argument("--steering_start", type=int, default=1000, help="Steering start")
    parser.add_argument("--steering_end", type=int, default=3000, help="Steering end")
    parser.add_argument("--use_avg", action="store_true", help="Use average representation")
    parser.add_argument("--random", action="store_true", help="Random actions")
    parser.add_argument("--full_random", action="store_true", help="Random representations")
    parser.add_argument("--only_phrase_tokens", action="store_true", help="Only steer on phrase tokens")
    parser.add_argument("--use_10k", action="store_true", help="Use 10k token representations")
    parser.add_argument("--rows_start", type=int, default=0, help="Rows start")
    parser.add_argument("--n_generations", type=int, default=3, help="Number of generations")
    return parser.parse_args()


cur_dir = Path(".").absolute()

def load_dataset_from_file(domain_name, task_name):
    prompt_dir = cur_dir / Path(f"./cot-planning/results/{domain_name}/qwq-32b/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

def extract_all_phrase_positions(tokens, phrase, tokenizer, cot_only=True, steering_start=None, only_phrase_tokens=False):
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

    start_shift = -1
    if only_phrase_tokens:
        start_shift = 0

    if cot_only:
        start_pos = torch.where(tokens == 151667)[0]
        start_mask = torch.arange(tokens.shape[0]) >= start_pos

    for phts in phrase_tokens:
        presence_mask = torch.ones_like(tokens)
        if cot_only:
            presence_mask = presence_mask * start_mask

        for i, t in enumerate(phts):
            presence_mask = presence_mask * (tokens == t)[i:]
            presence_mask = presence_mask[:-1]

        for p in (torch.where(presence_mask)[0]).tolist():
            if p < steering_start:
                continue
            positions.add(
                tuple([p + start_shift, p + len(phts)])
            )        
    
    return sorted(list(set(positions)))

def create_hook(phrases, action_phrases, masks_batch_combined, mean_reprs, mean_actions, mean_predicates, combined_len, block_size=4096, scale=1):
    def hook(module, input, output):
        meta = getattr(module, "_meta", {})
        meta["mask_offset"] = meta.get("mask_offset", 0)
        
        if meta["mask_offset"] >= combined_len:
            return output
        
        mask_start = meta["mask_offset"]
        mask_end = mask_start + block_size
        
        meta["mask_offset"] = mask_end
        module._meta = meta
        
        hs, res = output
      
        for ip, phrase in enumerate(phrases):
            if phrase in action_phrases:
                adjustment = mean_actions
            else:
                adjustment = mean_predicates
            
            steering_mask = masks_batch_combined[phrase]
            
            steering_mask = steering_mask[mask_start:mask_end]
            steering_mask = np.concatenate([steering_mask, np.zeros(hs.shape[0] - steering_mask.shape[0])], axis=0)
            
            steering_vector = mean_reprs[phrase] - adjustment
            steering_vector = steering_mask[:, None] * steering_vector
        
            steering_mask = torch.tensor(steering_mask[:, None], dtype=torch.int32, device=hs.device)
            steering_vector = torch.tensor(steering_vector, dtype=hs.dtype, device=hs.device)
            
            a = 1 / (1 + scale)
            b = 1 - a

            hs_norm = torch.norm(hs, dim=-1, keepdim=True)
            
            if scale > 0:
                new_hs = hs * a + steering_vector * b
                new_hs = new_hs / (torch.norm(new_hs, dim=-1, keepdim=True) + 1e-6) * hs_norm
            else:
                new_hs = hs
            
            hs = torch.where(steering_mask == 0, hs, new_hs)

            # hs += steering_vector * scale
        return hs, res
    
    return hook

def add_hook(module, hook_fn):
    module._forward_hooks = OrderedDict()
    module._meta = {}
    module.register_forward_hook(hook_fn)
    
    
def process_rows(row_ids: list[int], rank: int, gpus_per_worker: int, args):
    gpu_ids = list(range(rank * gpus_per_worker, (rank + 1) * gpus_per_worker))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    model_id = args.model_id
    domain_number = args.domain_number
    steering_start = args.steering_start
    steering_end = args.steering_end
    block_size = steering_end
    target_layer = args.target_layer
    use_avg = args.use_avg
    only_phrase_tokens = args.only_phrase_tokens
    use_10k = args.use_10k
    # Load tokenizer
    tokenizer = initialize_tokenizer(model_id)
    
    # Load dataset
    dataset_name = f"dmitriihook/qwq-32b-planning-mystery-{domain_number}-24k-greedy"
    
    dataset = load_dataset(dataset_name)["train"]
    
    repr_dir = "multilayer_7k"
    if use_10k:
        repr_dir = "multilayer_10k"

    if use_avg:
        if use_10k:
            repr_file = f"multilayer_representations_avg/{repr_dir}/mystery_{domain_number}/mean_reprs_mystery_{domain_number}_10k_multi_layer.json"
        else:
            repr_file = f"multilayer_representations_avg/{repr_dir}/mystery_{domain_number}/mean_reprs_mystery_{domain_number}_multi_layer.json"
    else:
        if use_10k:
            repr_file = f"multilayer_representations/{repr_dir}/mystery_{domain_number}/mean_reprs_mystery_{domain_number}_10k_multi_layer.json"
        else:
            repr_file = f"multilayer_representations/{repr_dir}/mystery_{domain_number}/mean_reprs_mystery_{domain_number}_multi_layer.json"
    
    print(f"Loading representations from {repr_file}")
    
    with open(repr_file, 'r') as f:
        reprs = json.load(f)[f"{target_layer}"]
    
    
    mean_reprs = {k: np.array(v) for k, v in reprs["mean_reprs"].items()}
    mean_actions = np.array(reprs["mean_actions"])
    mean_predicates = np.array(reprs["mean_predicates"])
    
    # Get domain phrases
    domain_key = f"mystery_{domain_number}"
    phrases = DOMAIN_PHRASES[domain_key]
    action_phrases = list(phrases["actions"].values())
    
    if args.actions:
        phrases = list(phrases["actions"].values())
    elif args.predicates:
        phrases = list(phrases["predicates"].values())
    else:
        phrases = list(phrases["actions"].values()) + list(phrases["predicates"].values())
    
    # Initialize LLM
    llm = LLM(
        model=model_id, 
        tensor_parallel_size=gpus_per_worker, 
        enforce_eager=True, 
        max_seq_len_to_capture=20000, 
        max_num_batched_tokens=block_size,
        max_num_seqs=400,
        gpu_memory_utilization=0.95
    )
    
    
    results = []
    
    chunk_size = 100
    
    chunks = [
        row_ids[i:i + chunk_size] for i in range(0, len(row_ids), chunk_size)
    ]

    json_results = []
    
    for _row_ids in chunks:
        for generation_n in range(args.n_generations):
            all_tokens = []
            phrase_masks = {phrase: [] for phrase in phrases}
            
            for i in _row_ids:
                row = dataset[i]
                
                # Process text
                # text = "\n\n".join(row["generation"].split("\n\n")[:initial_lines])
                tokens = tokenize_blocksworld_generation(tokenizer, row)[:, :-2][0]
                tokens = tokens[:steering_end]
                all_tokens.append(tokens)
                
                # Get phrase positions for this row
                phrase_positions = {
                    phrase: extract_all_phrase_positions(tokens, phrase, tokenizer, cot_only=False, steering_start=steering_start, only_phrase_tokens=only_phrase_tokens)
                    for phrase in phrases
                }
                
                # Create phrase masks for this row
                row_phrase_masks = {
                    phrase: np.zeros(tokens.shape[0])
                    for phrase in phrases
                }
                
                for phrase in phrases:
                    positions = phrase_positions[phrase]
                    for start, end in positions:
                        row_phrase_masks[phrase][start:end] = 1
                
                # Add masks to batch
                for phrase in phrases:
                    phrase_masks[phrase].append(row_phrase_masks[phrase])
            
            masks_combined = {
                k: np.concatenate(v, axis=0) for k, v in phrase_masks.items()
            }
            
            combined_len = masks_combined[phrases[0]].shape[0] if phrases else 0

            predicate_phrases = [x for x in phrases if x not in action_phrases]

            if args.random:
                shuffled_actions = action_phrases.copy()
                np.random.shuffle(shuffled_actions)
                
                shuffled_predicates = predicate_phrases.copy()
                np.random.shuffle(shuffled_predicates)

                shuffled_mapping = {
                    action_phrases[i]: shuffled_actions[i] for i in range(len(action_phrases))
                }

                for i in range(len(predicate_phrases)):
                    shuffled_mapping[predicate_phrases[i]] = shuffled_predicates[i]
                    
                masks_combined = {
                    shuffled_mapping[k]: v for k, v in masks_combined.items()
                }
                
                # phrases = list(shuffled_mapping.values())

            if args.full_random:
                action_reprs = np.stack([v for k, v in mean_reprs.items() if k in action_phrases])
                predicate_reprs = np.stack([v for k, v in mean_reprs.items() if k in predicate_phrases])

                actions_mean = np.mean(action_reprs, axis=0)
                predicates_mean = np.mean(predicate_reprs, axis=0)

                actions_std = np.std(action_reprs, axis=0)
                predicates_std = np.std(predicate_reprs, axis=0)

                mean_reprs = {
                    **{k: np.random.normal(actions_mean, actions_std) for k in action_phrases},
                    **{k: np.random.normal(predicates_mean, predicates_std) for k in predicate_phrases},
                }
                
            # Create a hook function using the factory
            current_hook = create_hook(
                phrases=phrases,
                action_phrases=action_phrases,
                masks_batch_combined=masks_combined,
                mean_reprs=mean_reprs,
                mean_actions=mean_actions,
                mean_predicates=mean_predicates,
                combined_len=combined_len,
                block_size=block_size,
                scale=args.scale / 2,
            )
            
            # Apply hook to model
            
            # for layer in range(40, 48):
            llm.apply_model(
                lambda x: add_hook(x.model.layers[target_layer], current_hook),
            )
            
            # Create prompts for each row in batch
            prompts = [TokensPrompt(prompt_token_ids=tokens.tolist()) for tokens in all_tokens]

                # Setup sampling parameters
            sampling_params = SamplingParams(
                max_tokens=19000,
                temperature=0.6,
                # top_k=20,
                seed=generation_n,
                top_p=0.95,
                min_p=0,
            )
            
            # Generate
            results = llm.generate(prompts, sampling_params=sampling_params)

            json_results.extend([
                {
                    "idx": _row_ids[i],
                    "copy": generation_n,
                    "steered_generation": results[i].outputs[0].text,
                    "original_input": dataset[_row_ids[i]]["generation"],
                }
                for i in range(len(results))
            ])
    
    output_path = Path(args.output_path) / f"worker_results/{rank}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
        


def main():
    args = parse_args()
    n_workers = args.n_workers
    n_rows = args.num_rows
    rows_start = args.rows_start
    total_available_gpus = torch.cuda.device_count()
    gpus_per_worker = total_available_gpus // n_workers
    
    if gpus_per_worker == 0:
        raise ValueError("Not enough GPUs available for the number of workers.")
    
    rows_per_worker = n_rows // n_workers
    
    # Create a list of row IDs for each worker
    row_ids = [list(range(rows_start + i * rows_per_worker, rows_start + (i + 1) * rows_per_worker)) for i in range(n_workers)]
    
    # Handle remaining rows if n_rows is not perfectly divisible by n_workers
    remainder = n_rows % n_workers
    for i in range(remainder):
        row_ids[i].append(n_rows - remainder + i)
    
    # Create output path if not specified
    if args.output_path is None:
        args.output_path = f"./results/mystery_{args.domain_number}/qwq-32b"
    
    print(f"Processing {n_rows} rows with {n_workers} workers ({gpus_per_worker} GPUs per worker)")
    # Import multiprocessing here to avoid issues with CUDA initialization
    import torch.multiprocessing as mp
    
    # # Start multiprocessing pool
    mp.set_start_method('spawn', force=True)
    
    # process_rows(row_ids[0], 0, gpus_per_worker, args)
    
    # Create processes
    processes = []
    for rank in range(n_workers):
        p = mp.Process(
            target=process_rows,
            args=(row_ids[rank], rank, gpus_per_worker, args)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Combine results from all workers
    all_results = []
    for rank in range(n_workers):
        output_path = Path(args.output_path) / f"worker_results/{rank}.json"
        if output_path.exists():
            with open(output_path, 'r') as f:
                worker_results = json.load(f)
                all_results.extend(worker_results)
    
    # Sort results by original index
    all_results.sort(key=lambda x: x["idx"])
    
    # Save combined results
    combined_output_path = Path(args.output_path) / f"steered_results_mystery_{args.domain_number}.json"
    with open(combined_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"All results combined and saved to {combined_output_path}")


if __name__ == "__main__":
    main()