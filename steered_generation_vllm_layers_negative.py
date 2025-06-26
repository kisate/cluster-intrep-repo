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
    parser.add_argument("--start_layer", type=int, default=0, help="Start layer to apply the hook")
    parser.add_argument("--end_layer", type=int, default=47, help="End layer to apply the hook")
    parser.add_argument("--actions", action="store_true", help="Use actions only")
    parser.add_argument("--predicates", action="store_true", help="Use predicates only")
    parser.add_argument("--steering_start", type=int, default=3000, help="Window start")
    parser.add_argument("--steering_end", type=int, default=4500, help="Window end")
    parser.add_argument("--no-steering", action="store_true", help="No steering")
    parser.add_argument("--random", action="store_true", help="Shuffle representations")
    parser.add_argument("--real_random", action="store_true", help="Random representations")
    parser.add_argument("--mean_only", action="store_true", help="Use mean only")
    parser.add_argument("--triple", action="store_true", help="Repeat generation 3 times")
    parser.add_argument("--zero", action="store_true", help="Replace with zeros")
    parser.add_argument("--random_all", action="store_true", help="Shuffle actions with predicates")
    parser.add_argument("--use_10k", action="store_true", help="Use 10k tokens")
    parser.add_argument("--row_start", type=int, default=0, help="Row start")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")

    return parser.parse_args()


cur_dir = Path(".").absolute()

def load_dataset_from_file(domain_name, task_name):
    prompt_dir = cur_dir / Path(f"./cot-planning/results/{domain_name}/qwq-32b/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

def extract_all_phrase_positions(tokens, phrase, tokenizer, cot_only=True, steering_start=None):
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
                tuple([p-1, p + len(phts)])
            )        
    
    return sorted(list(set(positions)))

def create_hook(phrases, masks_batch_combined, mean_reprs, combined_len, block_size=4096, no_steering=False, logging_file=None, scale=1.0):
    def hook(module, input, output):
        meta = getattr(module, "_meta", {})
        meta["mask_offset"] = meta.get("mask_offset", 0)
        
        if meta["mask_offset"] >= combined_len:
            module._meta = {}
            module._forward_hooks = OrderedDict()
            return output
        
        mask_start = meta["mask_offset"]
        mask_end = mask_start + block_size
        
        meta["mask_offset"] = mask_end
        module._meta = meta
        
        hs, res = output
                 
        for ip, phrase in enumerate(phrases):        
            steering_mask = masks_batch_combined[phrase]
            
            steering_mask = steering_mask[mask_start:mask_end]
            
            steering_vector = mean_reprs[phrase]
            steering_vector = steering_mask[:, None] * steering_vector
        
            steering_mask = torch.tensor(steering_mask[:, None], dtype=torch.int32, device=hs.device)
            steering_vector = torch.tensor(steering_vector, dtype=hs.dtype, device=hs.device)
            
            if no_steering:
                hs = torch.where(steering_mask == 0, hs, hs)
            else:
                hs_norm = torch.norm(hs, dim=-1, keepdim=True)
                # cosine_similarity = torch.nn.functional.cosine_similarity(hs, steering_vector, dim=-1, eps=1e-8)
                new_hs = hs - steering_vector * scale
                new_hs = new_hs / (torch.norm(new_hs, dim=-1, keepdim=True) + 1e-6) * hs_norm
                hs = torch.where(steering_mask == 0, hs, new_hs)
        return hs, res
    
    return hook

def add_hook(module, hook_fn):
    module._forward_hooks = OrderedDict()
    module._meta = {}
    module.register_forward_hook(hook_fn)

def make_mean_reprs(reprs, domain_reprs, early_reprs, args, layer, shuffled_mapping, action_phrases, predicate_phrases, phrases):
    mean_reprs = {k: np.array(v) for k, v in reprs[f"{layer}"]["mean_reprs"].items()}
    early_mean_reprs = {k: np.array(v) for k, v in early_reprs[f"{layer}"]["mean_reprs"].items()}

    mean_actions = np.array(domain_reprs[f"{layer}"]["mean_actions"])
    mean_predicates = np.array(domain_reprs[f"{layer}"]["mean_predicates"])

    early_mean_actions = np.array(early_reprs[f"{layer}"]["mean_actions"])
    early_mean_predicates = np.array(early_reprs[f"{layer}"]["mean_predicates"])

    adjustments = {
        k: mean_actions if k in action_phrases else mean_predicates for k in mean_reprs
    }

    centered_reprs = {k: mean_reprs[k] - adjustments[k] for k in mean_reprs}
    early_adjustments = {
        k: early_mean_actions if k in action_phrases else early_mean_predicates for k in early_mean_reprs
    }
    early_centered_reprs = {k: early_mean_reprs[k] - early_adjustments[k] for k in early_mean_reprs}

    mean_reprs = {
        k: centered_reprs[k] - early_centered_reprs[k] for k in mean_reprs
    }

    if args.random:
        mean_reprs = {k: mean_reprs[shuffled_mapping[k]] for k in mean_reprs}

    if args.random_all:
        mean_reprs = {k: mean_reprs[shuffled_mapping[k]] for k in mean_reprs}

    if args.zero:
        mean_reprs = {k: np.zeros_like(v) for k, v in mean_reprs.items()}

    if args.real_random:
        actions_mean = np.stack(
            [v for k, v in mean_reprs.items() if k in action_phrases]
        ).mean(axis=0)
        predicates_mean = np.stack(
            [v for k, v in mean_reprs.items() if k in predicate_phrases]
        ).mean(axis=0)

        actions_std = np.stack(
            [v for k, v in mean_reprs.items() if k in action_phrases]
        ).std(axis=0)
        predicates_std = np.stack(
            [v for k, v in mean_reprs.items() if k in predicate_phrases]
        ).std(axis=0)

        mean_reprs = {k: np.random.normal(actions_mean, actions_std) if k in action_phrases else np.random.normal(predicates_mean, predicates_std) for k in phrases}


    return mean_reprs
    
    
def process_rows(row_ids: list[int], rank: int, gpus_per_worker: int, args):
    gpu_ids = list(range(rank * gpus_per_worker, (rank + 1) * gpus_per_worker))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    if args.triple:
        row_ids = [
            x for x in row_ids for _ in range(3)
        ]
    else:
        row_ids = row_ids
    
    
    model_id = args.model_id
    domain_number = args.domain_number
    steering_start = args.steering_start
    steering_end = args.steering_end
    no_steering = args.no_steering
    start_layer = args.start_layer
    end_layer = args.end_layer
    block_size = steering_end
    scale = args.scale
    # Load tokenizer
    tokenizer = initialize_tokenizer(model_id)
    
    # Load dataset
    dataset_name = f"dmitriihook/qwq-32b-planning-mystery-{domain_number}-24k-greedy"
    
    dataset = load_dataset(dataset_name)["train"]
    
    # Load representations
    repr_file = f"multilayer_representations/multilayer_7k/mystery_{domain_number}/mean_reprs_mystery_{domain_number}_multi_layer.json"
    early_repr_file = f"multilayer_representations/multilayer_2k/mystery_{domain_number}/mean_reprs_mystery_{domain_number}.json"
    domain_repr_file = f"multilayer_representations/multilayer_7k/mystery_{domain_number}/mean_reprs_mystery_{domain_number}_multi_layer.json"
        
    with open(repr_file, 'r') as f:
        reprs = json.load(f)
    
    with open(domain_repr_file, 'r') as f:
        domain_reprs = json.load(f)
    
    with open(early_repr_file, 'r') as f:
        early_reprs = json.load(f)
    
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
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        max_tokens=19000,
        temperature=0,
        top_k=1,
        seed=0,
    )
    
    results = []
    
    chunk_size = 150
    
    chunks = [
        row_ids[i:i + chunk_size] for i in range(0, len(row_ids), chunk_size)
    ]

    predicate_phrases = [x for x in phrases if x not in action_phrases]

    shuffled_mapping = None

    if args.random:
        shuffled_action_phrases = action_phrases.copy()
        np.random.shuffle(shuffled_action_phrases)
        shuffled_predicate_phrases = predicate_phrases.copy()
        np.random.shuffle(shuffled_predicate_phrases)
        shuffled_mapping = {k: v for k, v in zip(phrases, shuffled_action_phrases + shuffled_predicate_phrases)}

        print(shuffled_mapping)  

    if args.random_all:
        shuffled_phrases = phrases.copy()
        np.random.shuffle(shuffled_phrases)
        shuffled_mapping = {k: v for k, v in zip(phrases, shuffled_phrases)}

    
    for _row_ids in chunks:
        all_tokens = []
        phrase_masks = {phrase: [] for phrase in phrases}
        
        for i in _row_ids:
            row = dataset[i]
            
            # Process text
            tokens = tokenize_blocksworld_generation(tokenizer, row)[:, :-2][0]
            tokens = tokens[:steering_end]

            all_tokens.append(tokens)
            
            # Get phrase positions for this row
            phrase_positions = {
                phrase: extract_all_phrase_positions(tokens, phrase, tokenizer, cot_only=False, steering_start=steering_start)
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
        
        for layer in range(start_layer, end_layer + 1):
            mean_reprs = make_mean_reprs(reprs, domain_reprs, early_reprs, args, layer, shuffled_mapping, action_phrases, predicate_phrases, phrases)


            logging_file = f"{args.output_path}/worker_results/{layer}.txt"
            Path(logging_file).parent.mkdir(parents=True, exist_ok=True)
            
            current_hook = create_hook(
                phrases=phrases,
                masks_batch_combined=masks_combined,
                mean_reprs=mean_reprs,
                combined_len=combined_len,
                block_size=block_size,
                no_steering=no_steering,
                logging_file=logging_file,
                scale=scale
            )
            # Apply hook to model
            llm.apply_model(
                lambda x: add_hook(x.model.layers[layer], current_hook),
            )
        
        # Create prompts for each row in batch
        prompts = [TokensPrompt(prompt_token_ids=tokens.tolist()) for tokens in all_tokens]
        
        # Generate
        results.extend(llm.generate(prompts, sampling_params=sampling_params))
    
    output_path = Path(args.output_path) / f"worker_results/{rank}.json"
    
    json_results = [
        {
            "idx": row_ids[i],
            "copy": i % 3,
            "steered_generation": results[i].outputs[0].text,
            "original_input": dataset[row_ids[i]]["generation"],
        }
        for i in range(len(results))
    ] 
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
        


def main():
    args = parse_args()
    n_workers = args.n_workers
    n_rows = args.num_rows
    row_start = args.row_start
    total_available_gpus = torch.cuda.device_count()
    gpus_per_worker = total_available_gpus // n_workers
    
    if gpus_per_worker == 0:
        raise ValueError("Not enough GPUs available for the number of workers.")
    
    rows_per_worker = n_rows // n_workers
    
    # Create a list of row IDs for each worker
    row_ids = [list(range(i * rows_per_worker + row_start, (i + 1) * rows_per_worker + row_start)) for i in range(n_workers)]
    
    # Handle remaining rows if n_rows is not perfectly divisible by n_workers
    remainder = n_rows % n_workers
    for i in range(remainder):
        row_ids[i].append(n_rows - remainder + i)
    
    # Create output path if not specified
    if args.output_path is None:
        args.output_path = f"./results/mystery_{args.domain_number}/qwq-32b"
    
    print(f"Processing {n_rows} rows with {n_workers} workers ({gpus_per_worker} GPUs per worker)")
    print(f"Using {args.start_layer} start layer and {args.end_layer} end layer")
    print(f"Using {args.steering_start} steering start and {args.steering_end} steering end")
    print(f"Using {args.random} random")
    
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