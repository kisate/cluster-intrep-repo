import torch
import json
import multiprocessing as mp
from transformers import AutoModelForCausalLM
from utils import initialize_tokenizer, tokenize_blocksworld_generation
from stacks_utils import *
from datasets import load_dataset
from tqdm import tqdm

compute_dtype = torch.bfloat16
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
n_blocks = 6
parsed_datasets = {
    4: "blocksworld-4-blocks-actions-first.json",
    6: "blocksworld-6-blocks-actions-first.json"
}

save_path = "first_action_token_logits_{core}.json"

def main(core_id: int, row_id: list[int]):
    tokenizer = initialize_tokenizer(model_id)
    dataset = load_dataset(
    f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type[n_blocks]}")["train"]

    device = f"cuda:{core_id}"

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=compute_dtype, attn_implementation="sdpa")

    all_blocks = [chr(65 + i) for i in range(n_blocks)]

    block_tokens = [
        tokenizer.encode(" " + b)[1] for b in all_blocks
    ]

    results = []

    for idx in tqdm(row_id):
        row = dataset[idx]
        block_logits = defaultdict(list)

        text = ""

        for line_n, step in enumerate(row["generation"].split("\n\n")[:50]):
            _text = text + "\n\nFirst, I need to move block" 
            tokens = tokenize_blocksworld_generation(tokenizer, row, _text)[0, :-1]

            with torch.no_grad():
                logits = model(tokens.unsqueeze(0).to(device))[0][0, -1, :]

            for block_name, block_token in zip(all_blocks, block_tokens):
                block_logits[block_name].append(logits[block_token].item())

            text = text + "\n\n" + step

        results.append({
            "row_id": idx,
            "block_logits": block_logits
        })

        with open(save_path.format(core=core_id), "w") as f:
            json.dump(results, f)



if __name__ == "__main__":
    n_processes = 8
    n_rows = 2000

    with mp.Pool(n_processes) as pool:
        rows_per_process = n_rows // n_processes
        row_ids = [list(range(i * rows_per_process, (i + 1) * rows_per_process)) for i in range(n_processes)]

        pool.starmap(main, [(i, row_ids[i]) for i in range(n_processes)])

    with open(save_path.format(core="all"), "w") as f:
        results = []

        for i in range(n_processes):
            with open(save_path.format(core=i), "r") as f:
                results.extend(json.load(f))

        json.dump(results, f)

        

    
