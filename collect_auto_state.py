import torch
import json
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
from utils import initialize_tokenizer, tokenize_blocksworld_generation, THINK_TOKEN, THINK_START_TOKEN
from pathlib import Path
from threading import Thread


compute_dtype = torch.bfloat16
device   = 'cuda'
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

tokenizer = initialize_tokenizer(model_id)
n_threads = 4

models = {
    i: AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation="sdpa", device_map=f"cuda:{i}") for i in range(n_threads)
}

save_path = Path(f"self_probing")
file_name = "self_probing_state_6_blocks_{part}.jsonl"

def process_row(row, device_id):
    text = ""
    generation = row["generation"]
    device = f"cuda:{device_id}"
    model = models[device_id]

    results = []

    for line_n, line in enumerate(generation.split("\n\n")):
        text = text + line + "\n\n"
        if line_n < 20 or len(line) < 50:
            results.append({"line": line, "hidden_states": None, "new_text": None, "line_n": line_n})
            continue

        _text = text + "Now, the stacks are:\n\n"
        tokens = tokenize_blocksworld_generation(tokenizer, row, _text)[:, :-1]

        with torch.no_grad():
            new_generation = model.generate(tokens.to(device), do_sample=False, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(tokens).to(device), temperature=None, top_p=None)

        new_text = tokenizer.decode(new_generation[0][tokens.shape[1]:], skip_special_tokens=True)

        with torch.no_grad():
            hidden_states = model(tokens.to(device), output_hidden_states=True)

        hidden_states = [x[0][-10:].cpu().to(torch.float16).numpy().tolist() for x in hidden_states.hidden_states[-1:]]

        results.append({"line": line, "hidden_states": hidden_states, "new_text": new_text, "line_n": line_n})


    combined = {
        "idx": row["idx"],
        "results": results
    }

    with open(save_path / file_name.format(part=device_id), "a") as f:
        f.write(json.dumps(combined) + "\n")

def thread_fn(rows, device_id):
    pbar = tqdm(total=len(rows))
    for row in rows:
        process_row(row, device_id)
        pbar.update(1)
        pbardesc = f"Thread {device_id}"
        pbar.set_description(pbardesc)
    pbar.close()


def main(n_rows):
    blocksworld_type = "6-blocks"
    dataset = load_dataset(f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type}")["train"]


    dataset = dataset.add_column("idx", range(len(dataset)))
    dataset = dataset.select(range(n_rows))

    save_path.mkdir(exist_ok=True)

    n_rows = len(dataset)

    rows_per_thread = n_rows // n_threads

    threads = []

    for i in range(n_threads):
        start = i * rows_per_thread
        end = (i + 1) * rows_per_thread
        thread = Thread(target=thread_fn, args=(dataset.select(range(start, end)), i))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main(300)