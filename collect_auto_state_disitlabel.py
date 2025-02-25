from typing import Any, Dict
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import Step
from pathlib import Path
from datasets import Dataset, load_dataset
import json
import os
from utils import initialize_tokenizer

prompt_template = """{{ instruction }}"""

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # Exchange with another smol distilled r1
tokenizer = initialize_tokenizer(model_id)

with Pipeline(
    name="distill-qwen-32b-r1-planning-4-blocks-self-probing-state-distilabel",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        cuda_devices=list(range(8)),
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 8,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0,
            "max_new_tokens": 50,
        },
        chat_template="{{messages[0]['content']}}"
    )
    prompt_column = "gen_text"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=1,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


from argparse import ArgumentParser

parser = ArgumentParser()

def apply_template(tokenizer, row, text):
    query = row["distilabel_metadata"]["raw_input_text_generation_0"][0]

    messages = [
        query,
        {"role": "assistant", "content": text}
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return chat

def process_row(row):
    text = ""
    generation = row["generation"]
    idx = row["idx"]
    
    rows = []

    for line_n, line in enumerate(generation.split("\n\n")):
        text = text + line + "\n\n"
        if line_n < 10 or len(line) < 30:
            continue
        
        _text = text + "Now, the stacks are:\n\n"

        _text = apply_template(tokenizer, row, _text)
        _text = _text.replace("<｜end▁of▁sentence｜>", "")

        rows.append({"gen_text": _text, "line_n": line_n, "item_idx": idx})

    return rows

def process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    folded_batch = []
    for i in range(len(batch["generation"])):
        folded_batch.append({k: v[i] for k, v in batch.items()})

    results = []
    for row in folded_batch:
        results.append(process_row(row))

    unfolded_results = {}

    for key in results[0][0]:
        unfolded_results[key] = [y[key] for x in results for y in x]

    return unfolded_results

if __name__ == "__main__":
    args = parser.parse_args()
    start = 0
    end = 1500

    blocksworld_type = "4-blocks"
    dataset = load_dataset(f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type}")["train"]
    dataset = dataset.add_column("idx", range(len(dataset)))    

    dataset: Dataset = dataset.select(range(start, end))

    dataset = dataset.map(process_batch, batched=True, num_proc=20, remove_columns=dataset.column_names, load_from_cache_file=False)
    
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="dmitriihook/deepseek-r1-qwen-32b-planning-4-blocks-self-probing-state-distilabel")
