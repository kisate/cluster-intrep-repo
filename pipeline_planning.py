from typing import Any, Dict
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import Step
from pathlib import Path
from datasets import Dataset
import json
import os

prompt_template = """{{ instruction }}"""

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-32b-r1-planning-mystery",
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
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "query"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=1,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--domain", type=str)
parser.add_argument("--task", type=str)

import os

print(os.listdir("."))

def load_dataset_from_file(domain_name, task_name):
    prompt_dir = Path(f"./planning/cot-planning/prompts/{domain_name}/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    args = parser.parse_args()

    domain_name = "blocksworld_mystery"
    task_name = "plan_generation_po"
    dataset = load_dataset_from_file(domain_name, task_name)

    dataset = Dataset.from_list(dataset["instances"])

    print(dataset)
    
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="dmitriihook/deepseek-r1-qwen-32b-planning-mystery")
