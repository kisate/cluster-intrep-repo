from typing import Any, Dict
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import Step
import json
import os

prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

def load_dataset_from_file(domain_name, task_name):
    prompt_dir = f"prompts/{domain_name}/"
    with open(prompt_dir + f"{task_name}.json", 'r') as file:
        return json.load(file)

domain_name = "your_domain_name"  # Replace with actual domain name
task_name = "your_task_name"  # Replace with actual task name
dataset = load_dataset_from_file(domain_name, task_name)

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1-train",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )

if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="dmitriihook/numina-deepseek-r1-qwen-7b-train")
