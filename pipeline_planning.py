from typing import Any, Dict
from distilabel.models import vLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import Step
from pathlib import Path
from datasets import Dataset
import json
import os

prompt_template = """{{ instruction }}\n\nThink step by step and then generate the plan. Make sure to use the exact format as in the examples. Otherwise correct answer won't be accepted."""
# prompt_template = """{{ instruction }}"""

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # Exchange with another smol distilled r1
model_id = "Qwen/QwQ-32B"
# model_id = "Qwen/Qwen2.5-32B-Instruct"
model_id = "meta-llama/Llama-3.3-70B-Instruct"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
model_id = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"

pipeline_name = "distill-qwen-32b-r1-planning-6-blocks-small"
pipeline_name = "qwq-32b-mystery-24k"
pipeline_name = "qwq-32b-6-blocks"
pipeline_name = "qwq-32b-mystery-2-24k"
pipeline_name = "qwq-32b-mystery-3-24k"
pipeline_name = "qwq-32b-mystery-4-24k"
pipeline_name = "gpt-4.1-4-blocks"
pipeline_name = "qwq-32b-4-blocks"
pipeline_name = "llama-3.3-70b-planning-4-blocks"
pipeline_name = "deepseek-llama-70b-planning-4-blocks"
pipeline_name = "nemotron-v1-49b-planning-4-blocks-2"

repo_id = "dmitriihook/qwq-32b-planning-4-blocks"
# repo_id = "dmitriihook/qwq-32b-planning-mystery-2-24k"
# repo_id = "dmitriihook/qwq-32b-planning-mystery-3-24k"
# repo_id = "dmitriihook/qwq-32b-planning-mystery-4-24k"
# repo_id = "dmitriihook/gpt-4.1-4-blocks"
repo_id = "dmitriihook/llama-3.3-70b-planning-4-blocks"
# gpt_id = "openai/gpt-4.1-2025-04-14"
repo_id = "dmitriihook/deepseek-llama-70b-planning-4-blocks"
repo_id = "dmitriihook/nemotron-v1-49b-planning-4-blocks-2"

with Pipeline(
    name=pipeline_name,
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        cuda_devices=list(range(8)),
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 8,
            "max_model_len": 8192 * 3,
        },
        generation_kwargs={
            "temperature": 0,
            "max_new_tokens": 8192 * 3,
        },
        trust_remote_code=True
    )

    # llm = OpenAILLM(
    #     model="openai/gpt-4.1-2025-04-14",
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     generation_kwargs={
    #         "temperature": 0,
    #         "max_new_tokens": 8192,
    #     },
    # )

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
    prompt_dir = Path(f"./cot-planning/prompts/{domain_name}/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    args = parser.parse_args()

    domain_name = "blocksworld_mystery_4"
    domain_name = "blocksworld_4_blocks"
    task_name = "plan_generation_po"
    dataset = load_dataset_from_file(domain_name, task_name)

    dataset = Dataset.from_list(dataset["instances"]).select(range(350))

    print(dataset)
    
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id=repo_id)
