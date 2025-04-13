from typing import Any, Dict
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import Step
from pathlib import Path
from datasets import Dataset
import json
import os
from argparse import ArgumentParser

def load_dataset_from_file(domain_name, task_name):
    prompt_dir = Path(f"./cot-planning/prompts/{domain_name}/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mystery", type=int, required=True, 
                        help="Mystery number (required)")
    parser.add_argument("--model", type=str, default="Qwen/QwQ-32B", 
                        help="Model ID to use")
    parser.add_argument("--task", type=str, default="plan_generation_po", 
                        help="Task name")
    parser.add_argument("--context_length", type=int, default=24576, 
                        help="Context length in tokens (default: 24k)")
    parser.add_argument("--num_samples", type=int, default=350, 
                        help="Number of samples to process")
    parser.add_argument("--gpu_count", type=int, default=8, 
                        help="Number of GPUs to use")
    parser.add_argument("--username", type=str, default="dmitriihook",
                        help="Username for HuggingFace repository")
    
    args = parser.parse_args()
    
    # Determine model ID
    model_id = args.model
    model_short_name = model_id.split("/")[-1].lower()
    
    # Configure domain name and pipeline name based on mystery number
    if args.mystery == 1:
        domain_name = "blocksworld_mystery"
    else:
        domain_name = f"blocksworld_mystery_{args.mystery}"
    
    pipeline_name = f"{model_short_name}-mystery-{args.mystery}-{args.context_length//1000}k-greedy"
    repo_id = f"{args.username}/{model_short_name}-planning-mystery-{args.mystery}-{args.context_length//1000}k-greedy"
    
    task_name = args.task
    
    # Print configuration information
    print(f"Running pipeline with configuration:")
    print(f"  Model: {model_id}")
    print(f"  Domain: {domain_name}")
    print(f"  Task: {task_name}")
    print(f"  Pipeline name: {pipeline_name}")
    print(f"  Repository ID: {repo_id}")
    print(f"  Context length: {args.context_length}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Using {args.gpu_count} GPUs")
    
    # Create and configure the pipeline
    prompt_template = """{{ instruction }}"""
    
    with Pipeline(
        name=pipeline_name,
        description=f"Pipeline to generate data from {model_id} for mystery {args.mystery}",
    ) as pipeline:
        llm = vLLM(
            cuda_devices=list(range(args.gpu_count)),
            model=model_id,
            tokenizer=model_id,
            extra_kwargs={
                "tensor_parallel_size": args.gpu_count,
                "max_model_len": args.context_length,
            },
            generation_kwargs={
                "temperature": 0,
                "max_new_tokens": args.context_length,
                "top_k": 1
            },
        )
        
        prompt_column = "query"
        text_generation = TextGeneration(
            llm=llm, 
            template=prompt_template,
            num_generations=1,
            input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
        )
    
        # Load and prepare dataset
        try:
            dataset = load_dataset_from_file(domain_name, task_name)
            dataset = Dataset.from_list(dataset["instances"]).select(range(args.num_samples))
            print(f"Dataset loaded successfully: {dataset}")
            
            # Run pipeline and push results
            distiset = pipeline.run(dataset=dataset)
            distiset.push_to_hub(repo_id=repo_id)
            print(f"Pipeline completed successfully. Results pushed to {repo_id}")
        except Exception as e:
            print(f"Error processing dataset: {e}")