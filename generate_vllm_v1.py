import os
import vllm
import json
from pathlib import Path
from argparse import ArgumentParser
from datasets import Dataset
from transformers import AutoTokenizer

def load_dataset_from_file(domain_name, task_name):
    prompt_dir = Path(f"./cot-planning/prompts/{domain_name}/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mystery", type=int, required=True, 
                        help="Mystery number (required)")
    parser.add_argument("--model", type=str, default="nvidia/Llama-3_3-Nemotron-Super-49B-v1", 
                        help="Model ID to use")
    parser.add_argument("--task", type=str, default="plan_generation_po", 
                        help="Task name")
    parser.add_argument("--context_length", type=int, default=24576, 
                        help="Context length in tokens (default: 24k)")
    parser.add_argument("--num_samples", type=int, default=400, 
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
    repo_id = f"{args.username}/{model_short_name}-planning-mystery-{args.mystery}-{args.context_length//1000}k-greedy-long"

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
    prompt_template = """{instruction}\n\nStrictly adhere to the format in the examples.\n\nThis is a complex problem, so make sure to think for an extended period of time. Do not mind the time constraints."""

    llm = vllm.LLM(
        model=model_id,
        tensor_parallel_size=args.gpu_count,
        max_seq_len_to_capture=args.context_length,
        enforce_eager=True,
        trust_remote_code=True
    )

    sampling_params = vllm.SamplingParams(
        max_tokens=20000,
        temperature=0,
        top_k=1,
        seed=0,
    )
    

    dataset = load_dataset_from_file(domain_name, task_name)
    dataset = Dataset.from_list(dataset["instances"]).select(range(args.num_samples))
    

    tokens = []

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    queries = []

    for i in range(args.num_samples):
        row = dataset[i]
        content = prompt_template.format(instruction=row["query"])
        
        messages = [{"role": "system", "content": f"detailed thinking on"},{"role": "user", "content": content}]
        tokens.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        queries.append(content)


    prompts = [
        vllm.TokensPrompt(prompt_token_ids=x) for x in tokens
    ]


    results = llm.generate(prompts, sampling_params)

    records = []

    for i, result in enumerate(results):
        records.append({
            "query": queries[i],
            "instance_id": dataset[i]["instance_id"],
            "Number of blocks": dataset[i]["Number of blocks"],
            "generation": result.outputs[0].text,
            "distilabel_metadata": {
                "raw_input_text_generation_0": [{
                    "content": queries[i],
                    "role": "user",
                }],
            },
        })

    dataset = Dataset.from_list(records)
    dataset.push_to_hub(repo_id)
