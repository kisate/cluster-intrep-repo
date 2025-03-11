import openai
from dotenv import load_dotenv
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
from huggingface_hub import HfApi
import json
import re

load_dotenv()

api = HfApi()

prompt_template_4 = """Blocksworld is a scenario in which agent needs to stack blocks in a certain way.
The agent has four possible actions: pick up a block, put down a block, stack a block on top of another block, and unstack a block from another block.
The agent can only pick up one block at a time, and can only stack or put down a block if it is holding one.

You will be given a reasoning step from the blocksworld problem solution. 
You have to extract all of the actions that were mentioned in the reasoning step.

If there were no actions considered in the step, return null.

Follow the format in the examples below. Answer in the correct JSON format.
You should only use the four possible actions: pick up a block, put down a block, stack a block on top of another block, and unstack a block from another block.

Reasoning step:
1. Unstack A from D.
2. Put down A.
3. Pick up D.
4. Stack D on B.
5. Pick up A.
6. Stack A on D.

Output:
{{"actions": [["unstack", "A", "D"], ["put down", "A"], ["pick up", "D"], ["stack", "D", "B"], ["pick up", "A"], ["stack", "A", "D"]]}}


Reasoning step:
1. Move A off of D. Since A is clear, I can unstack A from D. Then put A down somewhere, maybe on the table.

Outpit:
{{"goal_action": ["unstack", "A", "D"], "actions": [["unstack", "A", "D"], ["put down", "A"]]}}

Reasoning step:
{step}

Output:
"""

prompt_template = prompt_template_4

client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
)

def thread_fn(state):
    def gen_label(state):
        prompt = prompt_template.format(state=state)
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=100,
            response_format={ "type": "json_object" },
        )
        label = response.choices[0].message.content
        return label
    
    return gen_label(state)

def process_item(item):
    new_text = item["generation"]
    parsed = None
    new_text = new_text[:-40]
    
    regex = "[A-Z]-[A-Z]"

    if new_text and not re.search(regex, new_text):
        try:
            parsed = thread_fn(new_text)
            parsed = json.loads(parsed)
        except Exception as e:
            print(e)
            parsed = None
    
    return {
        "idx": item["item_idx"],
        "line_n": item["line_n"],
        "new_text": new_text,
        "parsed": parsed
    }

def main(dataset_name, split, n_threads, save_name):
    dataset = load_dataset(dataset_name, split="train")
    
    with ThreadPool(n_threads) as pool:
        results = list(tqdm(pool.imap(process_item, dataset), total=len(dataset)))
    
    with open(f"{save_name}.json", "w") as f:
        json.dump(results, f)
    
    try:
        api.create_repo(f"dmitriihook/{save_name}", repo_type="dataset")
    except Exception as e:
        print(e)
        pass
    
    api.upload_file(
        repo_id=f"dmitriihook/{save_name}",
        path_or_fileobj=f"{save_name}.json",
        path_in_repo=f"{save_name}.json",
        repo_type="dataset"
    )

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="dmitriihook/deepseek-r1-qwen-32b-planning-4-blocks-self-probing-state-distilabel")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--n_threads", type=int, default=20)
parser.add_argument("--save_name", type=str, default="blocksworld-4-self-probing-parsed-big-v2")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dataset_name, args.split, args.n_threads, args.save_name)
