import openai
from dotenv import load_dotenv
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
from huggingface_hub import HfApi
import json

load_dotenv()

api = HfApi()

prompt_template = """Blocksworld is a scenario in which agent needs to stack blocks in a certain way.
The agent has four possible actions: pick up a block, put down a block, stack a block on top of another block, and unstack a block from another block.
The agent can only pick up one block at a time, and can only stack or put down a block if it is holding one.

You will be given a reasoning step from the blocksworld problem solution. 
You have to extract the goal action from the reasoning step. 
Then you have to extract all of the actions that were found to be necessary to reach the goal action in the reasoning step.
If there was not explicitly stated goal action, return null in the goal action field. Then extract all of the actions that were planned in the reasoning step.

If there were no actions considered in the step, return null in the actions field.

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
{{"goal_action": null, "actions": [["unstack", "A", "D"], ["put down", "A"], ["pick up", "D"], ["stack", "D", "B"], ["pick up", "A"], ["stack", "A", "D"]]}}


Reasoning step:
1. Move A off of D. Since A is clear, I can unstack A from D. Then put A down somewhere, maybe on the table.

Outpit:
{{"goal_action": ["unstack", "A", "D"], "actions": [["unstack", "A", "D"], ["put down", "A"]]}}

Reasoning step:
{step}

Output:
"""

client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
)


from pydantic import BaseModel

class ExtractedActions(BaseModel):
    goal_action: list[str] | None
    actions: list[list[str]] | None


def check_step(step):
    keywords = ["pick up", "put down", "stack", "unstack"]
    for keyword in keywords:
        if keyword in step.lower():
            return True


def thread_fn(step):
    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(3))
    def gen_label(step):
        prompt = prompt_template.format(step=step)
        if check_step(step):
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": prompt},
                ],
                max_tokens=200,
                response_format={ "type": "json_object" },
            )

            label = response.choices[0].message.content
        else:
            label = None

        return label
    
    return gen_label(step)

def process_item(item):
    text = item["generation"]
    steps = text.split("\n\n")
    group = {
        "index": item["index"],
        "steps": []
    }
    for step in steps:
        try:
            label = thread_fn(step)
        except Exception as e:
            print(e)
            label = None

        group["steps"].append({
            "step": step,
            "label": label
        })
    return group

def main(start, end, n_threads, save_name):
    blocksworld_type = "big"
    dataset = load_dataset(f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type}")["train"]

    dataset = dataset.add_column("index", [i for i in range(len(dataset))])
    items = dataset.select(range(start, end))
    with ThreadPool(n_threads) as pool:
        results = list(tqdm(pool.imap(process_item, items), total=end - start))

    with open(f"{save_name}.json", "w") as f:
        json.dump(results, f)

    api.create_repo(f"dmitriihook/{save_name}", repo_type="dataset")
    api.upload_file(
        repo_id=f"dmitriihook/{save_name}",
        path_or_fileobj=f"{save_name}.json",
        path_in_repo=f"{save_name}.json",
        repo_type="dataset"
    )

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--start", type=int, default=600)
parser.add_argument("--end", type=int, default=1000)
parser.add_argument("--n_threads", type=int, default=20)
parser.add_argument("--save_name", type=str, default="blocksworld-big-step-labels-5")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.start, args.end, args.n_threads, args.save_name)

    

    
    