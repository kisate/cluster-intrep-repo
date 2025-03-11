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
You have to extract the blocks that the agent wants to move in the step. You should extract them in the order they are mentioned in the step.
If there were no actions considered in the step, return empty array.
You should only write blocks, when the agent directly considers moving them. 

Follow the format in the examples below. Answer in the correct JSON format.

Reasoning step:
But the goal also requires B to be on top of C. So I need to stack B on C. But B is now under D and A. So I need to unstack D and A first.

Output:
{{"blocks": ["B", "D"]}}

Reasoning step:
1. Move A off of D. Since A is clear, I can unstack A from D. Then put A down somewhere, maybe on the table.

Output:
{{"blocks": ["A"]}}

Reasoning step:
- Stack 1: B (top) -> C (bottom)
- Stack 2: D (top) -> A (bottom)

Output:
{{"blocks": []}}

Reasoning step:
The goal is to have A at the bottom, then C, then D, then B on top. So the final stack should be A-C-D-B. But wait, that's not possible because Block C is on the table initially, and I can't move it unless I unstack it. But Block C is under D, which is under B, which is under A. So I need to move A, B, D, and C around.

Output:
{{"blocks": ["C", "A", "B", "D"]}}

Reasoning step:
First, I need to move Block A somewhere, but it's on the table. Maybe I can pick it up and stack it on top of C later. But right now, Block C is under D and B, so I can't access it directly. I need to unstack D and B first.

Output:
{{"blocks": ["A", "D", "B"]}}

Reasoning step:
Wait, that can't be right because if D is on top of B, and B is on top of C, then D would be on top of B, which is on top of C. But A is on top of D, so the stack would be A on D on B on C. But that would mean C is at the bottom, then B, then D, then A. But the initial state has C on D, which is on the table. So I need to move things around.

Output:
{{"blocks": []}}

Reasoning step:
So, the stack would be A -> C -> B -> D. But D is initially on the table, so how can D be on top of B? That would require moving D on top of B, but D is currently under A. So I need to move D from under A to on top of B.

Output:
{{"blocks": ["D"]}}

Reasoning step:
First, I need to get A on top of D. Currently, A is on the table with B on top. So I need to move B off A first. Since B is clear, I can unstack it. So step 1: unstack B from A.

Output:
{{"blocks": ["A", "B]}}

Reasoning step:
First, I need to get Block A on top of C. But C is currently under B. So I need to move B somewhere else. Since B is on top of C, I can unstack B from C. But to do that, I need to pick up B, but B is clear, so I can unstack it. 

Output:
{{"blocks": ["A", "B"]}}

Reasoning step:
{step}

Output:
"""

client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
)


from pydantic import BaseModel

class ExtractedActions(BaseModel):
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

            label = json.loads(response.choices[0].message.content)
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

        if label is not None and len(label["blocks"]) > 0:
            break
    return group

def main(start, end, n_threads, save_name):
    blocksworld_type = "6-blocks-big"
    dataset = load_dataset(f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type}")["train"]

    dataset = dataset.add_column("index", [i for i in range(len(dataset))])
    items = dataset.select(range(start, end))
    with ThreadPool(n_threads) as pool:
        results = list(tqdm(pool.imap(process_item, items), total=end - start))

    with open(f"{save_name}.json", "w") as f:
        json.dump(results, f)   


    try:
        api.create_repo(f"dmitriihook/{save_name}", repo_type="dataset")
    except Exception as e:
        print(e)
    
    api.upload_file(
        repo_id=f"dmitriihook/{save_name}",
        path_or_fileobj=f"{save_name}.json",
        path_in_repo=f"{save_name}.json",
        repo_type="dataset"
    )

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=5000)
parser.add_argument("--n_threads", type=int, default=20)
parser.add_argument("--save_name", type=str, default="blocksworld-6-blocks-actions-first")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.start, args.end, args.n_threads, args.save_name)

    

    
    