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

prompt_template_6 = """You will be given a state description in natural language.
Parse it into a valid JSON format.
If the input does not contain all 6 blocks (A, B, C, D, E, F), parse only the blocks that are present.
Otherwise, return the "state" type with the blocks in the correct order.
Follow the format in the examples below. Answer in the correct JSON format.

Input:
- F (table) → B → D → C
- A (table)
- E (table)

Goal stacks:

- D (table) → B → C → E → A → F

Output:
{{
    "blocks": [["C", "D", "B", "F"], ["A"], ["E"]]
}}

Input:
- F → B (since D was on top of B, but now D is picked up)
- C is on the table
- A and E are on the table
- D is in hand

Output:
{{
    "blocks": [["B", "F"], ["C"], ["A"], ["E"], ["D"]]
}}

Input:
- D → B → C → E

- F on the table, clear.

- A on the table, clear.

Then, I need to stack A on E. So, pick up A and stack it on E.

Now, the stacks

Output:
{{
    "blocks": [["E", "C", "B", "D"], ["A"], ["F"]]
}}

Input:
{state}

Output:
"""

prompt_template = """You will be given a state description in natural language.
Parse it into a valid JSON format.
If the input does not contain all 4 blocks (A, B, C, D), parse only the blocks that are present.
Stacks should be from the top block to the bottom block. You should only write 4 blocks total.
Otherwise, return the "state" type with the blocks in the correct order.
Follow the format in the examples below. Answer in the correct JSON format.

Input:
Initial:
- A (on table) with B on top.
- D (on table) with C on top.

Goal:
- A is on D.
- B is on C.
- D is on B.

Wait, that seems impossible because

Output:
{{
    "blocks": [["B", "A"], ["C", "D"]]
}}

Input:
- B has D on top.
- D has A on top.
- C is on table.

But the goal is to have B on top of C, and D on top of B. So I need to adjust.

Wait, maybe I should

Output:
{{
    "blocks": [["A", "D", "B"], ["C"]]
}}

Input:
- Block A is on top of Block C.
- Block D is on top of Block A.
- Block B is on the table.
- Block C is on the table.

Wait, that seems contradictory because if Block C is on the table and

Output:
{{
    "blocks": [["D", "A", "C"], ["B"]]
}}

Input:
- B (table) -> D -> A -> C
- Hand is empty.

Goal:

- B (table) -> C -> D -> A

So, the plan is:

Output:
{{
    "blocks": [["C", "A", "D", "B"]]
}}

Input:
- B -> D

- A on table

- C on table

After step 7: Holding A.

After step 8: A is on D, so stack is 

Output:
{{
    "blocks": [["D", "B"], ["C"], ["A"]]
}}

Input:
{state}

Output:
"""


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
