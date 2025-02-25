import openai
from dotenv import load_dotenv
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt
from multiprocessing.pool import ThreadPool
from tqdm.auto import tqdm
from huggingface_hub import HfApi
from pathlib import Path
import json

load_dotenv()

api = HfApi()

prompt_template = """You will be given a state description in natural language.
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

client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
)

def thread_fn(state):
    # @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(1))
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
    results = item["results"]
    new_results = []

    for x in results:
        new_text = x["new_text"]

        new_item = {
            "line_n": x["line_n"],
            "line": x["line"],
            "new_text": x["new_text"],
            "parsed": None
        }

        if new_text is None:
            new_results.append(new_item)
            continue
        try:
            parsed = thread_fn(new_text)
            parsed = json.loads(parsed)
        except Exception as e:
            print(e)
            parsed = None

        new_item["parsed"] = parsed
        new_results.append(new_item)

    return {
        "idx": item["idx"],
        "results": new_results
    }

save_path = Path(f"self_probing")
file_name = "self_probing_state_6_blocks_{part}.jsonl"
n_parts = 4


def main(start, end, n_threads, save_name):
    combined_dataset = []

    for part in range(n_parts):
        with open(save_path / file_name.format(part=part), "r") as f:
            for line in f:
                combined_dataset.append(json.loads(line))

    combined_dataset = combined_dataset[start:end]

    with ThreadPool(n_threads) as pool:
        results = list(tqdm(pool.imap(process_item, combined_dataset), total=end - start))

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
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=300)
parser.add_argument("--n_threads", type=int, default=20)
parser.add_argument("--save_name", type=str, default="blocksworld-6-self-probing-parsed")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.start, args.end, args.n_threads, args.save_name)

    

    
    
