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

You will be given a reasoning trace from the blocksworld problem solution. 
You have to extract the final plan from the reasoning trace. You will also have to extract the phrase, from which this plan starts.

Follow the format in the examples below. Answer in the correct JSON format.
You should only use the four possible actions: pick up a block, put down a block, stack a block on top of another block, and unstack a block from another block.
Each action should be followed by the correct amount of blocks. If there is any error in the input, extract only actions before the error.

Reasoning trace:
Step 15: Pick up F.

Step 16: Stack F on E. Now, F is on E, which is on D, which is on C, which is on B.

Finally, stack A on F.

Step 17: Pick up A.

Step 18: Stack A on F. Now, A is on F, which is on E, which is on D, which is on C, which is on B.

So the final stack is B (bottom) - C - D - E - F - A (top).

Let me check if all the goal conditions are met:

- A is on F: Yes, A is on F.
- C is on B: Yes, C is on B.
- D is on C: Yes, D is on C.
- E is on D: Yes, E is on D.
- F is on E: Yes, F is on E.

Yes, all conditions are met.

But wait, in the initial state, F was on B, so when I unstacked F from B, I had to put it down, then stack it on E later. That makes sense.

Let me make sure I didn't miss any steps. Let me go through the plan again:

1. Unstack E from D.
2. Put down E.
3. Unstack D from A.
4. Put down D.
5. Unstack A from C.
6. Put down A.
7. Unstack F from B.
8. Put down F.
9. Pick up C.
10. Stack C on B.
11. Pick up D.
12. Stack D on C.
13. Pick up E.
14. Stack E on D.
15. Pick up F.
16. Stack F on E.
17. Pick up A.
18. Stack A on F.

Yes, that seems correct. Each step follows the rules: only one block at a time, hand is empty before each pick up or unstack, and each block is clear when unstacked or picked up.

I think this plan should work.

Output:
{{
  "starting_phrase": "Let me go through the plan again:",
  "final_plan": [
    ["unstack", "E", "D"],
    ["put down", "E"],
    ["unstack", "D", "A"],
    ["put down", "D"],
    ["unstack", "A", "C"],
    ["put down", "A"],
    ["unstack", "F", "B"],
    ["put down", "F"],
    ["pick up", "C"],
    ["stack", "C", "B"],
    ["pick up", "D"],
    ["stack", "D", "C"],
    ["pick up", "E"],
    ["stack", "E", "D"],
    ["pick up", "F"],
    ["stack", "F", "E"],
    ["pick up", "A"],
    ["stack", "A", "F"]
  ]
}}

Reasoning trace:
 which is on top of B. So the final stack would be B -> A -> D -> C -> E -> F.

But wait, the initial state has A on F, which is on the table. So I need to move A from F to B, and then build up the stack from there.

Let me list the initial stacks:

- Stack 1: F (on table) with A on top.
- Stack 2: C (on table) with D on top.
- Blocks B, E are on the table.

So, the initial stacks are:

- F (table) -> A
- C (table) -> D
- B (table)
- E (table)

The goal is to have:

- B (table) -> A -> D -> C -> E -> F

So, I need to move A from F to B, then move D from C to A, then move C from table to D, then E from table to C, and finally F from table to E.

But wait, initially, F is under A, so I need to unstack A from F first. But F is on the table, so to unstack A, I need to pick up A, but A is clear, so that's possible.

Let me outline the steps:

1. Unstack A from F. Now, A is in hand, F is on the table, clear.
2. Put down A on B. Now, A is on B, which is on the table. So stack is B -> A.
3. Now, I need to get D from C. So unstack D from C. Now, D is in hand, C is on the table, clear.
4. Stack D on A. Now, stack is B -> A -> D.
5. Now, I need to get C from the table. Pick up C.
6. Stack C on D. Now, stack is B -> A -> D -> C.
7. Now, pick up E from the table.
8. Stack E on C. Now, stack is B -> A -> D -> C -> E.
9. Finally, pick up F from the table.
10. Stack F on E. Now, the final stack is B -> A -> D -> C -> E -> F.

Wait, but in the initial state, F is on the table with A on top. So after step 1, F is on the table, clear. Then in step 2, A is placed on B. Then step 3, D is on the table, and C is on the table. Then step 4, D is stacked on A. Then step 5, pick up C, stack on D. Then E is picked up and stacked on C, and finally F is stacked on E.

But let me check if all the blocks are clear when needed.

After step 1: A is picked up, F is clear.

After step 2: A is on B, so B is no longer clear, but that's okay because we don't need to move B anymore.

After step 3: D is picked up from C, so C is clear.

After step 4: D is on A, so A is no longer clear, but we don't need to move A anymore.

After step 5: C is picked up, clear.

After step 6: C is on D, so D is no longer clear.

After step 7: E is picked up, clear.

After step 8: E is on C, so C is no longer clear.

After step 9: F is picked up, clear.

After step 10: F is on E, so E is no longer clear.

This seems to satisfy all the conditions. Let me make sure I didn't miss any steps or violate any rules.

Wait, in the initial state, Block D is on top of Block C, which is on the table. So to unstack D, I need to have D clear. Since D is on top of C, and C is on the table, D is clear because nothing is on top of it. So unstacking D is allowed.

Similarly, after moving D to A, then C is on the table, so I can pick it up and stack on D.

E is on the table, so I can pick it up and stack on C.

F is on the table, so I can pick it up and stack on E.

I think this plan works. Let me write it out step by step.

Output:
{{
  "starting_phrase": "Let me outline the steps:",
  "final_plan": [
    ["unstack", "A", "F"],
    ["put down", "A", "B"],
    ["unstack", "D", "C"],
    ["stack", "D", "A"],
    ["pick up", "C"],
    ["stack", "C", "D"],
    ["pick up", "E"],
    ["stack", "E", "C"],
    ["pick up", "F"],
    ["stack", "F", "E"]
  ]
}}

Reasoning trace:
{trace}

Output:
"""

client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
)



def thread_fn(trace):
    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(1))
    def gen_label(trace):
        prompt = prompt_template.format(trace=trace)
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=1000,
            response_format={ "type": "json_object" },
        )

        label = response.choices[0].message.content

        return label
    
    return gen_label(trace)

def process_item(item):
    text = item["generation"]

    think_token = "</think>"

    if think_token not in text:
        label = None
    else:
        cot = text.split(think_token)[0]
        cot = cot[-3000:]

        try:
            label = thread_fn(cot)
            label = json.loads(label)
        except Exception as e:
            print(e)
            label = None

    return {
        "index": item["index"],
        "label": label
    }

def main(start, end, n_threads, save_name):
    blocksworld_type = "4-blocks"
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
parser.add_argument("--end", type=int, default=1500)
parser.add_argument("--n_threads", type=int, default=20)
parser.add_argument("--save_name", type=str, default="blocksworld-4-final-labels")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.start, args.end, args.n_threads, args.save_name)

    

    
    