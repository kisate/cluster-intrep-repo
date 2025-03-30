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

prompt_template_high = """You will be given an reasoning chain from an LLM solving a planning problem. 
It can be roughly divided into the following parts:
1. Understanding the initial state and the goal state.
2. Exploration: the LLM explores, what the actions are and what the effects of these actions are. The model also investigates the difference between the current state and the goal state. This section may contain some short sequences of actions, but won't have the full plan.
3. Plan generation and verification: the LLM generates a plan and verifies it. Actions sequences in this section will be much longer than in the exploration section. The model will also reason about the plan and its correctness. It will also try to write new plans if the current one is incorrect. This section starts more than halfway through the reasoning chain. Plans in this section should contain more than 6 actions. If they are shorter, this is still the exploration section.
4. Final plan formulation and verification: the LLM will write the final plan and verify it.

Please identify the parts of the reasoning chain that contain the start of the plan generation and verification and the start of the final plan formulation and verification. Write at least 100 words for each extract.
Write them down in the following JSON format:
{{
    "plan-generation-verification": "Start of the plan generation and verification",
    "final-plan-formulation-verification": "Start of the final plan formulation and verification"
}}

Reasoning chain:
{trace}
"""


prompt_template = prompt_template_high

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
            max_tokens=8192 * 6,
            response_format={ "type": "json_object" },
        )

        label = json.loads(response.choices[0].message.content)

        return label
    
    return gen_label(trace)

def process_item(item):
    text = item["generation"]

    think_token = "</think>"

    if think_token not in text:
        label = None
    else:
        cot = text.split(think_token)[0]
        # cot = cot.replace("\n\n", "\n")
        try:
            label = thread_fn(cot)
        except Exception as e:
            label = None
            print(e)

    return {
        "index": item["index"],
        "label": label
    }

def main(start, end, n_threads, save_name):
    blocksworld_type = "mystery-3-24k"
    dataset = load_dataset(f"dmitriihook/qwq-32b-planning-{blocksworld_type}")["train"]

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
parser.add_argument("--end", type=int, default=400)
parser.add_argument("--n_threads", type=int, default=30)
parser.add_argument("--save_name", type=str, default="blocksworld-mystery-3-qwq-reasoning-parts-exploration")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.start, args.end, args.n_threads, args.save_name)

    

    
    