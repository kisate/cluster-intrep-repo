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

prompt_template_high = """Please split the following reasoning chain of an LLM solving a blocksworld problem into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviors indicated by the labels. 
Available labels:

"state-understanding" → The model parses and encodes the configuration of blocks, including both initial and goal states.
"plan-formulation" → The model develops a sequence of actions to transform the initial state into the goal state, identifying necessary subgoals.
"constraint-reasoning" → The model applies the domain rules to determine which actions are valid, checking whether preconditions are satisfied (e.g., "To unstack F from B, F must be clear and on top of B").
"inconsistency-detection" → The model identifies apparent contradictions or misunderstandings in its interpretation of the problem statement (e.g., "Wait, that can't be right. If E is on C and A is on E, how can A also be on C?").
"plan-evaluation" → The model critically examines its plan, identifying potential errors and making necessary revisions.
"alternative-exploration" → The model considers different approaches or action sequences that might achieve the goal.

The reasoning chain to analyze: {trace}
Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
"""

prompt_template_high = """Please split the following reasoning chain of an LLM solving a blocksworld problem into annotated parts using labels and the following format ["label"]...["end-section"]. Try covering as many contiguous lines as possible with a single label.
Available labels:

"state-understanding" → The model parses and encodes the configuration of blocks, including both initial and goal states.
"plan-formulation" → The model develops a sequence of actions to transform the initial state into the goal state, identifying necessary subgoals. Also covers plan fixes.
"plan-evaluation" → The model critically examines its plan and identifies potential errors.
"plan-finalizing" → The model repeats an already written plan, but in a new format. 

The reasoning chain to analyze: {trace}
Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
"""

prompt_template_high = """Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels. Try covering as many contiguous lines as possible with a single label. You should cover as much of the reasoning as possible. Leave just the several first and several last lines of the sections, since they will be too big. Available labels:

goal-and-initial-state-understanding -> The model is parsing the initial conditions, constraints and goals of the planning problem.
exploration -> The model explores action possibilities, analyzes dependencies between conditions and effects, and traces causal chains to achieve goals.
plan-proposal -> The model proposes specific action sequences that might solve the problem.
plan-verification -> The model checks whether each step in a proposed plan satisfies the necessary preconditions and achieves the expected effects.
final-plan-formulation -> After testing various approaches, the model settles on what it believes is the correct plan.
final-plan-verification -> The model performs a comprehensive check to ensure the entire plan is valid and achieves all goals.

The reasoning chain to analyze: {trace}
Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out."""

prompt_template_low = """Please split the following reasoning chain of an LLM solving a blocksworld problem into annotated parts using labels and the following format ["label"]...["end-section"]. Try to combine contiguous same-label lines into a single label. Try to be as fine-grained as possible at the same time. No tag should cover more than 5 lines. You should cover the entire reasoning chain.
Available labels:

"stack-visualization" → The model explicitly writes out the stacks, either during the initial problem understanding or during the plan formulation. It can use a variety of formats, such as "A on B" or "A → B". You should label separate state (initial, goal) visualizations separately. Should just cover the visualization, if it includes any comparison or reasoning, use "comparative-analysis" instead.

"comparative-analysis" → The model compares the current state of the blocks with the goal state, identifying differences and determining how to resolve them. This label SHOULD NOT include any action planning or recursive search, just the state comparison. If the model start writing out a plan, this is no longer a "comparative-analysis" Example: "1. A is on top of B. So B must be under A. But currently, B is under F, which is under A. Wait, in the initial state, F is on A, so A is under F. So to have A on top of B, we need to move A so that it's above B. But B is currently on F, which is on A. So that's a circular dependency? Wait, maybe I need to restructure the stacks."

"recursive-search" → The model wants to peform an action, but realizes it needs to perform another action first. This label should be used when the model is recursively searching for the next action to take. Should only cover the action search. Plan formulation should be annotated with "plan-formulation". Example: "To move A, we need to unstack F from A. But F is on A, so to unstack F from A, F must be clear. Currently, F has B on it. So first, we need to unstack B from F. But B is under D and C. So first, we need to unstack C and D from B."

"plan-formulation" → The model writes a single action or a sequence of actions. A plan formulation should only include consecutive actions. If it stops to check constraints use the "constraint-analysis" label. This label should only include ready-made plans. Any action search should be labeled with "recursive-search". Example: "So steps:

1. Unstack C from F → hold C.

2. Put down C → now C is on table, clear.

3. Unstack D from E → hold D.

4. Stack D onto C → D is now on C. Now, D is clear (since nothing is on top of it). Wait, but after stacking, D is on top of C, so C is under D. So D is clear, but C is not.
"

"constraint-analysis" → The model analyses the contstraints for the planned actions.

The reasoning chain to analyze: {trace}
Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
"""

prompt_template_low = """Please split the following reasoning chain of an LLM solving a blocksworld problem into annotated parts using labels and the following format ["label"]...["end-section"]. Try to be as fine-grained as possible at the same time. No tag should cover more than 5 lines. You should cover the entire reasoning chain.
Available labels:

"initial-state-understanding" → The model the model expicitly writes the initial state of the blocks. It can use a variety of formats, such as "A on B" or "A → B". You should label separate state visualizations separately. Should just cover the visualization, if it includes any comparison or reasoning enclose the reasoning part in "comparative-analysis".

"goal-state-understanding" → The model the model expicitly writes the goal state of the blocks. It can use a variety of formats, such as "A on B" or "A → B". You should label separate state visualizations separately. Should just cover the visualization, if it includes any comparison or reasoning enclose the reasoning part in "comparative-analysis".

"state-tracking" → The model analyzes the state after applying some actions. It can use a variety of formats, such as "A on B" or "A → B". You should label separate state visualizations separately. Should just cover the visualization, if it includes any comparison or reasoning enclose the reasoning part in "comparative-analysis". Example:
["state-tracking"]Wait, let's go step by step.

Initial state:

Stack: A (table) - F - E - C - B (top). D is on table, clear.
["end-section"] ["stating-actions"] Step 1: Unstack B from C. So now, B is in hand, and the stack becomes A-F-E-C (with C now clear?), and B is held. Then, stack B onto D. So now, D has B on top. Hand is empty again. ["end-section"]

["state-tracking"]Now, the stacks are:

D-B (stack), and A-F-E-C (stack on A). Also, C is now clear? Because after unstacking B, C was the top of its stack, so yes, C is clear again.["end-section"]

"comparative-analysis" → The model compares the current state of the blocks with the goal state, identifying differences and determining how to resolve them. This label SHOULD NOT include any action planning or recursive search, just the state comparison. If the model start writing out a plan, this is no longer a "comparative-analysis". Example: "1. A is on top of B. So B must be under A. But currently, B is under F, which is under A. Wait, in the initial state, F is on A, so A is under F. So to have A on top of B, we need to move A so that it's above B. But B is currently on F, which is on A. So that's a circular dependency? Wait, maybe I need to restructure the stacks." 

"recursive-search" → The model wants to peform an action, but realizes it needs to perform another action first. This label should be used when the model is recursively searching for the next action to take. Should only cover the action search. Plan formulation should be annotated with "stating-actions". This label can only cover several lines at a time. If it covers more, split it up. Example 1: "To move A, we need to unstack F from A. But F is on A, so to unstack F from A, F must be clear. Currently, F has B on it. So first, we need to unstack B from F. But B is under D and C. So first, we need to unstack C and D from B."

Example 2: ["recursive-search"]Wait, but the goal requires D to be on F. So maybe after moving D, we can stack it on F later. Let's see.

But first, after unstacking D, we need to put it down. Let's say we put D down on the table. Then D is on the table, clear. Then we can proceed. ["end-section"]

["stating-actions"]But let's see the plan step by step.

Let me try to outline possible steps:

1. Unstack D from A. ["end-section"] ["state-tracking"](Now D is held, A is clear, and the stack is E-F-C-B-A, with D in hand.)["end-section"]

["stating-actions"]2. Put down D. ["end-section"] ["state-tracking"](Now D is on table, clear. Hand is empty.)

Now, the stack is E-F-C-B-A, and D is on the table.["end-section"]

["recursive-search"]Next, we need to get A off of B. To do that, we need to unstack A from B. But A is clear now (since D was on it before, but now D is moved). Wait, after step 1, when we unstacked D from A, A became clear. So yes, A is clear. So:["end-section"]


"stating-actions" → The model writes a single action or a sequence of actions. This should only include consecutive actions. If it stops to check constraints use the "constraint-analysis" label. If it writes out the state after an action, you should switch to "state-tracking" temporarily. This label should only include ready-made plans. Any action search should be labeled with "recursive-search". Example: "["stating-actions"]Step 10: Stack C on D.["end-section"] ["state-tracking"]Now, the stack is B -> D -> C.["end-section"] 
"

"constraint-analysis" → The model analyses the contstraints for the planned actions.

The reasoning chain to analyze: {trace}
Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
"""

prompt_template_low = """Please split the following reasoning chain of an LLM solving a blocksworld problem into annotated parts using labels and the following format ["label"]...["end-section"]. You don't need to cover the entire reasoning chain, just the parts that are relevant. Try to be as fine-grained as possible at the same time. No tag should cover more than 5 lines. Available labels:

"initial-state-understanding" → The model the model expicitly writes the initial state of the blocks. It can use a variety of formats, such as "A on B" or "A → B". You should label separate state visualizations separately. Should just cover the visualization, if it includes any comparison or reasoning enclose the reasoning part in "comparative-analysis".

"goal-state-understanding" → The model the model expicitly writes the goal state of the blocks. It can use a variety of formats, such as "A on B" or "A → B". You should label separate state visualizations separately. Should just cover the visualization, if it includes any comparison or reasoning enclose the reasoning part in "comparative-analysis".

"state-tracking" → The model analyzes the state after applying some actions. It can use a variety of formats, such as "A on B" or "A → B". You should label separate state visualizations separately. Extract with this tag should only start with state visualization. It can't contain any step mentions first. Example:

["state-tracking"]Now, the stacks are:

D-B (stack), and A-F-E-C (stack on A). Also, C is now clear? Because after unstacking B, C was the top of its stack, so yes, C is clear again.["end-section"]

"action-exploration" → The model considers a single action or a sequence of actions. This should only include consecutive actions. If the model starts thinking about action consequences, you should break this label. If it writes out the state after an action, you should switch to "state-tracking" temporarily. Example: "["action-exploration"]Step 10: Stack C on D.["end-section"] ["state-tracking"]Now, the stack is B -> D -> C.["end-section"] 

The reasoning chain to analyze: {trace}
Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
"""



prompt_template = prompt_template_low

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
        # cot = cot.replace("\n\n", "\n")
        try:
            label = thread_fn(cot)
        except Exception as e:
            label = None

    return {
        "index": item["index"],
        "label": label
    }

def main(start, end, n_threads, save_name):
    blocksworld_type = "6-blocks"
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
parser.add_argument("--end", type=int, default=2500)
parser.add_argument("--n_threads", type=int, default=30)
parser.add_argument("--save_name", type=str, default="blocksworld-6-blocks-qwq-reasoning-parts-low-v4")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.start, args.end, args.n_threads, args.save_name)

    

    
    