from collections import defaultdict
from typing import Optional

import re
import numpy as np
from tqdm.auto import tqdm

def stacks_to_pairs(stacks: list[list[str]]) -> tuple[dict, dict, Optional[str]]:
    above = {}
    below = {}

    for stack in stacks:
        for i, block in enumerate(stack):
            if i == 0:
                above[block] = "sky"
            else:
                above[block] = stack[i - 1]
                below[stack[i - 1]] = block
            below[block] = "table"
        
    return above, below

def check_if_block(block: str, n_blocks: int) -> bool:
    return block in [chr(65 + i) for i in range(n_blocks)]

def check_stacks(stacks: list[list[str]], n_blocks: int) -> bool:
    blocks = set()
    for stack in stacks:
        for block in stack:
            if not check_if_block(block, n_blocks):
                return False
            blocks.add(block)
    return len(blocks) == n_blocks

def make_labels_dict(labels_dataset, n_blocks):   
    labels_dict = defaultdict(dict)
    for item in labels_dataset:
        idx = item["idx"]
        line_n = item["line_n"]
        parsed = item["parsed"]

        if parsed is None:
            continue

        if "blocks" not in parsed:
            continue
        
        if not check_stacks(parsed["blocks"], n_blocks):
            continue

        labels_dict[idx][line_n] = item

    return labels_dict

def extract_actions(row):
    generation = row["generation"]
    if "[PLAN]" not in generation:
        return None
    if "[PLAN END]" not in generation:
        return None
    
    plan_start = generation.index("[PLAN]") + len("[PLAN]")
    plan = generation[plan_start:].strip()
    plan = plan.split("[PLAN END]")[0].strip()
    actions = plan.split("\n")

    return actions


def parse_block_actions(commands):
    actions = ["unstack", "put down", "pick up", "stack"]
    parsed_commands = []

    for command in commands:
        for action in actions:
            if command.startswith(action):
                blocks = re.findall(r'Block [A-Z]', command)
                blocks = [block.split()[-1] for block in blocks]  # Extract only the letter
                parsed_commands.append((action, blocks))
                break

    return parsed_commands


def parse_blocks(text):
    initial_state = []
    goal_state = []
    
    # Extract the initial conditions and goal state
    initial_match = re.search(r'As initial conditions I have that:(.*?)My goal is for the following to be true:', text, re.DOTALL)
    goal_match = re.search(r'My goal is for the following to be true:(.*?)\n\n', text, re.DOTALL)

    if initial_match:
        initial_conditions = re.findall(r'Block [A-Z] is on top of Block [A-Z]', initial_match.group(1))
        init_table_blocks = re.findall(r'Block ([A-Z]) is on the table', initial_match.group(1))
        initial_state = process_conditions(initial_conditions)

    
    if goal_match:
        goal_conditions = re.findall(r'Block [A-Z] is on top of Block [A-Z]', goal_match.group(1))
        goal_table_blocks = re.findall(r'Block ([A-Z]) is on the table', goal_match.group(1))
        goal_state = process_conditions(goal_conditions)

    
    return (initial_state, init_table_blocks), (goal_state, goal_table_blocks)

def process_conditions(conditions):
    pairs = {}
    
    for cond in conditions:
        block, below = re.findall(r'Block ([A-Z])', cond)
        pairs[block] = below
    
    return pairs



def collect_all_blocks(initial_state):
    all_blocks = list(initial_state[0].keys())
    all_blocks.extend(initial_state[1])
    all_blocks.extend(initial_state[0].values())
    return list(set(all_blocks))

def state_to_pairs(state, all_blocks):
    pairs, _ = state
    below = {}

    for block, below_block in pairs.items():
        below[block] = below_block

    for block in all_blocks:
        if block not in below:
            below[block] = "table"

    above = {}

    for block, below_block in below.items():
        if below_block != "table":
            above[below_block] = block

    for block in all_blocks:
        if block not in above:
            above[block] = "sky"
    
    return above, below


def state_compare(above1, below1, above2, below2):
    for block in above1:
        if above1[block] != above2[block]:
            return False
    for block in below1:
        if below1[block] != below2[block]:
            return False
        
    return True


row_ns = {
    4: 1500,
    6: 2000
}

parsed_datasets = {
    4: "blocksworld-4-self-probing-parsed-big-v2.json",
    6: "blocksworld-6-self-probing-parsed-big.json"
}

blocksworld_type = {
    4: "4-blocks",
    6: "6-blocks-big",
}


def make_data_to_process(dataset, n_rows, eval_results, answer_type, n_blocks, labels_dict, take_prob, tokenizer):
    data_to_process = []
    for idx, row in enumerate(tqdm(dataset.select(range(n_rows)))):
        if eval_results[idx]["llm_correct"] and answer_type == "incorrect":
            continue
        if not eval_results[idx]["llm_correct"] and answer_type == "correct":
            continue
        query = row["query"]
        stmt = query.split("[STATEMENT]")[-1].strip()
        initial_state, goal_state = parse_blocks(stmt)
        all_blocks = collect_all_blocks(initial_state)
        i_above, i_below = state_to_pairs(initial_state, all_blocks)
        g_above, g_below = state_to_pairs(goal_state, all_blocks)

        generation = row["generation"]

        text = ""

        # prompts = []

        for line_n, line in enumerate(generation.split("\n\n")):
            _text = text
            text = text + line + "\n\n"
            if n_blocks == 4 and line_n < 10 or len(line) < 30:
                continue
            if n_blocks == 6 and line_n < 20 or len(line) < 50:
                continue
            # if line_n >= 20 and len(line) >= 50:
            #     continue

            if line_n not in labels_dict[idx]:
                continue

            above, below = stacks_to_pairs(
                labels_dict[idx][line_n]["parsed"]["blocks"])
            if state_compare(i_above, i_below, above, below):
                if np.random.rand() > take_prob:
                    continue
            if state_compare(g_above, g_below, above, below):
                if np.random.rand() > take_prob:
                    continue

            # _text = text + "Now, the stacks are:\n\n-"

            _query = row["distilabel_metadata"]["raw_input_text_generation_0"][0]

            messages = [
                _query,
                {"role": "assistant", "content": _text}
            ]
            tokens_pre = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False)[:-1]

            messages = [
                _query,
                {"role": "assistant", "content": text}
            ]

            tokens_post = tokenizer.apply_chat_template(
                messages, tokenize=True, add_special_tokens=False)[:-1]

            data_to_process.append({
                "idx": idx,
                "line_n": line_n,
                "tokens_pre": tokens_pre,
                "tokens_post": tokens_post,
                "above": above,
                "below": below
            })


act2int = {
    "put down": 0,
    "pick up": 1,
    "stack": 2,
    "unstack": 3
}


def block2int(block, n_blocks):
    if block == "table":
        return n_blocks
    if block == "sky":
        return n_blocks + 1
    block_n = ord(block) - ord("A")
    assert block_n < n_blocks
    assert block_n >= 0

    return block_n


def int2block(i, n_blocks):
    if i == n_blocks:
        return "table"
    if i == n_blocks + 1:
        return "sky"

    return chr(i + ord("A"))

def collect_block_positions(items, top_block, bottom_block, block_positions):
    new_items = []

    for item in items:
        if item["line_n"] > 60:
            continue
        prev_pos = len(item["tokens_pre"])
        post_pos = len(item["tokens_post"])

        _block_positions = block_positions[item["idx"]]

        top_block_pos = _block_positions[top_block]
        bottom_block_pos = _block_positions[bottom_block]

        top_block_pos = top_block_pos[top_block_pos < post_pos]
        bottom_block_pos = bottom_block_pos[bottom_block_pos < post_pos]

        top_block_pos = top_block_pos[top_block_pos > prev_pos]
        bottom_block_pos = bottom_block_pos[bottom_block_pos > prev_pos]

        if len(top_block_pos) == 0 or len(bottom_block_pos) == 0:
            continue

        new_items.append({
            "idx": item["idx"],
            "line_n": item["line_n"],
            "top_positions": top_block_pos,
            "bottom_positions": bottom_block_pos,
            "post_pos": post_pos,
            "above": item["above"],
            "below": item["below"]
        })

    return new_items
