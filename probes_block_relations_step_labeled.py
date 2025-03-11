
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from vllm import TokensPrompt
from vllm import LLM
from pathlib import Path
from argparse import ArgumentParser
import os
import torch
import json
import numpy as np

from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset

from utils import initialize_tokenizer, tokenize_blocksworld_generation, THINK_TOKEN
from stacks_utils import *

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

compute_dtype = torch.float
device = 'cuda'
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

tokenizer = initialize_tokenizer(model_id)


parser = ArgumentParser()

parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--probe_type", type=str, default="gru")
parser.add_argument("--answer_type", type=str, default="correct")

args = parser.parse_args()

n_blocks = args.n_blocks

probe_type = args.probe_type
answer_type = args.answer_type


dataset = load_dataset(
    f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type[n_blocks]}")["train"]


with open(parsed_datasets[n_blocks], "r") as f:
    labels_dataset = json.load(f)

# labels_dataset = load_dataset(f"dmitriihook/blocksworld-6-self-probing-parsed")["train"]


def load_dataset_from_file(domain_name, task_name):
    prompt_dir = Path(f"./cot-planning/results/{domain_name}/deepseek-32b/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)


task_name = "plan_generation_po"
domain_name = f"blocksworld_{n_blocks}_blocks"
eval_results = load_dataset_from_file(domain_name, task_name)["instances"]

eval_results = {x["dataset_idx"]: x for x in eval_results}


n_rows = row_ns[n_blocks]

take_prob = 0.5

labels_dict = make_labels_dict(labels_dataset, n_blocks)

data_to_process = make_data_to_process(dataset, n_rows, eval_results, answer_type, n_blocks, labels_dict, take_prob, tokenizer)


# batch_size = 100
if n_blocks == 4:
    data_to_process = data_to_process[:13500]


llm = LLM(model=model_id, task="reward", tensor_parallel_size=8)


batch_size = 200

last_hidden_states = []

for i in tqdm(range(0, n_rows, batch_size)):
    batch = dataset.select(range(i, min(i + batch_size, n_rows)))
    tokens = [tokenize_blocksworld_generation(
        tokenizer, row)[0] for row in batch]

    tokens = [TokensPrompt(prompt_token_ids=t) for t in tokens]

    output = llm.encode(tokens)

    for x in output:
        hs = x.outputs.data
        last_hidden_states.append(hs.cpu().to(torch.float16).numpy())


block_positions = []

all_blocks = [chr(65 + i) for i in range(n_blocks)]

for row in dataset.select(range(n_rows)):
    tokens = tokenize_blocksworld_generation(tokenizer, row)[0]
    row_block_positions = {}

    for block in all_blocks:
        block_token = tokenizer.encode(
            " " + block, add_special_tokens=False)[0]
        _block_positions = np.where(tokens == block_token)[0]
        row_block_positions[block] = _block_positions

    block_positions.append(row_block_positions)


n_prev_tokens = 30


def state_to_label(state):
    above, below, hand = state
    label = np.zeros((n_blocks * 2, ), dtype=np.int64)

    # return int(below["C"] == "B")

    for block, below_block in below.items():
        label[block2int(block, n_blocks)] = block2int(below_block, n_blocks)
    for block, above_block in above.items():
        label[block2int(block, n_blocks) +
              n_blocks] = block2int(above_block, n_blocks)

    return label


def state_to_label(state, top_block, bottom_block):
    above, below, hand = state
    top_block_stack = [top_block]
    while top_block_stack[-1] != "table":
        top_block_stack.append(below[top_block_stack[-1]])
    top_block_stack = top_block_stack[1:]

    return int(bottom_block in top_block_stack)


class StepProbeDataset(Dataset):
    def __init__(self, items, n_layer, top_block, bottom_block):
        self.items = collect_block_positions(
            items, top_block, bottom_block, block_positions)
        self.hidden_states = last_hidden_states
        self.n_layer = n_layer
        self.top_block = top_block
        self.bottom_block = bottom_block

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        if probe_type == "linear":
            top_positions = item["top_positions"]
            bottom_positions = item["bottom_positions"]

            top_representations = self.hidden_states[item["idx"]][top_positions].mean(
                axis=0)
            bottom_representations = self.hidden_states[item["idx"]][bottom_positions].mean(
                axis=0)

            # hidden_states = self.hidden_states[item["idx"]][post_pos-10:post_pos + 1]
            hidden_states = top_representations - bottom_representations
            above, below = item["above"], item["below"]
            # print(above, below)
            return {
                "input": hidden_states,
                "labels": state_to_label((above, below, None), self.top_block, self.bottom_block)
            }
        elif probe_type == "gru":
            post_pos = item["post_pos"]
            hidden_states = self.hidden_states[item["idx"]
                                               ][post_pos-n_prev_tokens:post_pos + 1]

            above, below = item["above"], item["below"]
            return {
                "input": hidden_states,
                "labels": state_to_label((above, below, None), self.top_block, self.bottom_block)
            }


class StepProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_blocks):
        super().__init__()
        # self.fc = torch.nn.Linear(input_size, hidden_size)
        # self.fc2 = torch.nn.Linear(hidden_size, n_blocks * (n_blocks + 2) * 2)
        # self.fc2 = torch.nn.Linear(input_size, n_blocks * (n_blocks + 2) * 2)
        self.fc2 = torch.nn.Linear(input_size, 2)
        # self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        # x = self.fc(x)
        # x = torch.nn.functional.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
        # return x.view(-1, n_blocks + 2, n_blocks * 2)


class GRUProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_blocks):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1])
        return x


training_data = data_to_process


train_test_split = 0.9
n_train = int(len(training_data) * train_test_split)

train_items = training_data[:n_train]
test_items = training_data[n_train:]


def collate_fn(batch):
    inputs = [torch.tensor(x["input"], dtype=compute_dtype) for x in batch]
    inputs = pad_sequence(inputs, batch_first=True,
                          padding_value=0, padding_side="left")
    labels = np.stack([x["labels"] for x in batch])
    labels = torch.tensor(labels, dtype=torch.int64)
    return {
        "input": inputs.to(device),
        "labels": labels.to(device)
    }


def train_probe(probe, train_dataset, test_dataset, patience=30):
    optimizer = Adam(probe.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)

    n_epochs = 500
    best_f1 = float('-inf')
    early_stop_counter = 0

    for epoch in range(n_epochs):
        probe.train()
        total_loss = 0
        n_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input = batch["input"].to(device).float()
            labels = batch["labels"].to(device)

            output = probe(input)

            # print(output.shape, labels.shape, input.shape)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch["input"])
            n_samples += len(batch["input"])

        avg_train_loss = total_loss / n_samples

        # Evaluation
        probe.eval()
        with torch.no_grad():
            # block_wise_hits = np.zeros((n_blocks * 2), dtype=np.int64)
            block_wise_hits = 0
            total = 0
            val_loss = 0
            all_preds = []
            all_labels = []

            for batch in test_loader:
                input = batch["input"].to(device).float()
                labels = batch["labels"].to(device)

                output = probe(input)

                loss = criterion(output, labels)
                val_loss += loss.item() * len(batch["input"])

                preds = output.argmax(dim=1)  # Assuming classification task
                hits = (preds == labels)

                block_wise_hits += hits.sum(dim=0).cpu().numpy()
                total += len(labels)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            block_wise_hits = block_wise_hits / total

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            # Compute F1 score block-wise
            # block_wise_f1 = np.zeros(n_blocks * 2)
            # for i in range(n_blocks * 2):
            #     block_wise_f1[i] = f1_score(all_labels[:, i], all_preds[:, i], average='macro')

            # avg_f1 = block_wise_f1.mean()
            avg_f1 = f1_score(all_labels, all_preds, average='macro')

            val_loss /= total

            print(
                f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Hits: {block_wise_hits.mean():.4f}, F1: {avg_f1:.4f}, Val Loss: {val_loss:.4f}")

            # Early Stopping Check
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return block_wise_hits, best_f1


run_results = []


for top_block in all_blocks:
    for bottom_block in all_blocks:
        if top_block == bottom_block:
            continue
        train_dataset = StepProbeDataset(
            train_items, 0, top_block, bottom_block)
        test_dataset = StepProbeDataset(test_items, 0, top_block, bottom_block)

        n_dim = 5120

        if probe_type == "linear":
            probe = StepProbe(n_dim, 500, n_blocks).to(device)
        elif probe_type == "gru":
            probe = GRUProbe(n_dim, 1000, n_blocks).to(device)

        block_wise_hits, best_f1 = train_probe(
            probe, train_dataset, test_dataset, patience=30)

        print(
            f"Top block: {top_block}, Bottom block: {bottom_block}, Hits: {block_wise_hits.mean():.4f}, F1: {best_f1:.4f}")
        run_results.append({
            "top_block": top_block,
            "bottom_block": bottom_block,
            "hits": block_wise_hits.mean(),
            "f1": best_f1
        })


with open(f"blocksworld-{n_blocks}-self-probing-step-results-{answer_type}-{probe_type}.json", "w") as f:
    json.dump(run_results, f)
