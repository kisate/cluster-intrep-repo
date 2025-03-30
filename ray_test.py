from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from pathlib import Path
from argparse import ArgumentParser

import os
import ray
import time
import torch
import json
import numpy as np

from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset

from utils import initialize_tokenizer, tokenize_blocksworld_generation, THINK_TOKEN
from stacks_utils import *
from tqdm.auto import tqdm, trange


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper


ray.init(address="auto", namespace="blocksworld")


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

compute_dtype = torch.bfloat16
device = 'cuda'
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

tokenizer = initialize_tokenizer(model_id)



n_blocks = 6


parsed_datasets = {
    4: "blocksworld-4-blocks-actions-first.json",
    6: "blocksworld-6-blocks-actions-first.json"
}


cur_dir = Path(".").absolute()


def load_dataset_from_file(domain_name, task_name):
    prompt_dir = cur_dir / Path(f"./cot-planning/results/{domain_name}/deepseek-32b/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)
        

def load_datasets():
    dataset = load_dataset(
    f"dmitriihook/deepseek-r1-qwen-32b-planning-{blocksworld_type[n_blocks]}")["train"]
    
    task_name = "plan_generation_po"
    domain_name = f"blocksworld_{n_blocks}_blocks"
    eval_results = load_dataset_from_file(domain_name, task_name)["instances"]
    eval_results = {x["dataset_idx"]: x for x in eval_results}

    with open(cur_dir / "first_action_token_logits.json") as f:
        labels_dataset = json.load(f)

    labels_dataset = {
        int(k): v for k, v in labels_dataset.items()
    }

    return dataset, eval_results, labels_dataset


def make_labels_dict(labels_dataset, dataset):
    all_blocks = [
        chr(ord('A') + i) for i in range(n_blocks)
    ]

    n_rows = row_ns[n_blocks]
    n_rows = 5000

    labels_dict = defaultdict(dict)

    for idx, row in enumerate(dataset.select(range(n_rows))):
        generation = row["generation"]

        steps = generation.split("\n\n")[:50]

        if idx not in labels_dataset:
            continue

        for line_n, step in enumerate(steps):
            logits = []
            for block in all_blocks:
                logits.append(labels_dataset[idx]["block_logits"][block][line_n])

            logits = np.array(logits)
            smax = np.exp(logits) / np.sum(np.exp(logits))
            amax = np.argmax(logits)

            labels_dict[idx][line_n] = {
                "logits": logits,
                "probs": smax,
                "max_block": all_blocks[amax],
                "step": step
            }

    return labels_dict


# total_layers = model.config.num_hidden_layers
total_layers = 64


def make_data_to_process(dataset, labels_dict, n_rows, eval_results, answer_type, tokenizer):
    data_to_process = []
    for idx, row in enumerate(dataset.select(range(n_rows))):
        if eval_results[idx]["llm_correct"] and answer_type == "incorrect":
            continue
        if not eval_results[idx]["llm_correct"] and answer_type == "correct":
            continue
        generation = row["generation"]

        if idx not in labels_dict:
            continue

        # need to remove the eos token
        pos_start = len(tokenize_blocksworld_generation(tokenizer, row, "")[0][:-1])

        pos_pre = pos_start

        for line_n, line in enumerate(generation.split("\n\n")[:50]):
            if labels_dict[idx][line_n]["probs"].max() < 0.9:
                continue
            # if line_n < 25:
            #     continue
            line_tokens = tokenizer.tokenize("\n\n" + line + "\n\n")[1:]
            pos_post = pos_pre + len(line_tokens)

            data_to_process.append({
                "idx": idx,
                "line_n": line_n,
                "pos_pre": pos_pre,
                "pos_post": pos_post,
                "pos_start": pos_start,
                "logits": labels_dict[idx][line_n]["logits"],
                "label": labels_dict[idx][line_n]["max_block"],
            })
            pos_pre = pos_post


    return data_to_process


def process_data(items, n_blocks, max_tokens = 3200):
    new_items = []

    for item in items:

        block = item["label"]

        if item["pos_post"] >= max_tokens:
            continue

        try:
            label = block2int(block, n_blocks)
        except Exception as e:
            print(e)
            continue

        new_items.append({
            "pos_pre": item["pos_pre"],
            "pos_post": item["pos_post"],
            "pos_start": item["pos_start"],
            "label": label,
            "idx":  item["idx"],
            "line_n": item["line_n"],
            "logits": item["logits"],
            "label": label
        })

    return new_items
    


def collate_fn(batch):
    inputs = [torch.tensor(x) for x in batch["input"]]    
    masks = [torch.ones(x.shape[0], dtype=torch.bool) for x in inputs]
    # inputs = pad_sequence(inputs, batch_first=True,
    #                       padding_value=0, padding_side="left")
    # masks = pad_sequence(masks, batch_first=True,
    #                      padding_value=True, padding_side="left")

    inputs = torch.stack(inputs)
    masks = torch.stack(masks)

    labels = np.stack([x for x in batch["labels"]])
    labels = torch.tensor(labels, dtype=torch.int64)

    logits = np.stack([x for x in batch["logits"]])
    logits = torch.tensor(logits, dtype=torch.float32)
    return {
        "input": inputs.to(device),
        "labels": labels.to(device),
        "logits": logits.to(device),
        "mask": masks.to(device)
    }


class StepProbeDataset(Dataset):
    def __init__(self, items, n_layer, n_prev_tokens, shift_tokens, dataset_actor_name, n_blocks, batch_size):
        self.items = process_data(items, n_blocks)
        self.n_layer = n_layer
        self.n_blocks = n_blocks
        self.n_prev_tokens = n_prev_tokens
        self.shift_tokens = shift_tokens
        self.dataset_actor = ray.get_actor(dataset_actor_name)
        self.batch_size = batch_size

    def get_batch(self, idxs):
        items = [self.items[idx] for idx in idxs]

        _hidden_states = ray.get(self.dataset_actor.get_batch_layer.remote([item["idx"] for item in items], self.n_layer))

        inputs = []
        labels = []
        logits = []

        for item in items:
            pos_pre = item["pos_pre"]
            pos_post = item["pos_post"]
            pos_start = item["pos_start"]

            pos = pos_post

            label = item["label"]
            _logits = item["logits"]
            hidden_states = _hidden_states[item["idx"]]

            inputs.append(hidden_states[pos - self.shift_tokens - self.n_prev_tokens:pos - self.shift_tokens + 1])
            labels.append(label)
            logits.append(_logits)

        return {
            "input": inputs,
            "labels": labels,
            "logits": logits
        }
        

    def __len__(self):
        return len(self.items) // self.batch_size
    
    def __getitem__(self, idx):
        start_item_idx = idx * self.batch_size
        end_item_idx = min((idx + 1) * self.batch_size, len(self.items))

        batch = self.get_batch(range(start_item_idx, end_item_idx))

        batch = collate_fn(batch)

        return batch


class StepProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_blocks):
        super().__init__()
        # self.fc = torch.nn.Linear(input_size, hidden_size)
        # self.fc2 = torch.nn.Linear(hidden_size, n_blocks * (n_blocks + 2) * 2)
        # self.fc2 = torch.nn.Linear(input_size, n_blocks * (n_blocks + 2) * 2)
        self.fc2 = torch.nn.Linear(input_size, n_blocks)
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
        self.fc = torch.nn.Linear(hidden_size, n_blocks)

    def forward(self, x, *args):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1])
        return x
    
class MultiProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_probes):
        super().__init__()
        self.probes = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_size) for _ in range(n_probes)])
        self.fc = torch.nn.Linear(hidden_size, n_blocks)

    def forward(self, x):
        
        for i in range(len(self.probes)):
            z_ = self.probes[i](x[:, i])
            if i == 0:
                z = z_
            else:
                z = z + z_
        z = z / len(self.probes)

        z = self.fc(z)

        return z
    
class AHProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_blocks):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(hidden_size))
        self.v = torch.nn.Parameter(torch.randn(hidden_size))

        self.proj = torch.nn.Linear(input_size, hidden_size)

        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x, mask):
        x = self.proj(x)
        # scores = torch.einsum("bsh,h->bs", x, self.q)

        scores = x @ self.q

        # print(scores.shape, mask.shape)
        scores = scores.masked_fill(mask, -1000)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        z = torch.matmul(scores.unsqueeze(1), x).squeeze(1)
        z = self.fc(z)
        return z
    
class MLHAProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, n_blocks):
        super().__init__()
        self.head_dim = hidden_size // n_heads
        self.n_heads = n_heads

        self.q = torch.nn.Parameter(torch.randn(n_heads, self.head_dim))

        self.proj_k = torch.nn.Linear(input_size, hidden_size)
        self.proj_v = torch.nn.Linear(input_size, hidden_size)

        self.fc = torch.nn.Linear(hidden_size, n_blocks)

    def forward(self, x, mask):
        v = self.proj_v(x)
        x = self.proj_k(x)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim)
        scores = torch.einsum("bshd,hd->bsh", x, self.q)

        scores = scores / np.sqrt(self.head_dim)

        scores = scores.masked_fill(mask.unsqueeze(-1), -1000)
        scores = torch.nn.functional.softmax(scores, dim=-2)

        z = torch.einsum("bshd,bsh->bhd", v, scores)
        z = z.view(z.shape[0], -1)
        z = self.fc(z)

        return z

class DataFetcher:
    def __init__(self, dataset: StepProbeDataset, macro_batch_size: int, rank: int, world_size:int, shuffle: bool = True):
        self.dataset = dataset
        self.macro_batch_size = macro_batch_size
        self.total_items = len(dataset.items)
        self.item_ids = np.arange(self.total_items)
        self.rank = rank
        self.world_size = world_size
        if shuffle:
            self.item_ids = np.random.permutation(self.item_ids)
        self.item_ids = np.array_split(self.item_ids, world_size)[rank]
        print(f"Rank {rank} has {len(self.item_ids)} items")
        
    def iter_batches(self):
        for macro_batch_start in range(0, len(self.item_ids), self.macro_batch_size):
            macro_batch = self.item_ids[macro_batch_start:macro_batch_start + self.macro_batch_size]
            data = self.dataset.get_batch(macro_batch)
            data_keys = data.keys()

            for mini_batch_start in range(0, len(macro_batch), self.dataset.batch_size):
                mini_batch = {
                    k: data[k][mini_batch_start:mini_batch_start + self.dataset.batch_size] for k in data_keys
                }                
                mini_batch = collate_fn(mini_batch)

                yield mini_batch


import ray.train.torch

def train_func(config):
    n_rows = config.get("n_rows", 5000)
    train_test_split = config.get("train_test_split", 0.8)
    n_dim = config.get("n_dim", 5120)
    n_blocks = config.get("n_blocks", 6)
    lr = config.get("lr", 1e-4)
    patience = config.get("patience", 10)
    n_epochs = config.get("n_epochs", 500)
    n_prev_tokens = config.get("n_prev_tokens", 100)
    n_shift_tokens = config.get("n_shift_tokens", 0)
    dataset_actor_name = config.get("dataset_actor_name", "dataset_actor")
    batch_size = config.get("batch_size", 2048)
    macro_batch_size = config.get("macro_batch_size", 10000)
    n_layer = config.get("n_layer", 63)
    
    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    tokenizer = initialize_tokenizer(model_id)
    dataset, eval_results, labels_dataset = load_datasets()

    labels_dict = make_labels_dict(labels_dataset, dataset)

    training_data = make_data_to_process(dataset, labels_dict, n_rows, eval_results, "all", tokenizer)

    n_train = int(len(training_data) * train_test_split)

    train_items = training_data[:n_train]
    test_items = training_data[n_train:]

    probe = MLHAProbe(n_dim, n_dim, 40, n_blocks)
    # probe = GRUProbe(n_dim, 1000, n_blocks)
    probe = ray.train.torch.prepare_model(probe)

    train_dataset = StepProbeDataset(
    train_items, n_layer, n_prev_tokens, n_shift_tokens, dataset_actor_name, n_blocks, batch_size)

    test_dataset = StepProbeDataset(test_items, n_layer, n_prev_tokens, n_shift_tokens, dataset_actor_name, n_blocks, batch_size)

    optimizer = Adam(probe.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()


    train_loader = DataFetcher(train_dataset, macro_batch_size, rank, world_size)
    test_loader = DataFetcher(test_dataset, macro_batch_size, rank, world_size, shuffle=False)


    best_f1 = float('-inf')
    early_stop_counter = 0

    for epoch in range(n_epochs):
        probe.train()
        total_loss = 0
        n_samples = 0
        epoch_start = time.time()

        batch_times = []
        forward_times = []

        prev_time = time.time()
        for batch in train_loader.iter_batches():
            next_time = time.time()

            batch_times.append(next_time - prev_time)

            optimizer.zero_grad()
            input = batch["input"].float().to(probe.device)
            labels = batch["labels"].to(probe.device)
            # labels = batch["logits"].to(probe.device)
            mask = batch["mask"].to(probe.device)

            fwd_start = time.time()

            output = probe(input, mask)

            fwd_end = time.time()

            forward_times.append(fwd_end - fwd_start)

            # print(output.shape, labels.shape, input.shape)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch["input"])
            n_samples += len(batch["input"])


            print(f"Mean batch time: {np.mean(batch_times[-5:])}")
            print(f"Mean forward time: {np.mean(forward_times[-5:])}")

            prev_time = next_time

        
        next_time = time.time()

        print(
            f"Epoch time: {next_time - prev_time:.4f} seconds"
        )

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

            prev_time = time.time()
            for batch in test_loader.iter_batches():
                # print(batch["input"].shape)
                input = batch["input"].float().to(probe.device)
                labels = batch["labels"].to(probe.device)
                labels = batch["logits"].to(probe.device)
                mask = batch["mask"].to(probe.device)

                output = probe(input, mask)

                loss = criterion(output, labels)
                val_loss += loss.item() * len(batch["input"])

                labels = batch["labels"]

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
            
            if ray.train.get_context().get_world_rank() == 0:
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

    
config = {
    "batch_size": 128,
    "macro_batch_size": 512,
    "patience": 30
}

scaling_config = ray.train.ScalingConfig(num_workers=6, use_gpu=True)

# [5] Launch distributed training job.
trainer = ray.train.torch.TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    train_loop_config=config,
)
result = trainer.fit()


print("WWW")

