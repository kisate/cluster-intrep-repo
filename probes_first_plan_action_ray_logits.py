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
import torchmetrics
import numpy as np

import torchmetrics.aggregation
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset

from utils import initialize_tokenizer, tokenize_blocksworld_generation, THINK_TOKEN
from stacks_utils import *
from tqdm.auto import tqdm, trange
from more_itertools import chunked


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


def make_data_to_process(dataset, labels_dict, n_rows, eval_results, answer_type, tokenizer, tgt_action, n_cons_lines):
    data_to_process = []
    for idx, row in enumerate(dataset.select(range(n_rows))):
        if eval_results[idx]["llm_correct"] and answer_type == "incorrect":
            continue
        if not eval_results[idx]["llm_correct"] and answer_type == "correct":
            continue
        generation = row["generation"]

        actions = extract_actions(row)
        if actions is None:
            continue
        parsed_actions = parse_block_actions(actions)

        if idx not in labels_dict:
            continue

        try:
            tgt_block = parsed_actions[tgt_action][1][0]
            tgt_block = block2int(tgt_block, n_blocks)
            first_block = parsed_actions[0][1][0]
            first_block = block2int(first_block, n_blocks)
        except Exception as e:
            print(e)
            continue

        # need to remove the eos token
        pos_start = len(tokenize_blocksworld_generation(tokenizer, row, "")[0][:-1])

        n_cons_lines = n_cons_lines

        for line_n in range(10, 50 - n_cons_lines + 1):
            if not all( ln in labels_dict[idx] for ln in range(line_n, line_n + n_cons_lines)):
                continue

            if not all(labels_dict[idx][ln]["probs"][first_block] > 0.9 for ln in range(line_n, line_n + n_cons_lines)):
                continue

            text = "\n\n".join(generation.split("\n\n")[:line_n + n_cons_lines])

            pos = len(tokenize_blocksworld_generation(tokenizer, row, text)[0][:-1])

            data_to_process.append({
                "idx": idx,
                "line_n": line_n,
                "tgt_block": tgt_block,
                "pos": pos,
                "pos_start": pos_start
            })

    return data_to_process


def process_data(items, n_blocks, tgt_action, max_tokens = 3200):
    new_items = []

    for item in items:
        new_items.append({
            "pos_start": item["pos_start"],
            "label": item["tgt_block"],
            "idx":  item["idx"],
            "line_n": item["line_n"],
            "pos": item["pos"]
        })

    return new_items
    

def collate_fn(batch):
    inputs = [torch.tensor(x, dtype=compute_dtype) for x in batch["input"]]    
    masks = [torch.ones(x.shape[0], dtype=torch.bool) for x in inputs]
    inputs = pad_sequence(inputs, batch_first=True,
                          padding_value=0, padding_side="left")
    masks = pad_sequence(masks, batch_first=True,
                         padding_value=True, padding_side="left")

    # inputs = torch.stack(inputs)
    # masks = torch.stack(masks)

    labels = np.stack([x for x in batch["labels"]])
    labels = torch.tensor(labels, dtype=torch.int64)

    return {
        "input": inputs,
        "labels": labels,
        "mask": masks
    }

@ray.remote
class CollateActor:
    def __init__(self):
        pass

    def collate_fn(self, batch):
        return collate_fn(batch)


class StepProbeDataset(Dataset):
    def __init__(self, items, n_layer, n_prev_tokens, shift_tokens, hidden_states, n_blocks, batch_size, tgt_action):
        self.items = process_data(items, n_blocks, tgt_action)
        self.n_layer = n_layer
        self.n_blocks = n_blocks
        self.n_prev_tokens = n_prev_tokens
        self.shift_tokens = shift_tokens
        self.hidden_states = hidden_states
        self.batch_size = batch_size

    def get_batch(self, idxs):
        items = [self.items[idx] for idx in idxs]
        refs = [self.hidden_states[item["idx"]][self.n_layer] for item in items]
        _hidden_states = ray.get(refs)

        inputs = []
        labels = []

        for i, item in enumerate(items):
            pos_start = item["pos"]

            pos = pos_start

            window_start = max(0, pos - self.n_prev_tokens - self.shift_tokens)
            window_end = pos - self.shift_tokens + 1


            label = item["label"]
            hidden_states = _hidden_states[i]

            inputs.append(hidden_states[window_start:window_end])
            labels.append(label)
        return {
            "input": inputs,
            "labels": labels,
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
        self.fc2 = torch.nn.Linear(input_size, n_blocks)

    def forward(self, x):
        x = self.fc2(x)
        return x


class GRUProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_blocks):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True, dtype=compute_dtype)
        self.fc = torch.nn.Linear(hidden_size, n_blocks, dtype=compute_dtype)

    def forward(self, x, *args):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1])
        return x
    

class MLHAProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, n_blocks):
        super().__init__()
        self.head_dim = hidden_size // n_heads
        self.n_heads = n_heads

        self.q = torch.nn.Parameter(torch.randn(n_heads, self.head_dim, dtype=compute_dtype))

        self.proj_k = torch.nn.Linear(input_size, hidden_size, dtype=compute_dtype)
        self.proj_v = torch.nn.Linear(input_size, hidden_size, dtype=compute_dtype)

        self.fc = torch.nn.Linear(hidden_size, n_blocks, dtype=compute_dtype)

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
    def __init__(self, dataset: StepProbeDataset, macro_batch_size: int, rank: int, world_size:int, collate_actors: list, shuffle: bool = True):
        self.dataset = dataset
        self.macro_batch_size = macro_batch_size
        self.total_items = len(dataset.items)
        self.item_ids = np.arange(self.total_items)
        self.rank = rank
        self.world_size = world_size
        if shuffle:
            self.item_ids = np.random.permutation(self.item_ids)
        self.item_ids = np.array_split(self.item_ids, world_size)[rank]
        self.collate_actors = collate_actors
        print(f"Rank {rank} has {len(self.item_ids)} items")
        
    def iter_batches(self):
        for macro_batch_start in range(0, len(self.item_ids), self.macro_batch_size):
            macro_batch = self.item_ids[macro_batch_start:macro_batch_start + self.macro_batch_size]
            data = self.dataset.get_batch(macro_batch)
            data_keys = data.keys()

            mini_batches = [
                {
                    k: data[k][mini_batch_start:mini_batch_start + self.dataset.batch_size] for k in data_keys
                }  for mini_batch_start in range(0, len(macro_batch), self.dataset.batch_size)
            ]

            for mini_batch in mini_batches:
                yield collate_fn(mini_batch)

            # for mini_batch_batch in chunked(mini_batches, len(self.collate_actors)):
            #     futures = [
            #         collate_actor.collate_fn.remote(mini_batch) for collate_actor, mini_batch in zip(self.collate_actors, mini_batch_batch)
            #     ]
            #     for mini_batch in ray.get(futures):
            #         yield mini_batch

            # futures = [
            #     collate_fn.remote(mini_batch) for mini_batch in mini_batches
            # ]

            # for mini_batch in ray.get(futures):
            #     yield mini_batch


import ray.train.torch

import cProfile
import pstats

def train_func(config: dict):
    do_profile = config.get("do_profile", False)
    def _train_func(config: dict):

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
        tgt_action = config.get("tgt_action", 0)
        n_cons_lines = config.get("n_cons_lines", 2)
        n_collate_actors = config.get("n_collate_actors", 8)
        
        rank = ray.train.get_context().get_world_rank()
        world_size = ray.train.get_context().get_world_size()

        tokenizer = initialize_tokenizer(model_id)
        dataset, eval_results, labels_dataset = load_datasets()

        labels_dict = make_labels_dict(labels_dataset, dataset)

        training_data = make_data_to_process(dataset, labels_dict, n_rows, eval_results, "all", tokenizer, tgt_action, n_cons_lines)

        n_train = int(len(training_data) * train_test_split)

        train_items = training_data[:n_train]
        test_items = training_data[n_train:]

        if config.get("probe_type") == "mlha":
            probe = MLHAProbe(n_dim, config.get("head_dim", n_dim), config.get("n_heads", 40), n_blocks)
        elif config.get("probe_type") == "gru":
            probe = GRUProbe(n_dim, config.get("gru_hidden", 1000), n_blocks)
        else:
            raise ValueError("Invalid probe type")

        probe = ray.train.torch.prepare_model(probe)

        actor_handle = ray.get_actor(dataset_actor_name)
        hidden_states = ray.get(actor_handle.get_hidden_states.remote())

        train_dataset = StepProbeDataset(
        train_items, n_layer, n_prev_tokens, n_shift_tokens, hidden_states, n_blocks, batch_size, tgt_action)

        test_dataset = StepProbeDataset(test_items, n_layer, n_prev_tokens, n_shift_tokens, hidden_states, n_blocks, batch_size, tgt_action)

        optimizer = Adam(probe.parameters(), lr=lr)
        criterion = CrossEntropyLoss()
        # criterion = torch.nn.MSELoss()

        # collate_actors = [CollateActor.remote() for _ in range(n_collate_actors)]
        collate_actors = []

        train_loader = DataFetcher(train_dataset, macro_batch_size, rank, world_size, collate_actors)
        test_loader = DataFetcher(test_dataset, macro_batch_size, rank, world_size, collate_actors, shuffle=False)

        device = "cuda"

        acc_meter = torchmetrics.Accuracy(task="multiclass", num_classes=n_blocks).to(device)
        f1_meter = torchmetrics.F1Score(task="multiclass", num_classes=n_blocks).to(device)
        loss_meter = torchmetrics.aggregation.MeanMetric().to(device)
        val_loss_meter = torchmetrics.aggregation.MeanMetric().to(device)

        best_f1 = float('-inf')
        early_stop_counter = 0

        for epoch in range(n_epochs):
            probe.train()
            epoch_start = time.time()

            batch_times = []
            forward_times = []

            prev_time = time.time()
            for batch in train_loader.iter_batches():
                next_time = time.time()

                batch_times.append(next_time - prev_time)

                optimizer.zero_grad()
                input = batch["input"].to(device)
                labels = batch["labels"].to(device)
                # labels = batch["logits"].to(probe.device)
                mask = batch["mask"].to(device)

                fwd_start = time.time()
                output = probe(input, mask)
                fwd_end = time.time()

                forward_times.append(fwd_end - fwd_start)

                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                loss_meter.update(loss.detach())

                # print(f"Mean batch time: {np.mean(batch_times[-5:])}")
                # print(f"Mean forward time: {np.mean(forward_times[-5:])}")

                prev_time = next_time

            
            next_time = time.time()
            train_time = next_time - epoch_start
            start_time = time.time()

            # Evaluation
            probe.eval()
            with torch.no_grad():
                prev_time = time.time()
                for batch in test_loader.iter_batches():
                    # print(batch["input"].shape)
                    input = batch["input"].to(device)
                    labels = batch["labels"].to(device)
                    # labels = batch["logits"].to(probe.device)
                    mask = batch["mask"].to(device)

                    output = probe(input, mask)

                    loss = criterion(output, labels)

                    val_loss_meter.update(loss)

                    preds = output.argmax(dim=1)
                    labels = labels

                    acc_meter.update(preds, labels)
                    f1_meter.update(preds, labels)

                avg_train_loss = loss_meter.compute().item()
                avg_acc = acc_meter.compute().item()
                avg_f1 = f1_meter.compute().item()
                val_loss = val_loss_meter.compute().item()

                eval_time = time.time() - start_time
                    
                ray.train.report(
                    {
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "val_acc": avg_acc,
                        "val_f1": avg_f1,
                        "eval_time": eval_time,
                        "train_time": train_time
                    }
                )

                # if ray.train.get_context().get_world_rank() == 0:
                #     print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Train Acc: {avg_acc:.4f} Train F1: {avg_f1:.4f} Eval Time: {eval_time:.4f} Train Time: {train_time:.4f}")

                # Early Stopping Check
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                loss_meter.reset()
                val_loss_meter.reset()
                acc_meter.reset()
                f1_meter.reset()

                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    if ray.train.get_context().get_world_rank() == 0 and do_profile:
        with cProfile.Profile() as pr:
            _train_func(config)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats(30)
    else:
        _train_func(config) 


orig_batch_size = 64
n_gpus = 8
n_dim = 5120

# for n_shift_tokens in range(100, 2000, 100):
#     for tgt_action in range(8):

tgt_action = 4
n_shift_tokens = 0

config = {
    "batch_size": orig_batch_size // n_gpus,
    "macro_batch_size": orig_batch_size * 4,
    "patience": 100,
    "n_epochs": 500,
    "n_rows": 5000,
    "train_test_split": 0.9,
    "do_profile": False,
    "n_prev_tokens": 100,
    "n_shift_tokens": -n_shift_tokens,
    "probe_type": "mlha",
    "gru_hidden": 100,
    "n_dim": n_dim,
    "head_dim": n_dim // 4,
    "n_heads": 40,
    "n_layer": 55,
    "tgt_action": tgt_action,
    "n_cons_lines": 2,
    "lr": 5e-5
}

scaling_config = ray.train.ScalingConfig(num_workers=n_gpus, use_gpu=True)

# [5] Launch distributed training job.
trainer = ray.train.torch.TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    train_loop_config=config,
)

result = trainer.fit()

print(result)
# 0     2       4
# 0.71  0.41    0.25