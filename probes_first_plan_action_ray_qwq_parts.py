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
import datasets

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
model_id = "Qwen/QwQ-32B"

tokenizer = initialize_tokenizer(model_id)


n_blocks = 6


cur_dir = Path(".").absolute()


def load_dataset_from_file(domain_name, task_name):
    prompt_dir = cur_dir / Path(f"./cot-planning/results/{domain_name}/deepseek-32b/")
    with open(prompt_dir / f"{task_name}.json", 'r') as file:
        return json.load(file)

def map_sections_into_tokens(sections: list[tuple[str, str]], row: dict) -> list[dict]:
    # tokens = tokenize_blocksworld_generation(tokenizer, row)
    generation = row["generation"]
    section_tokens = []
    for label, content in sections[1:]:
        text_pos = generation.find(content[:300])
        if text_pos == -1:
            continue
        text_before = generation[:text_pos]
        tokens_before = tokenize_blocksworld_generation(tokenizer, row, text_before)[0, :-2]
        
        if len(tokens_before) > 6000:
            continue
        
        content_tokens = tokenizer.encode(" " + content)[:-2]
        
        section_tokens.append({
            "label": label,
            "pos_before": len(tokens_before),
            "pos_after": len(tokens_before) + len(content_tokens),
            "text_pos": text_pos,
            "content": content
        })

    return section_tokens

import re

labels_low = [
    "initial-state-understanding",
    "goal-state-understanding",
    "state-tracking",
    "action-exploration",
    "state-tracking"
]

def parse_sections(text:str, labels: list[str]) -> list[tuple[str, str]]:
    pattern = r'\["([^"]+)"\](.*?)\["end-section"\]'
    matches = re.findall(pattern, text, re.DOTALL)
    
    sections = []
    for label, content in matches:
        if label in labels:
            sections.append((label, content.strip()))
    
    return sections

@ray.remote
def collect_section_tokens_remote(labels_dataset: dict, dataset: dict, sections):
    section_tokens = {}
    print("Collecting section tokens")
    for idx in dataset:
        label = labels_dataset[idx]["label"]
        if label is None:
            continue
        sections = parse_sections(labels_dataset[idx]["label"], labels_low)
        section_tokens[idx] = map_sections_into_tokens(sections, dataset[idx])
        
    return section_tokens

def collect_section_tokens(labels_dataset: datasets.Dataset, dataset: datasets.Dataset, labels_low: list[str], n_rows: int, n_workers: int):
    section_tokens = {}
    futures = []
    rows_per_worker = n_rows // n_workers
    
    idx_to_row = {i: x for i, x in enumerate(dataset)}
    idx_to_label = {i: x for i, x in enumerate(labels_dataset)}
    
    for i in range(n_workers):
        start_idx = i * rows_per_worker
        end_idx = (i + 1) * rows_per_worker
        futures.append(collect_section_tokens_remote.remote(
            {
                k: idx_to_label[k] for k in range(start_idx, end_idx)    
            }, 
            {
                k: idx_to_row[k] for k in range(start_idx, end_idx)  
            },
            labels_low,
        ))

    for future in futures:
        section_tokens.update(ray.get(future))

    return section_tokens
    

def load_datasets():
    dataset = load_dataset(
    f"dmitriihook/qwq-32b-planning-6-blocks")["train"]
    
    task_name = "plan_generation_po"
    domain_name = f"blocksworld_{n_blocks}_blocks"
    eval_results = load_dataset_from_file(domain_name, task_name)["instances"]
    eval_results = {x["dataset_idx"]: x for x in eval_results}

    labels_dataset = load_dataset("dmitriihook/blocksworld-6-blocks-qwq-reasoning-parts-low-v4")["train"]
    
    section_tokens = collect_section_tokens(labels_dataset, dataset, labels_low, len(labels_dataset), 32)

    return dataset, eval_results, section_tokens


# total_layers = model.config.num_hidden_layers
total_layers = 64


def make_data_to_process(dataset, section_tokens, n_rows, eval_results, answer_type, tokenizer):
    data_to_process = []
    for idx, row in enumerate(dataset.select(range(n_rows))):
        if eval_results[idx]["llm_correct"] and answer_type == "incorrect":
            continue
        if not eval_results[idx]["llm_correct"] and answer_type == "correct":
            continue
        generation = row["generation"]

        if "[PLAN]\n" not in generation:
            continue

        if idx not in section_tokens:
            continue
        
        actions = extract_actions(row)
        if actions is None:
            continue

        parsed_actions = parse_block_actions(actions)
        plan_start = generation.index("[PLAN]\n") + len("[PLAN]\n")
        counts = defaultdict(int)

        for section in section_tokens[idx]:
            if section["pos_after"] < section["pos_before"] + 2:
                continue
            
            counts[section["label"]] += 1
                
            data_to_process.append({
                "idx": idx,
                "pos_before": section["pos_before"],
                "pos_after": section["pos_after"],
                "actions": parsed_actions,
                "label": section["label"],
                "label_n": counts[section["label"]],
            })


    return data_to_process


def process_data(items, n_blocks: int, tgt_action: int, part_type: str, min_n: int, max_n: int, max_tokens = 3200):
    new_items = []

    for item in items:
        actions = item["actions"]
        if item["label"] != part_type:
            continue
        if item["label_n"] < min_n or item["label_n"] > max_n:
            continue
        try:
            blocks = actions[tgt_action][1]
            block = blocks[0]
            label = block2int(block, n_blocks)
        except Exception as e:
            continue

        new_items.append({
            "pos_before": item["pos_before"],
            "label": label,
            "idx":  item["idx"],
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
    def __init__(self, items, n_layer, n_prev_tokens, shift_tokens, hidden_states, n_blocks, batch_size, tgt_action, part_type, min_n, max_n):
        self.items = process_data(items, n_blocks, tgt_action, part_type, min_n, max_n)
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
            pos_start = item["pos_before"]

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


dataset, eval_results, section_tokens = load_datasets()

section_tokens_ref = ray.put(section_tokens)
dataset_ref = ray.put(dataset)
eval_results_ref = ray.put(eval_results)

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
        n_collate_actors = config.get("n_collate_actors", 8)
        part_type = config.get("part_type", "goal-state-understanding")
        min_n = config.get("min_n", 0)
        max_n = config.get("max_n", 6)
        
        rank = ray.train.get_context().get_world_rank()
        world_size = ray.train.get_context().get_world_size()

        tokenizer = initialize_tokenizer(model_id)
        dataset, eval_results, section_tokens = ray.get(dataset_ref), ray.get(eval_results_ref), ray.get(section_tokens_ref)

        training_data = make_data_to_process(dataset, section_tokens, n_rows, eval_results, "all", tokenizer)

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
        train_items, n_layer, n_prev_tokens, n_shift_tokens, hidden_states, n_blocks, batch_size, tgt_action, part_type, min_n, max_n)

        test_dataset = StepProbeDataset(test_items, n_layer, n_prev_tokens, n_shift_tokens, hidden_states, n_blocks, batch_size, tgt_action, part_type, min_n, max_n)

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
n_gpus = 4
n_dim = 5120

# for n_shift_tokens in range(100, 2000, 100):
#     for tgt_action in range(8):

tgt_action = 2
n_shift_tokens = 60

config = {
    "batch_size": orig_batch_size // n_gpus,
    "macro_batch_size": orig_batch_size * 4,
    "patience": 100,
    "n_epochs": 500,
    "n_rows": 2000,
    "train_test_split": 0.9,
    "do_profile": False,
    "n_prev_tokens": 61,
    "n_shift_tokens": -n_shift_tokens,
    "probe_type": "mlha",
    "gru_hidden": 100,
    "n_dim": n_dim,
    "head_dim": n_dim // 4,
    "n_heads": 40,
    "n_layer": 39,
    "tgt_action": tgt_action,
    "lr": 5e-5,
    "part_type": "action-exploration",
    "min_n": 1,
    "max_n": 1,
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


