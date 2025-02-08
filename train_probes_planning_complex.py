import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange, tqdm
from utils import Probe, set_seed, train_probe
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

os.chdir("..")

seed = 42
set_seed(seed)

activations_file = "planning_activations_32b_big_step.pt"
metadata_file = "planning_metadata.json"

activations = torch.load(activations_file)

with open(metadata_file) as f:
    metadata = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = []
all_steps = set()

for x in metadata:
    dataset_idx = x["dataset_idx"]
    activation = activations[dataset_idx]

    extracted_plan = x["bench_item"]["extracted_llm_plan"]
    validity = x["bench_item"]["llm_validity"]

    if validity != 1:
        continue

    steps = set(extracted_plan.split("\n"))
    steps = {step for step in steps if step != ""}
    all_steps.update(steps)

    think_pos = x["think_pos"]

    dataset.append({
        "activations": activation[:think_pos // 10],
        "steps": steps,
        "think_pos": think_pos
    })

import random
random.shuffle(dataset)

test_size = 0.2
test_size = int(len(dataset) * test_size)

train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

n_dim = 5120

train_data_items = {step: [] for step in all_steps}

for item in train_dataset:
    activations = item["activations"]
    steps = item["steps"]
    think_pos = item["think_pos"]

    for step in steps:
        train_data_items[step].append((activations, think_pos))

counts = {step: len(train_data_items[step]) for step in all_steps}
cutoff = 10
all_steps = [step for step in all_steps if counts[step] > cutoff]

positive_data = {
    "train": {step: [] for step in all_steps},
    "test": {step: [] for step in all_steps}
}
negative_data = {
    "train": {step: [] for step in all_steps},
    "test": {step: [] for step in all_steps}
}

for x in train_dataset:
    activations = x["activations"]
    steps = x["steps"]
    think_pos = x["think_pos"]

    for step in all_steps:
        if step in steps:
            positive_data["train"][step].append((activations, think_pos, True))
        else:
            negative_data["train"][step].append((activations, think_pos, False))

for x in test_dataset:
    activations = x["activations"]
    steps = x["steps"]
    think_pos = x["think_pos"]

    for step in all_steps:
        if step in steps:
            positive_data["test"][step].append((activations, think_pos, True))
        else:
            negative_data["test"][step].append((activations, think_pos, False))

final_steps = set()

for step in all_steps:
    pos_neg_ratio = len(positive_data["train"][step]) / len(negative_data["train"][step])

    if pos_neg_ratio < 0.5:
        continue
    if pos_neg_ratio > 2:
        continue

    final_steps.add(step)

class ProbeDataset(Dataset):
    def __init__(self, dataset, probe_pos, step, positive_data, negative_data, aggregate=False, balance=True):
        self.dataset = dataset
        self.probe_pos = probe_pos
        self.aggregate = aggregate

        self.positive_samples, self.negative_samples = negative_data[step], positive_data[step]

        # fix imbalance
        n_positive = len(self.positive_samples)
        n_negative = len(self.negative_samples)

        n_samples = min(n_positive, n_negative)

        if balance:
            self.positive_samples = self.positive_samples[:n_samples]
            self.negative_samples = self.negative_samples[:n_samples]

        self.samples = self.positive_samples + self.negative_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, _, is_positive = self.samples[idx]
        sample = sample[-200:].float()
        
        return {
            "inputs": sample,
            "label": int(is_positive)
        }

def collate_fn(batch, device):
    inputs = [x["inputs"] for x in batch]
    labels = [x["label"] for x in batch]

    # pad inputs left
    masks = [torch.ones(x.shape[0]) for x in inputs]
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return {
        "inputs": inputs.to(device),
        "label": labels.to(device),
        "mask": masks.to(device)
    }

train_datasets = {}
test_datasets = {}

for step in final_steps:
    train_datasets[step] = ProbeDataset(dataset, None, step, positive_data["train"], negative_data["train"], aggregate=False)
    test_datasets[step] = ProbeDataset(dataset, None, step, positive_data["test"], negative_data["test"], aggregate=False, balance=True)

class LSTMProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() 

        self.lstm = torch.nn.GRU(input_size, hidden_size, batch_first=True, dtype=torch.float32, num_layers=1)
        self.fc = torch.nn.Linear(hidden_size, output_size, dtype=torch.float32)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

probes = {
    step: LSTMProbe(n_dim, 256, 2).to(device) for step in final_steps
}

def train_on_device(step, probe, train_dataset, test_dataset, device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    probe.to(device)
    collate_fn_with_device = partial(collate_fn, device=device)
    return train_probe(probe, train_dataset, test_dataset, n_epochs=10, silent=True, lr=1e-4, collate_fn=collate_fn_with_device, batch_size=16)[1]

accuracy = {}

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(train_on_device, step, probes[step], train_datasets[step], test_datasets[step], device_id): step
        for device_id, step in enumerate(final_steps)
    }
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        step = futures[future]
        accuracy[step] = future.result()

for step in final_steps:
    propotion = len(positive_data["test"][step]) / (len(positive_data["test"][step]) + len(negative_data["test"][step]))
    print(step, accuracy[step], len(positive_data["train"][step]), len(negative_data["train"][step]), len(positive_data["test"][step]), len(negative_data["test"][step]), max(propotion, 1 - propotion))
