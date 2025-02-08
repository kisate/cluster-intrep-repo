import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange, tqdm
from utils import Probe, set_seed, train_probe

seed = 42
set_seed(seed)

activations_file = "planning_activations_32b.pt"
metadata_file = "planning_metadata.json"

activations = torch.load(activations_file)

with open(metadata_file) as f:
    metadata = json.load(f)

dataset = []

all_steps = set()

for x in metadata:
    dataset_idx = x["dataset_idx"]
    activation = activations[dataset_idx]

    extracted_plan = x["bench_item"]["extracted_llm_plan"]
    steps = set(extracted_plan.split("\n"))

    all_steps.update(steps)

    think_pos = x["think_pos"]

    dataset.append({
        "activations": activation,
        "steps": steps,
        "think_pos": think_pos
    })

test_size = 0.2
test_size = int(len(dataset) * test_size)

train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

n_dim = 5120

print(len(all_steps))

probes = {
    step: Probe(n_dim, 2) for step in all_steps
}

print(all_steps)