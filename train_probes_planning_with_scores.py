import torch
from torch.utils.data import DataLoader
from utils import set_seed, train_probe, collate_fn, ProbeDataset
import json

# Set random seed for reproducibility
seed = 42
set_seed(seed)

# Define the new model architecture
class ScoreBasedProbe(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScoreBasedProbe, self).__init__()
        self.linear_per_element = torch.nn.Linear(input_size, 1)
        self.final_linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        # Apply linear layer to each sequence element
        scores = self.linear_per_element(x).squeeze(-1)
        # Use scores to weight the sequence elements
        weighted_sum = torch.einsum('bse,bs->be', x, scores.softmax(dim=-1))
        # Final linear layer
        output = self.final_linear(weighted_sum)
        return output

# Load data
activations_file = "planning_activations_32b_big_step.pt"
metadata_file = "planning_metadata.json"

activations = torch.load(activations_file)

with open(metadata_file) as f:
    metadata = json.load(f)

# Prepare datasets
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

train_datasets = {}
test_datasets = {}

for step in final_steps:
    train_datasets[step] = ProbeDataset(dataset, None, step, positive_data["train"], negative_data["train"], aggregate=False)
    test_datasets[step] = ProbeDataset(dataset, None, step, positive_data["test"], negative_data["test"], aggregate=False, balance=True)

# Initialize and train the model
probes = {
    step: ScoreBasedProbe(n_dim, 256, 2).to(device) for step in final_steps
}

accuracy = {}

collate_fn = lambda batch: collate_fn(batch, device)

for step in final_steps:
    accuracy[step] = train_probe(probes[step], train_datasets[step], test_datasets[step], n_epochs=10, silent=False, lr=1e-4, collate_fn=collate_fn, batch_size=16)[1]
    exit()


for step in final_steps:
    propotion = len(positive_data["test"][step]) / (len(positive_data["test"][step]) + len(negative_data["test"][step]))
    print(step, accuracy[step], len(positive_data["train"][step]), len(negative_data["train"][step]), len(positive_data["test"][step]), len(negative_data["test"][step]), max(propotion, 1 - propotion))
