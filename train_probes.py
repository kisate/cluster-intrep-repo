import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange, tqdm
from utils import Probe, set_seed, load_data, train_probe

seed = 42
set_seed(seed)

train_dataset, test_dataset = load_data("test_activations_7b.pt", "positions.json")

window_sizes = np.logspace(4, 9, 10, base=2).astype(int)


def generate_sample(window_size, pos, offset, total, positive=True):
    if positive:
        left = max(0, pos - window_size - offset)
        right = pos
    else:
        left = 0
        right = pos - window_size - 1

    if right - left < 1:
        raise ValueError("Window too big", left, right, pos - window_size - offset,window_size,  pos, offset)

    sample_idx = np.random.randint(left, right)

    return sample_idx, positive


def generate_samples(dataset, window_size, offset, n_samples):
    positive_samples = []
    negative_samples = []

    for i in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        pos = dataset[idx]["pos"]
        total = dataset[idx]["total"]

        try:
            positive_samples.append((idx, generate_sample(window_size, pos, offset, total, positive=True)))
            negative_samples.append((idx, generate_sample(window_size, pos, offset, total, positive=False)))
        except ValueError as e:
            print(e)
            continue

    return positive_samples, negative_samples


class ProbeDataset(Dataset):
    def __init__(self, dataset, window_size, n_samples=4000):
        self.dataset = dataset
        self.window_size = window_size
        self.n_samples = n_samples
        self.offset = 0

        self.positive_samples, self.negative_samples = generate_samples(dataset, window_size, self.offset, n_samples)
        self.samples = self.positive_samples + self.negative_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        act_idx, (token_pos, sample_type) = self.samples[idx]

        sample = self.dataset[act_idx]
        activations = sample["activations"]
        is_correct = sample["correct"]


        return {
            "inputs": activations[token_pos],
            "label": int(is_correct),
            "sample_idx": act_idx,
            "token_pos": token_pos
        }

train_datasets = []
test_datasets = []

for window_size in window_sizes:
    train_datasets.append(ProbeDataset(train_dataset, window_size))
    test_datasets.append(ProbeDataset(test_dataset, window_size))


n_dim = 3584



probes = []
for i, window_size in enumerate(window_sizes):
    probe = Probe(n_dim, 2)
    probe = train_probe(probe, train_datasets[i], test_datasets[i])
    probes.append({
        "probe": probe,
        "window_size": window_size  
    })

torch.save(probes, "probes_correct.pt")