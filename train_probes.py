import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange, tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # If you are using CuDNN, you can also set the deterministic flag
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
seed = 42
set_seed(seed)

activations_file = "test_activations_7b.pt"
pos_info_file = "positions.json"

activations = torch.load(activations_file)

with open(pos_info_file) as f:
    pos_info = json.load(f)

dataset = []
for p in pos_info:
    if p["result"] == "fail":
        continue

    idx, pos, total = p["id"], p["pos"], p["len"]
    dataset.append({
        "id": idx,
        "pos": pos,
        "total": total,
        "activations": activations[idx]
    })

test_size = 0.2
test_size = int(len(dataset) * test_size)

train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

window_sizes = 2 ** np.arange(6, 10)


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
    def __init__(self, dataset, window_size, n_samples=2000):
        self.dataset = dataset
        self.window_size = window_size
        self.n_samples = n_samples
        self.offset = min(window_size // 2, 300)

        self.positive_samples, self.negative_samples = generate_samples(dataset, window_size, self.offset, n_samples)
        self.samples = self.positive_samples + self.negative_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        act_idx, (token_pos, sample_type) = self.samples[idx]

        sample = self.dataset[act_idx]
        activations = sample["activations"]


        return {
            "inputs": activations[token_pos],
            "label": int(sample_type),
            "sample_idx": act_idx,
            "token_pos": token_pos
        }

train_datasets = []
test_datasets = []

for window_size in window_sizes:
    train_datasets.append(ProbeDataset(train_dataset, window_size))
    test_datasets.append(ProbeDataset(test_dataset, window_size))


n_dim = 3584

class Probe(torch.nn.Module):
    def __init__(self, n_dim, n_classes):
        super().__init__()

        self.linear = torch.nn.Linear(n_dim, n_classes, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(x)


def train_probe(probe, train_dataset, test_dataset, n_epochs=1, lr=1e-3):
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    pbar = trange(n_epochs)

    for epoch in pbar:
        probe.train()

        for batch in train_loader:
            inputs = batch["inputs"]
            labels = batch["label"]

            optimizer.zero_grad()

            outputs = probe(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}, loss: {loss.item()}")

        probe.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"]
                labels = batch["label"]

                outputs = probe(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch}, acc: {acc}")

    return probe


probes = []
for i, window_size in enumerate(window_sizes):
    probe = Probe(n_dim, 2)
    probe = train_probe(probe, train_datasets[i], test_datasets[i])
    probes.append(probe)

torch.save(probes, "probes2.pt")