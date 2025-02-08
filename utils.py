import torch
import json

from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange, tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # If you are using CuDNN, you can also set the deterministic flag
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Probe(torch.nn.Module):
    def __init__(self, n_dim, n_classes):
        super().__init__()

        self.linear = torch.nn.Linear(n_dim, n_classes, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(x)


def load_data(activations_file, pos_info_file):
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
            "correct": p["correct"],
            "activations": activations[idx]
        })

    test_size = 0.2
    test_size = int(len(dataset) * test_size)

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    return train_dataset, test_dataset


def train_probe(probe, train_dataset, test_dataset, n_epochs=10, lr=1e-3, silent=False, collate_fn=None, batch_size=256, patience=5):
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # if not silent:
    #     pbar = trange(n_epochs)
    # else:
    #     pbar = range(n_epochs)

    pbar = range(n_epochs)

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in pbar:
        if epochs_no_improve >= patience:
            if not silent:
                print("Early stopping triggered")
            break
        probe.train()

        for batch in train_loader:
            inputs = batch["inputs"]
            labels = batch["label"]

            optimizer.zero_grad()

            outputs = probe(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # if not silent:
            #     pbar.set_description(f"Epoch {epoch}, loss: {loss.item()}")
        if not silent:
            print(f"Epoch {epoch}, loss: {loss.item()}")

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
        if not silent:
            print(f"Epoch {epoch}, acc: {acc}")

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

    return probe, best_acc, loss.item()

