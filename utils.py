import torch
import json

from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange, tqdm
from transformers import AutoTokenizer

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


def train_probe(probe, train_dataset, test_dataset, n_epochs=10, lr=1e-3, silent=False, collate_fn=None, batch_size=256, patience=5, n_steps=None):
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

        for step_n, batch in enumerate(train_loader):
            inputs = batch["inputs"]
            labels = batch["label"]

            optimizer.zero_grad()

            outputs = probe(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if n_steps is not None and step_n >= n_steps:
                break

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

THINK_TOKEN = 151649
THINK_START_TOKEN = 151648

def initialize_tokenizer(model_id) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = tokenizer.chat_template.replace("{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", "")

    return tokenizer

def tokenize_blocksworld_generation(tokenizer, row, generation=None):
    if generation is None:
        generation = row["generation"]
    query = row["distilabel_metadata"]["raw_input_text_generation_0"][0]

    messages = [
        query,
        {"role": "assistant", "content": "<think>\n"+generation}
    ]
    chat    = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

    return chat

# Phrases to steer with
DOMAIN_PHRASES = {
    "mystery_1": {
        "actions": {
            "attack": "attack",
            "succumb": "succumb",
            "overcome": "overcome",
            "feast": "feast"
        },
        "predicates": {
            "planet": "planet",
            "province": "province",
            "harmony": "harmony",
            "craves": "craves",
            "pain": "pain"
        }
    },
    "mystery_2": {
        "actions": {
            "attack": "illuminate",
            "succumb": "silence",
            "overcome": "distill",
            "feast": "divest"
        },
        "predicates": {
            "planet": "aura",
            "province": "essence",
            "harmony": "nexus",
            "craves": "harmonizes",
            "pain": "pulse"
        }
    },
    "mystery_3": {
        "actions": {
            "attack": "tltezi",
            "succumb": "jchntg",
            "overcome": "deesdu",
            "feast": "xavirm"
        },
        "predicates": {
            "planet": "oxtslo",
            "province": "adohre",
            "harmony": "jqlyol",
            "craves": "gszswg",
            "pain": "ivbmyg"
        }
    },
    "mystery_4": {
        "actions": {
            "attack": "swim",
            "succumb": "fire",
            "overcome": "deduct",
            "feast": "respond"
        },
        "predicates": {
            "planet": "fever",
            "province": "marble",
            "harmony": "craving",
            "craves": "mines",
            "pain": "shadow"
        }
    },
    "mystery_5": {
        "actions": {
            "attack": "whisper",
            "succumb": "calculate",
            "overcome": "orbit",
            "feast": "navigate"
        },
        "predicates": {
            "planet": "crystal",
            "province": "fountain",
            "harmony": "autumn",
            "craves": "illuminates",
            "pain": "legend"
        }
    },
    "mystery_6": {
        "actions": {
            "attack": "decode",
            "succumb": "hibernate",
            "overcome": "thunder",
            "feast": "quench"
        },
        "predicates": {
            "planet": "prism",
            "province": "hollow",
            "harmony": "zenith",
            "craves": "echoes",
            "pain": "emblem"
        }
    },
    "mystery_7": {
        "actions": {
            "attack": "explore",
            "succumb": "ripen",
            "overcome": "weave",
            "feast": "bloom"
        },
        "predicates": {
            "planet": "fossil",
            "province": "dialect",
            "harmony": "equinox",
            "craves": "fractures",
            "pain": "symphony"
        }
    },
    "mystery_8": {
        "actions": {
            "attack": "harvest",
            "succumb": "ignite",
            "overcome": "carve",
            "feast": "suspend"
        },
        "predicates": {
            "planet": "nebula",
            "province": "labyrinth",
            "harmony": "mirage",
            "craves": "captivates",
            "pain": "cascade"
        }
    },
    "mystery_9": {
        "actions": {
            "attack": "construct",
            "succumb": "demolish",
            "overcome": "reinforce",
            "feast": "collapse"
        },
        "predicates": {
            "planet": "eclipse",
            "province": "vintage",
            "harmony": "paradox",
            "craves": "resonates",
            "pain": "twilight"
        }
    },
    "mystery_10": {
        "actions": {
            "attack": "plant",
            "succumb": "harvest",
            "overcome": "nurture",
            "feast": "prune"
        },
        "predicates": {
            "planet": "crystal",
            "province": "puzzle",
            "harmony": "vortex",
            "craves": "whispers",
            "pain": "cipher"
        }
    },
    "mystery_11": {
        "actions": {
            "attack": "prosecute",
            "succumb": "acquit",
            "overcome": "testify",
            "feast": "appeal"
        },
        "predicates": {
            "planet": "nebula",
            "province": "molecule",
            "harmony": "anthem",
            "craves": "silhouettes",
            "pain": "voltage"
        }
    },
    "mystery_12": {
        "actions": {
            "attack": "broadcast",
            "succumb": "receive",
            "overcome": "encrypt",
            "feast": "decode"
        },
        "predicates": {
            "planet": "horizon",
            "province": "compass",
            "harmony": "solstice",
            "craves": "orbits",
            "pain": "quantum"
        }
    },
    "mystery_13": {
        "actions": {
            "attack": "whisper",
            "succumb": "banish",
            "overcome": "entangle",
            "feast": "unmask"
        },
        "predicates": {
            "planet": "tethered",
            "province": "unburdened",
            "harmony": "hollow",
            "craves": "shrouds",
            "pain": "consuming"
        }
    },
    "mystery_14": {
        "actions": {
            "attack": "question",
            "succumb": "resolve",
            "overcome": "interweave",
            "feast": "liberate"
        },
        "predicates": {
            "planet": "echoing",
            "province": "sovereign",
            "harmony": "potential",
            "craves": "obscures",
            "pain": "contemplating"
        }
    },
    "mystery_15": {
        "actions": {
            "attack": "summon",
            "succumb": "dismiss",
            "overcome": "fold",
            "feast": "unravel"
        },
        "predicates": {
            "planet": "suspended",
            "province": "timeless",
            "harmony": "interval",
            "craves": "transcends",
            "pain": "enveloping"
        }
    },
    "mystery_16": {
        "actions": {
            "attack": "illuminate",
            "succumb": "silence",
            "overcome": "distill",
            "feast": "divest"
        },
        "predicates": {
            "planet": "aura",
            "province": "essence",
            "harmony": "nexus",
            "craves": "harmonizes",
            "pain": "pulse"
        }
    }
}