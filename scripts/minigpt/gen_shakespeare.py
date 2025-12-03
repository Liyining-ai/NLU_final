import math, torch, torch.nn as nn, torch.nn.functional as F, sys
from torch.utils.data import Dataset, DataLoader
sys.path.append('.')
from config import GPT2Config
from models.gpt2 import GPT2Model

# params
configs = [
    {"name": "Tiny",  "d": 64,  "l": 2, "h": 4, "lr": 1e-3, "bs": 16, "drop": 0.0, "block": 32},
    {"name": "Small", "d": 128, "l": 4, "h": 4, "lr": 5e-4, "bs": 32, "drop": 0.1, "block": 64},
    {"name": "Medium", "d": 768, "l": 6, "h": 12, "lr": 5e-5, "bs": 16, "drop": 0.1, "block": 128},
]
epochs       = 5
eval_every   = 100 # print freq
device       = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

# tokenizer
with open('/home/ubuntu/NLP/downstream-tasks/data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
# 80/10/10  split
n_all   = len(data)
n_train = int(0.80 * n_all)
n_val   = int(0.10 * n_all)
n_test  = n_all - n_train - n_val
train_data, val_data, test_data = torch.split(data, [n_train, n_val, n_test])

def chunk(seq, size):
    n_seq = len(seq) // size
    return seq[:n_seq * size].view(n_seq, size)

class ChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# eval
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mask = torch.ones_like(x)
        logits = model.hidden_state_to_token(model(x, mask)['last_hidden_state'])
        B, T, V = logits.shape
        total_loss += F.cross_entropy(logits.view(B*T, V), y.view(B*T), reduction='sum').item()
        total_tokens += B * T
    model.train()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 100))

# gen text
@torch.no_grad()
def generate(model, prefix="To be or not to be", max_new=300):
    model.eval()
    ids = torch.tensor(encode(prefix), dtype=torch.long).unsqueeze(0).to(device)
    block = model.config.max_position_embeddings
    for _ in range(max_new):
        idx_crop = ids[:, -block:]
        mask = torch.ones_like(idx_crop)
        logits = model.hidden_state_to_token(model(idx_crop, mask)['last_hidden_state'])[:, -1, :]
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
    return decode(ids[0].tolist())

# train a certain config
def train(cfg):
    print(f"\n>>>>  Training {cfg['name']}  <<<<")
    block_size = cfg["block"]
    train_chunks = chunk(train_data, block_size + 1)
    val_chunks   = chunk(val_data,   block_size + 1)
    test_chunks  = chunk(test_data,  block_size + 1)

    train_set = ChunkDataset(train_chunks)
    val_set   = ChunkDataset(val_chunks)
    test_set  = ChunkDataset(test_chunks)

    train_ld = DataLoader(train_set, batch_size=cfg["bs"], shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_set,   batch_size=cfg["bs"], shuffle=False, drop_last=True)
    test_ld  = DataLoader(test_set,  batch_size=cfg["bs"], shuffle=False, drop_last=True)

    model = GPT2Model(GPT2Config(
        vocab_size=vocab_size,
        hidden_size=cfg["d"],
        num_hidden_layers=cfg["l"],
        num_attention_heads=cfg["h"],
        intermediate_size=4 * cfg["d"],
        max_position_embeddings=block_size,
        attention_probs_dropout_prob=cfg["drop"],
        hidden_dropout_prob=cfg["drop"],
    )).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    step = 0
    loss_list = []
    ppl_list = []
    for epoch in range(epochs):
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            mask = torch.ones_like(x)
            logits = model.hidden_state_to_token(model(x, mask)['last_hidden_state'])
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))  # mean over tokens
            perplexity = math.exp(min(loss.item(), 100))
            loss_list.append(loss.item())
            ppl_list.append(perplexity)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % eval_every == 0 and step != 0:
                avg_train_loss = sum(loss_list) / eval_every
                avg_train_ppl  = sum(ppl_list)  / eval_every
                # train_loss, train_ppl = loss.item(), math.exp(min(loss.item(), 100))
                val_loss,   val_ppl   = evaluate(model, val_ld)
                print(f"step {step:4d} (epoch {epoch}) | "
                    f"train loss {avg_train_loss:.4f} ppl {avg_train_ppl:.2f} | "
                    f"val loss {val_loss:.4f} ppl {val_ppl:.2f}")
                loss_list = []
                ppl_list = []
            step += 1

    # test
    test_loss, test_ppl = evaluate(model, test_ld)
    print(f"=== {cfg['name']}  TEST  loss={test_loss:.4f}  ppl={test_ppl:.2f} ===")
    # sample
    for i in range(3):
        print(f"\n--- {cfg['name']} sample {i+1} ---")
        print(generate(model))
    return test_ppl


def main():
    results = {}
    for cfg in configs:
        results[cfg["name"]] = train(cfg)
    print("\nResults:")
    for k, v in results.items():
        print(f"{k:5s}  test_ppl = {v:.2f}")

if __name__ == "__main__":
    main()