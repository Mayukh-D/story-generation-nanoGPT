import os
import json
import torch
import torch.nn.functional as F
import random
import tiktoken
from model import GPT, GPTConfig

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "dpo_dataset.json"
BASE_MODEL_DIR = "/content/drive/MyDrive/nanoGPT9/out-story-mix2"
OUT_DIR = "/content/drive/MyDrive/nanoGPT9/out-dpo"

DEVICE = "cuda"
BATCH_SIZE = 16
BLOCK_SIZE = 128

LR = 1e-5
BETA = 0.1
MAX_ITERS = 3000

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
ckpt = torch.load(f"{BASE_MODEL_DIR}/ckpt.pt", map_location="cpu")

gptconf = GPTConfig(**ckpt["model_args"])
model = GPT(gptconf)

state_dict = ckpt["model"]
for k in list(state_dict.keys()):
    if k.startswith("_orig_mod."):
        state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print("Model vocab size:", model.config.vocab_size)
import tiktoken
enc = tiktoken.get_encoding("gpt2")
print("Tokenizer vocab size:", enc.n_vocab)

# ---------------------------
# LOAD DATA
# ---------------------------
with open(DATA_PATH, "r") as f:
    data = json.load(f)

enc = tiktoken.get_encoding("gpt2")

def encode(s):
    return enc.encode(s)

# ---------------------------
# LOG PROB FUNCTION
# ---------------------------
def sequence_logprob(tokens):
    tokens = tokens.unsqueeze(0)

    logits, _ = model(tokens[:, :-1], tokens[:, 1:])
    log_probs = F.log_softmax(logits, dim=-1)

    targets = tokens[:, 1:]
    selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    return selected.sum()

# ---------------------------
# TRAIN LOOP
# ---------------------------
for iter in range(MAX_ITERS):

    batch = random.sample(data, BATCH_SIZE)
    loss_total = 0

    for item in batch:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_tokens = encode(prompt + " " + chosen)[:BLOCK_SIZE]
        rejected_tokens = encode(prompt + " " + rejected)[:BLOCK_SIZE]

        chosen_tokens = torch.tensor(chosen_tokens, dtype=torch.long, device=DEVICE)
        rejected_tokens = torch.tensor(rejected_tokens, dtype=torch.long, device=DEVICE)

        logp_chosen = sequence_logprob(chosen_tokens)
        logp_rejected = sequence_logprob(rejected_tokens)

        loss = -F.logsigmoid(BETA * (logp_chosen - logp_rejected))
        loss_total += loss

    loss_total = loss_total / BATCH_SIZE

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    if iter % 50 == 0:
        print(f"iter {iter} | loss {loss_total.item():.4f}")

        torch.save({
            "model": model.state_dict(),
            "model_args": gptconf.__dict__,
        }, f"{OUT_DIR}/ckpt.pt")

# ---------------------------
# SAVE MODEL
# ---------------------------
torch.save({
    "model": model.state_dict(),
    "model_args": gptconf.__dict__,
}, f"{OUT_DIR}/ckpt.pt")

print("DPO training complete.")