import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import math
import pickle
import numpy as np
import torch
import tiktoken
from datasets import load_dataset

from model import GPTConfig, GPT

# -------------------
# CONFIG
# -------------------
CHECKPOINT_PATH = 'out-rocstories-p15ab/ckpt.pt'
OUT_DIR = 'data/story_mix'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOW_PERCENTILE = 10
HIGH_PERCENTILE = 70
BLOCK_SIZE = 128

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------
# LOAD MODEL
# -------------------
print("Loading model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

config = GPTConfig(**checkpoint['model_args'])
model = GPT(config)

state_dict = checkpoint['model']
state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
              for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# -------------------
# TOKENIZER (GPT-2 for scoring ONLY)
# -------------------
enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token

# -------------------
# LOAD DATA
# -------------------
print("Loading ROCStories...")
dataset = load_dataset("mintujupally/ROCStories", split="train")
lines = [row['text'] for row in dataset]

# -------------------
# PPL SCORING
# -------------------
@torch.no_grad()
def score_story(text):
    tokens = enc.encode(text) + [EOT]
    if len(tokens) < 10:
        return float('inf')

    tokens_t = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, tokens_t.size(1) - 1, BLOCK_SIZE):
        chunk = tokens_t[:, i:i+BLOCK_SIZE+1]

        if chunk.size(1) < 2:
            continue

        xc = chunk[:, :-1]
        yc = chunk[:, 1:]

        _, loss = model(xc, yc)

        total_loss += loss.item() * yc.size(1)
        total_tokens += yc.size(1)

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

# -------------------
# SCORE ALL STORIES
# -------------------
print("Scoring stories...")
ppls = []

for i, line in enumerate(lines):
    ppl = score_story(line)
    ppls.append(ppl)

    if (i+1) % 1000 == 0:
        print(f"{i+1}/{len(lines)} done")

ppls = np.array(ppls)

# -------------------
# FILTER
# -------------------
low_thresh = np.percentile(ppls, LOW_PERCENTILE)
high_thresh = np.percentile(ppls, HIGH_PERCENTILE)

print(f"Keeping PPL between {low_thresh:.2f} and {high_thresh:.2f}")

mask = (ppls >= low_thresh) & (ppls <= high_thresh)
filtered = [lines[i] for i in range(len(lines)) if mask[i]]

print(f"Kept {len(filtered)} / {len(lines)} stories")

# -------------------
# SAVE FILTERED TEXT
# -------------------
filtered_path = os.path.join(OUT_DIR, "filtered_roc.stories")

with open(filtered_path, "w", encoding="utf-8") as f:
    for story in filtered:
        f.write(story.strip() + "\n")

print(f"Saved filtered stories to: {filtered_path}")

# -------------------
# OPTIONAL: SAVE GPT2 BIN
# -------------------
def encode_stories(stories):
    tokens = []
    for s in stories:
        tokens.extend(enc.encode(s))
        tokens.append(EOT)
    return np.array(tokens, dtype=np.uint16)

idx = np.random.permutation(len(filtered))
split = int(0.9 * len(filtered))

train_stories = [filtered[i] for i in idx[:split]]
val_stories   = [filtered[i] for i in idx[split:]]

train_ids = encode_stories(train_stories)
val_ids   = encode_stories(val_stories)

train_ids.tofile(os.path.join(OUT_DIR, 'train.bin'))
val_ids.tofile(os.path.join(OUT_DIR, 'val.bin'))

meta = {'vocab_size': enc.n_vocab}

with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done.")
