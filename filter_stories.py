"""
Perplexity-based ROCStories filter.
Scores every training story using P15ab checkpoint,
then keeps stories in the 10th-70th percentile PPL range (most representative).

Run from nanoGPT9 folder:
  pip install datasets -q
  python filter_stories.py

Output: data/rocstories_filtered/train.bin, val.bin, meta.pkl
"""

import os
import math
import pickle
import numpy as np
import torch
import tiktoken
from datasets import load_dataset

# ── CONFIG ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = '/content/drive/MyDrive/nanoGPT9/out-rocstories-p15ab/ckpt.pt'
OUT_DIR         = 'data/rocstories_filtered'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
LOW_PERCENTILE  = 10
HIGH_PERCENTILE = 70
BLOCK_SIZE      = 128
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
from model import GPTConfig, GPT
config = GPTConfig(**checkpoint['model_args'])
model = GPT(config)
state_dict = checkpoint['model']
state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
              for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token

print("Loading stories...")
data = np.fromfile('data/rocstories/train.bin', dtype=np.uint16).tolist()
lines = []
current = []
for t in data:
    if t == EOT:
        if current:
            lines.append(enc.decode(current))
            current = []
    else:
        current.append(t)
print(f"Total stories: {len(lines)}")

@torch.no_grad()
def score_story(text):
    tokens = enc.encode(text) + [EOT]
    if len(tokens) < 10:
        return float('inf')
    tokens_t = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, tokens_t.size(1) - 1, BLOCK_SIZE):
        xc = tokens_t[:, i:i+BLOCK_SIZE]
        yc = tokens_t[:, i+1:i+BLOCK_SIZE+1]
        if xc.size(1) == 0:
            break
        logits, loss = model(xc, yc)
        total_loss += loss.item() * xc.size(1)
        total_tokens += xc.size(1)
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

print("Scoring stories (~5 mins)...")
ppls = []
for i, line in enumerate(lines):
    ppl = score_story(line)
    ppls.append(ppl)
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{len(lines)} scored, latest PPL: {ppl:.2f}")

ppls = np.array(ppls)
low_thresh  = np.percentile(ppls, LOW_PERCENTILE)
high_thresh = np.percentile(ppls, HIGH_PERCENTILE)
print(f"\nPPL range: {ppls.min():.2f} - {ppls.max():.2f}")
print(f"Keeping PPL {low_thresh:.2f} to {high_thresh:.2f} ({LOW_PERCENTILE}th-{HIGH_PERCENTILE}th percentile)")

mask = (ppls >= low_thresh) & (ppls <= high_thresh)
filtered = [lines[i] for i in range(len(lines)) if mask[i]]
print(f"Kept {len(filtered)} / {len(lines)} stories ({100*len(filtered)/len(lines):.1f}%)")

np.random.seed(42)
idx = np.random.permutation(len(filtered))
split = int(0.9 * len(filtered))
train_stories = [filtered[i] for i in idx[:split]]
val_stories   = [filtered[i] for i in idx[split:]]
print(f"Train: {len(train_stories)}, Val: {len(val_stories)}")

def encode_stories(stories):
    tokens = []
    for s in stories:
        tokens.extend(enc.encode(s))
        tokens.append(EOT)
    return np.array(tokens, dtype=np.uint16)

print("Encoding and saving...")
train_ids = encode_stories(train_stories)
val_ids   = encode_stories(val_stories)

train_ids.tofile(os.path.join(OUT_DIR, 'train.bin'))
val_ids.tofile(os.path.join(OUT_DIR, 'val.bin'))

meta = {'vocab_size': enc.n_vocab, 'itos': {}, 'stoi': {}}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nDone!")
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")
print(f"Saved to: {OUT_DIR}")