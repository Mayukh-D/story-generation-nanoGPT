"""
sample.py (Mistral tokenizer + CLI via configurator.py + 5-sentence stop)
"""
import os
from contextlib import nullcontext
import torch
from transformers import AutoTokenizer
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' or a gpt2 variant
out_dir = 'out'
start = "\n"  # or "FILE:prompt.txt"
num_samples = 10
max_new_tokens = 120
temperature = 0.7
top_k = 40
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())  # CLI overrides
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---------------- MODEL ----------------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError("init_from must be 'resume' or start with 'gpt2'")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# ---------------- TOKENIZER (MISTRAL ONLY) ----------------
print("Loading Mistral tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
decode = lambda l: tokenizer.decode(l)

# ---------------- PROMPT ----------------
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# ---------------- GENERATION ----------------
def stop_5_sentences(text: str) -> bool:
    # simple heuristic; works well for your dataset
    return text.count('.') >= 5

with torch.no_grad():
    with ctx:
        for _ in range(num_samples):
            y = x.clone()
            for _ in range(max_new_tokens):
                idx_cond = y if y.size(1) <= model.config.block_size else y[:, -model.config.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                y = torch.cat((y, idx_next), dim=1)

                text = decode(y[0].tolist())
                if stop_5_sentences(text):
                    break

            print(text)
            print('---------------')
