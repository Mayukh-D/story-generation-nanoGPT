# Story Language Modeling with nanoGPT

**COMP4680/ENGN8650 Mini-Project 1** — Mayukh Das (U7965027), ANU

A decoder-only transformer trained from scratch to generate coherent five-sentence stories. The twist: only 3.7M tokens of training data and a strict 30M parameter budget — forcing every design choice to be about efficiency under scarcity, not raw scale.

---

## What This Project Explores

With a data budget 160x below what Chinchilla recommends for a 30M-parameter model, conventional wisdom from large-scale training doesn't always apply. This project systematically tests:

1. **Does depth or width matter more** when data is scarce?
2. **Which modern architecture components** (RoPE, RMSNorm, SwiGLU) actually help at this scale?
3. **Can synthetic data or pretraining** compensate for limited real data?
4. **Can DPO** improve story quality beyond what perplexity captures?

---

## Results at a Glance

```
Baseline (7L, GPT-2 tokenizer)          → 25.77 PPL
+ Deeper (20L) + narrower               → 25.12 PPL
+ RoPE + RMSNorm                        → 24.67 PPL  ← final model
+ Mistral tokenizer (research only)     → 23.71 PPL  ← best overall but incompatible with eval
```

---

## Task 1: Baseline

A near-default nanoGPT configuration, tuned for the data-scarce regime.

| | |
|:---|:---|
| Architecture | 7 layers, 6 heads, 384 embedding |
| Parameters | 31.69M |
| Tokenizer | GPT-2 BPE (50,257 vocab) |
| Training | 12,000 steps, lr=2.5e-4, cosine decay |
| **Test PPL** | **25.77** |

Key tuning: lower learning rate (2.5e-4 vs 1e-3), longer training (12K steps), dropout 0.2 — all oriented toward preventing memorization on limited data.

---

## Task 2: Architecture Search

### Tokenizer — the biggest single win

Mistral's 32K vocab frees 7M parameters from the embedding table back into transformer layers:

| Tokenizer | Vocab Size | PPL |
|:---|:---:|:---:|
| GPT-2 BPE | 50,257 | 28.62 |
| Mistral | 32,000 | **23.71** |

> With GPT-2 BPE, the embedding table alone consumes 19.3M parameters — over 60% of the budget — sitting in a lookup table rather than contributing to learning.

### Modern Components

Testing LLaMA-inspired changes one at a time:

| Component | PPL | Why |
|:---|:---:|:---|
| Baseline | 25.33 | — |
| + RoPE | 24.69 | Encodes position in attention at zero parameter cost |
| + RMSNorm | 25.01 | Removes re-centering, stabilizes small-data training |
| + SwiGLU | 25.62 | Gating needs more data to learn — hurts here |
| **RoPE + RMSNorm** | **24.67** | Complementary, not additive |

### Depth vs Width

At fixed parameter count, deeper is strictly better on small data:

| Layers | Embedding | Params | PPL |
|:---:|:---:|:---:|:---:|
| 7 | 384 | 31.69M | 25.77 |
| 16 | 296 | 31.71M | 25.17 |
| **20** | **272** | **31.53M** | **25.12** |

> Each layer refines the previous layer's features. Width adds parallel capacity the model lacks data to exploit.

### Data Augmentation

Distribution match trumps quantity — every augmentation that drifts from ROCStories hurts:

| Strategy | PPL | |
|:---|:---:|:---|
| Baseline (78k stories) | 28.62 | — |
| + 2k synthetic (multi-LLM) | 30.35 | Distribution mismatch |
| + 20k synthetic (Gemini) | 30.16 | Still mismatched |
| TinyStories pretrain → finetune | 53.26 | Vocabulary too different |
| **Quality-filtered ROCStories** | **24.92** | Removing noise helps |

---

## Task 3: Final Model

The winning combination: deep architecture + RoPE + RMSNorm + quality-filtered training data.

| | |
|:---|:---|
| Architecture | 20 layers, 8 heads, 256 embedding |
| Components | RoPE + RMSNorm |
| Parameters | 28.60M |
| Training | 12,000 steps, lr=2.5e-4 |
| **Test PPL** | **24.67** |

### Generated Stories

**Conservative** (temp=0.6, top_k=20):
> *"Sarah wanted to bake a cake for her friend's birthday. She wanted to make a cake. She bought all the ingredients. She mixed it together. She made it the cake."*

**Balanced** (temp=0.65, top_k=50):
> *"Sarah wanted to bake a cake for her friend's birthday. She drove to the store and bought all the ingredients. Sarah then went home to bake the cake. She was excited to bake the cake. Sarah baked the cake and it was delicious."*

**Creative** (temp=0.8, top_k=50):
> *"Sarah wanted to bake a cake for her friend's birthday. She wanted some chocolate but her mom was out of town so she called her mom. Her mom made her a pumpkin cake and put it in the oven. She looked at the pie and she found the crust. Sarah was glad she made the cake and she ate it."*

---

## DPO Fine-tuning

Direct Preference Optimization aligns the model toward higher-quality outputs using curated preference pairs — improving narrative coherence beyond what perplexity alone captures.

---

## Key Takeaways

| Lesson | Detail |
|:---|:---|
| Tokenizer > architecture | Smaller vocab = more parameters for actual learning |
| Depth > width (small data) | Each layer refines; width adds unused capacity |
| Not all LLaMA tricks transfer | SwiGLU needs scale; RoPE/RMSNorm are universal |
| Data quality > data quantity | Synthetic data hurts unless it matches the distribution |
| PPL ≠ quality | Lower PPL doesn't always mean better stories |

---

## Structure

```
model.py                    # GPT model (RoPE + RMSNorm)
train.py                    # Training loop
train_dpo.py                # DPO fine-tuning
eval.py                     # Perplexity evaluation
sample.py                   # Story generation (tiktoken)
sample_mistral.py           # Story generation (Mistral tokenizer)
configurator.py             # Config system
filter_stories.py           # Quality filtering for data augmentation
hf_load.py                  # HuggingFace data loading
data/prepare.py             # Dataset tokenization
dpo_dataset.json            # DPO preference pairs
config_baseline_deep.py     # Deep baseline (20L/8H/256E)
config_rope_rmsnorm.py      # + RoPE + RMSNorm
config_final_model.py       # Final submission model
```

## Running

```bash
# Prepare data
python data/prepare.py

# Train the final model
python train.py config_final_model.py

# Generate stories
python sample.py --out_dir=out --num_samples=5

# DPO fine-tuning
python train_dpo.py
```

Requires Python 3.10+, PyTorch, and tiktoken. Training takes ~44 minutes on an L4 GPU.

## Report

Full writeup with training curves, loss analysis, and extended samples: [`u7965027_MiniProject1_nanoGPT-1.pdf`](u7965027_MiniProject1_nanoGPT-1.pdf).

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy (2022)
- [Scaling Laws](https://arxiv.org/abs/2001.08361) — Kaplan et al. (2020)
- [Chinchilla](https://arxiv.org/abs/2203.15556) — Hoffmann et al. (2022)
- [TinyStories](https://arxiv.org/abs/2305.07759) — Eldan & Li (2023)
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al. (2023)
