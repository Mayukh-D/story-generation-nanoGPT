# P15ab — RoPE + RMSNorm combined
# Both changes from P10n: wpe removed (RoPE) + LayerNorm → RMSNorm
# Everything else identical to P10n
# Run from nanoGPT9 folder
# Run with: python train.py config/train_rocstories_p15ab.py

out_dir = '/content/drive/MyDrive/nanoGPT9/out-rocstories-p15ab'
dataset = 'rocstories'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
wandb_log = False

batch_size = 128
gradient_accumulation_steps = 1
block_size = 128
n_layer = 20
n_head = 8
n_embd = 256
dropout = 0.2
weight_decay = 0.2
learning_rate = 2.5e-4
max_iters = 12000
lr_decay_iters = 12000
min_lr = 2e-5
beta2 = 0.99
warmup_iters = 200
dtype = 'bfloat16'
compile = True
