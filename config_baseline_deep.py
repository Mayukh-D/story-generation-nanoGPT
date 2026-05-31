# Pass 10N — Friend's LR recipe + deep architecture
# n_layer=20, n_head=8, n_embd=256, batch=128, block=128 — 28.64M params
# lr=2.5e-4, min_lr=2e-5, weight_decay=0.06, warmup=200, 12000 steps
# Combining friend's optimizer settings with deep narrow architecture
# Run with: python train.py config/train_rocstories_p10n.py

out_dir = '/content/drive/MyDrive/nanoGPT5/out-rocstories-p10n'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
wandb_log = False

dataset = 'rocstories'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 128

n_layer = 20
n_head = 8
n_embd = 256
dropout = 0.2
label_smoothing = 0.0

learning_rate = 2.5e-4
min_lr = 2e-5
max_iters = 12000
lr_decay_iters = 12000
warmup_iters = 200
weight_decay = 0.06
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

device = 'cuda'
compile = False
dtype = 'bfloat16'
