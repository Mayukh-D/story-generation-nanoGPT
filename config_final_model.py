# P16ab — RoPE + RMSNorm on 20/8/272 (P14a architecture + tonight's best changes)
# Changes from P15ab: n_embd=272, out_dir updated
# Run from nanoGPT9 folder
# Run with: python train.py config/train_rocstories_p16ab.py

out_dir = '/content/drive/MyDrive/nanoGPT9/out-rocstories-p16ab2'
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
n_embd = 272
dropout = 0.2
weight_decay = 0.2
learning_rate = 2e-4
max_iters = 15000
lr_decay_iters = 15000
min_lr = 2e-5
beta2 = 0.99
warmup_iters = 200
dtype = 'bfloat16'
compile = True
