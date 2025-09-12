"""
This training script can be run both on a single GPU in debug mode,
and also in a large training run with distributed data parallel (DDP).
"""

from pathlib import Path

import torch

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # "scratch" or "resume" or "gpt2*"
# wandb logging
wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2"  # "run" + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 40
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretrain 0 is good, for finetune try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# AdamW optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla, ref: https://arxiv.org/pdf/2203.15556
min_lr = 6e-5  # minimal learning rate, should be ~= learning_rate / 10 per Chinchilla
# DDP settings
backend = "nccl"
# system
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(Path("configurator.py").read_text())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------
