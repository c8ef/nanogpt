"""
Sample from a trained model
"""

import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant
out_dir = "out"  # ignored if init_from is not 'resume'
start = "\n"  # or '<|endoftext|>' or etc. Can also specify a file: 'FILE:prompt.txt'
num_samples = 10
max_new_tokens = 500
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200  # retain the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True
exec(Path("configurator.py").read_text())
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# model init
if init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."  # introduced by torch.compile
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, {"dropout": 0.0})

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(string):
        return [stoi[c] for c in string]

    def decode(indices):
        return "".join([itos[i] for i in indices])

else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")

    def encode(string):
        return enc.encode(string, allowed_special={"<|endoftext|>"})

    def decode(indices):
        return enc.decode(indices)


# encode the beginning of the prompt
if start.startswith("FILE:"):
    start = Path(start[5:]).read_text("utf-8")
start_ids = encode(start)
# [None, ...] add the batch_size dimension
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad(), ctx:
    for _ in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature, top_k)
        print(decode(y[0].tolist()))
        print("-------------------")
