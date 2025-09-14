# nanogpt

## training

```bash
# using shakespeare text
python train.py config/train_shakespeare_char.py
# using OpenWebText
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

## baselines

```bash
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

| model       | params | train loss | val loss |
| ----------- | ------ | ---------- | -------- |
| gpt2        | 124M   | 3.12       | 3.11     |
| gpt2-medium | 354M   | 2.83       | 2.83     |
| gpt2-large  | 773M   | 2.67       | 2.67     |
| gpt2-xl     | 1556M  | 2.56       | 2.56     |
