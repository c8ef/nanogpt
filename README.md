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
| gpt2        | 117M   |            |          |
| gpt2-medium | 345M   |            |          |
| gpt2-large  | 762M   |            |          |
| gpt2-xl     | 1542M  |            |          |
