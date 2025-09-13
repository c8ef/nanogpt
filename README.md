# nanogpt

## training

```bash
# using shakespeare text
python train.py config/train_shakespeare_char.py
# using OpenWebText
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```
