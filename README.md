## Attribution

This code is based on code taken from `github:tensorflow/privacy` for our summer research.
The original PATE paper this is based on is [arXiv:1610.05755](https://arxiv.org/abs/1610.05755).

## Dependencies

This requires `torch`, `torchvision`, `scipy`, `numpy`, and `tensorboard`. So far (fingers crossed!) we've been able to use never versions of all these packages without much difficulty.

## How to run

For our purposes, we train teachers like so:

```sh
python torch_teachers.py
```

