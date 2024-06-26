## Attribution

This code is based on code taken from `github:tensorflow/privacy` for our summer research.
The original PATE paper this is based on is [arXiv:1610.05755](https://arxiv.org/abs/1610.05755).

## Dependencies

This requires `tensorflow`, `scipy`, `numpy`, and `six`. The original code required older versions, but so far (fingers crossed!) we've been able to use never versions of all these packages without much difficulty.

## How to run

For our purposes, we train teachers like so:

```sh
python train_teachers.py --nb_teachers=250 --teacher_id=ID --dataset=svhn
```

Then we train the student with a similar line, `python train_student.py --nb_teachers=250 --dataset=svhn --stdnt_share=5000`. Read the `tf.flags.DEFINE_XXX` lines in the code, especially in `train_teachers.py`, for more details on the flags.
