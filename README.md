# graph2gauss in PyTorch

This is a pytorch implementation of [Deep Gaussian Embedding of Graphs:
Unsupervised Inductive Learning via
Ranking](https://openreview.net/forum?id=r1ZdKJ-0W). Run `python g2g.py -h` to
learn about training options. Below you can see an example run on the citeseer
dataset as provided in the [implementation by the original
authors](https://github.com/abojchevski/graph2gauss/).

```sh
$ python g2g.py --seed 0 --samples 3 --epochs 120 --workers 5 -k 1 citeseer.npz
LR F1 score 0.4491554535256343
Epoch 10 - Loss 35145104.000
Epoch 20 - Loss 28642094.000
Epoch 30 - Loss 21908854.000
Epoch 40 - Loss 17939046.000
Epoch 50 - Loss 14932645.000
LR F1 score 0.7872897465883691
Epoch 60 - Loss 12318501.000
Epoch 70 - Loss 11219567.000
Epoch 80 - Loss 9815305.000
Epoch 90 - Loss 8618428.000
Epoch 100 - Loss 7848496.500
LR F1 score 0.8252173651998809
```
