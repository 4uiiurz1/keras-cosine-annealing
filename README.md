# Keras implementation of Cosine Annealing Scheduler
This repository contains code for **Cosine Annealing Scheduler** based on [SGDR: Stochastic Gradient Descent with Warm Restarts
](https://arxiv.org/abs/1608.03983) implemented in Keras.

## Requirements
- Python 3.6
- Keras 2.2.4

## Usage
Append **CosineAnnealingScheduler** to list of callbacks and pass to `.fit()` or `.fit_generator()`:
```python
from cosine_annealing import CosineAnnealingScheduler

callbacks = [
    CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
]

model.fit(x, y, batch_size=32, callbacks=callbacks)
```

## Training
### CIFAR-10
Use CosineAnnealingScheduler:
```
python train.py --scheduler CosineAnnealingScheduler
```

## Results
### CIFAR-10
| Model                                      |   Accuracy (%)    |   Loss   |
|:-------------------------------------------|:-----------------:|:--------:|
| WideResNet28-2 baseline                    |             92.91 | **0.403**|
| WideResNet28-2 w/ CosineAnnealingScheduler |         **93.22** |     0.413|
