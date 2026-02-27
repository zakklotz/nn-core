import math


class AverageMeter:
    """
    Tracks running average of a scalar.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class EMAMeter:
    """
    Exponential moving average meter.
    """

    def __init__(self, beta: float = 0.98):
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be between 0 and 1.")
        self.beta = beta
        self.value = None

    def update(self, x: float):
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x

    @property
    def avg(self) -> float:
        return 0.0 if self.value is None else self.value
