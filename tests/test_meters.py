from nncore.utils import AverageMeter, EMAMeter


def test_average_meter():
    m = AverageMeter()
    m.update(2.0)
    m.update(4.0)
    assert abs(m.avg - 3.0) < 1e-6


def test_ema_meter():
    m = EMAMeter(beta=0.9)
    m.update(1.0)
    m.update(1.0)
    assert m.avg > 0.0
