from timm.data import Mixup

import torch

def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)

def mixup_target(target, lam=1., smoothing=0.0):
    """
    This function assumes that the input target is already one-hot encoded.
    """
    off_value = smoothing / target.shape[1]
    on_value = 1. - smoothing + off_value
    y1 = target * on_value + (1. - target) * off_value
    y2 = target.flip(0) * on_value + (1. - target.flip(0)) * off_value
    return y1 * lam + y2 * (1. - lam)


class CutMixUp(Mixup):
    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, lam, self.label_smoothing)
            
        return x, target