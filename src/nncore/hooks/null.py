import torch.nn as nn


class NullHook(nn.Module):
    def on_hidden(self, h, **kwargs):
        return h

    def on_logits(self, logits, **kwargs):
        return logits

    def on_loss(self, loss_dict, **kwargs):
        return loss_dict
