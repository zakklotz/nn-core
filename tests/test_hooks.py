import torch

from nncore.models import MoEConfig, Transformer, TransformerConfig


class ZeroHiddenHook:
    def on_hidden(self, h, **kwargs):
        return torch.zeros_like(h)

    def on_logits(self, logits, **kwargs):
        return logits

    def on_loss(self, loss_dict, **kwargs):
        return loss_dict


class AddOneLogitsHook:
    def on_hidden(self, h, **kwargs):
        return h

    def on_logits(self, logits, **kwargs):
        return logits + 1.0

    def on_loss(self, loss_dict, **kwargs):
        return loss_dict


class InjectLossHook:
    def on_hidden(self, h, **kwargs):
        return h

    def on_logits(self, logits, **kwargs):
        return logits

    def on_loss(self, loss_dict, **kwargs):
        out = dict(loss_dict)
        out["hook/test"] = torch.tensor(5.0, device=next(iter(loss_dict.values())).device if loss_dict else "cpu")
        return out


def _make_model():
    cfg = TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        max_seq_len=16,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    model = Transformer(config=cfg)
    model.eval()
    return model


def test_hidden_hook_modifies_outputs():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randint(0, 64, (2, 8))

    logits_base = model(x)
    logits_hook = model(x, hooks=[ZeroHiddenHook()])

    assert not torch.allclose(logits_base, logits_hook)


def test_logits_hook_shifts_outputs():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randint(0, 64, (2, 8))

    logits_base = model(x)
    logits_hook = model(x, hooks=[AddOneLogitsHook()])

    assert torch.allclose(logits_hook, logits_base + 1.0)


def test_loss_hook_injects_aux_key():
    torch.manual_seed(0)
    cfg = TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        max_seq_len=16,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    cfg.block.ffn_type = "moe"
    cfg.block.moe = MoEConfig(num_experts=4, top_k=2, aux_loss=True)

    model = Transformer(config=cfg)
    model.eval()

    x = torch.randint(0, 64, (1, 8))
    _, aux = model(x, return_aux=True, hooks=[InjectLossHook()])

    assert "hook/test" in aux


def test_no_hooks_behavior_unchanged():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randint(0, 64, (2, 8))
    a = model(x)
    b = model(x, hooks=None)
    assert torch.allclose(a, b)
