"""Tests for the baseline stack: DeltaRNN, quantization, ablation maskers,
and RigL on the dense backbones (including LTC via ncps).

Everything on random tensors, CPU, ``fast_dev_run``, no logger/checkpointing,
temp root dirs: no clutter left behind.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dsrnn.nn.quantization import quantize_tensor

B, T, D, H, C = 8, 10, 12, 16, 4


def _loader():
    x = torch.randn(4 * B, T, D)
    y = torch.randint(0, C, (4 * B,))
    return DataLoader(TensorDataset(x, y), batch_size=B)


def test_quantize_tensor():
    x = torch.randn(32)
    assert torch.equal(quantize_tensor(x, 8, 8, q_type=0), x)  # identity in training
    q = quantize_tensor(x, 2, 4, q_type=1)                     # round to 4 frac bits
    assert torch.allclose(q * 16, torch.round(q * 16), atol=1e-5)
    # STE gradient: 1 inside the representable range, 0 where the value
    # saturates (the final clamp in quantize_tensor kills the gradient there).
    xg = x.clone().requires_grad_(True)
    quantize_tensor(xg, 2, 4, q_type=1).sum().backward()
    power = 16.0                      # 2 ** qf
    clip = float(2 ** (2 + 4 - 1))    # qi=2, qf=4 -> 32
    inside = (xg.detach() * power >= -clip) & (xg.detach() * power <= clip - 1)
    assert torch.allclose(xg.grad[inside], torch.ones_like(xg.grad[inside]))
    assert torch.all(xg.grad[~inside] == 0)


def test_deltagru_forward():
    from dsrnn.nn.deltarnn import DeltaGRU
    rnn = DeltaGRU(input_size=D, hidden_size=H, num_layers=1, thx=0.1, thh=0.1, debug=0)
    rnn.eval()
    x = torch.randn(T, B, D)  # (T, B, F)
    out, state, reg = rnn(x)
    assert out.shape == (T, B, H)
    assert reg.ndim == 0 and torch.isfinite(reg)


def test_deltalstm_forward():
    from dsrnn.nn.deltarnn import DeltaLSTM
    rnn = DeltaLSTM(input_size=D, hidden_size=H, num_layers=1, thx=0.1, thh=0.1, debug=0)
    rnn.eval()
    x = torch.randn(T, B, D)
    out, state, reg = rnn(x)
    assert out.shape == (T, B, H)
    assert torch.isfinite(reg)


def test_deltarnn_classifier_smoke():
    import tempfile
    from lightning import Trainer
    from dsrnn.training.deltarnn import DeltaRNNClassifier

    model = DeltaRNNClassifier(input_size=D, hidden_size=H, num_classes=C,
                               num_layers=1, task_type="classification",
                               model_type="gru", debug=0)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=tmp, accelerator="cpu")
        trainer.fit(model, train_dataloaders=_loader(), val_dataloaders=_loader())
        trainer.test(model, dataloaders=_loader())


def test_maskers_budgets_and_shapes():
    from dsrnn.training.ablation import (LearnedGlobalMasker, RandomDynamicMasker,
                                         StructuredSNIPMasker)
    x_t = torch.randn(B, D)
    h = torch.randn(B, H)
    for cls in (StructuredSNIPMasker, LearnedGlobalMasker, RandomDynamicMasker):
        m = cls(input_size=D, hidden_size=H, sparsity=0.5)
        m_x, m_h = m.get_masks(x_t, h)
        assert m_x.shape == (B, D) and m_h.shape == (B, H)
        if cls is RandomDynamicMasker:
            assert (m_x.sum(dim=-1) == m.k_x).all()
            assert (m_h.sum(dim=-1) == m.k_h).all()


def test_learned_masker_gradient():
    from dsrnn.training.ablation import LearnedGlobalMasker
    m = LearnedGlobalMasker(input_size=D, hidden_size=H, sparsity=0.5)
    m_x, _ = m.get_masks(torch.randn(B, D), torch.randn(B, H))
    m_x.sum().backward()
    assert m.logits_x.grad is not None


def test_masked_rnn_forward_all_cells():
    from dsrnn.training.ablation import MaskedRNN
    x = torch.randn(B, T, D)
    for model_type in ("lstm", "gru", "ltc"):
        rnn = MaskedRNN(model_type, input_size=D, hidden_size=H, sparsity=0.5,
                        masker_type="random")
        out, h = rnn(x)
        assert out.shape == (B, T, H), (model_type, out.shape)


def test_ablation_module_smoke_and_snip_init():
    import tempfile
    from lightning import Trainer
    from dsrnn.training.ablation import AblationLitModule, StructuredSNIPMasker

    module = AblationLitModule(
        model_type="lstm", masker_type="static", input_size=D, hidden_size=H,
        output_size=C, sparsity=0.5, return_sequences=False, loss_type="ce",
        metric_params={"accuracy": {"task": "multiclass", "num_classes": C,
                                    "average": "macro"}},
    )
    module.maybe_init_static_mask(_loader())
    assert isinstance(module.rnn.masker, StructuredSNIPMasker)
    assert module.rnn.masker.initialized
    assert module.rnn.masker.mask_x.sum().item() == module.rnn.masker.k_x

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=tmp, accelerator="cpu")
        trainer.fit(module, train_dataloaders=_loader(), val_dataloaders=_loader())
        trainer.test(module, dataloaders=_loader())


def test_rigl_on_dense_ltc():
    """Dense LTC backbone + RigL masks through SequenceModule."""
    import tempfile
    from lightning import Trainer
    from dsrnn.training.pruning import (MaskCallback, RigLCallback, apply_masks_to_model,
                                        count_nonzero_masks, make_initial_masks)
    from dsrnn.training.sequence import SequenceModule

    torch.manual_seed(0)
    module = SequenceModule(
        model_type="ltc", input_size=D, hidden_size=H, output_size=C,
        return_sequences=False, loss_type="ce",
        metric_params={"accuracy": {"task": "multiclass", "num_classes": C,
                                    "average": "macro"}},
        optimizer_kwargs={"all": {"lr": 1e-3, "weight_decay": 0.0}},
        test_with_corruptions=False,
    )
    masks = make_initial_masks(module, sparsity=0.5, deterministic=True)
    module.masks = masks
    apply_masks_to_model(module, masks)
    kept_before, _ = count_nonzero_masks(masks)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=tmp, accelerator="cpu",
                          callbacks=[MaskCallback(masks),
                                     RigLCallback(masks, update_frequency=1,
                                                  rewire_fraction=0.1)],
                          fast_dev_run=True)
        # pass a val loader too, to avoid Lightning's "no val_dataloader" note
        trainer.fit(module, train_dataloaders=_loader(), val_dataloaders=_loader())

    kept_after, _ = count_nonzero_masks(masks)
    assert kept_before == kept_after


def main():
    test_quantize_tensor()
    test_deltagru_forward()
    test_deltalstm_forward()
    test_deltarnn_classifier_smoke()
    test_maskers_budgets_and_shapes()
    test_learned_masker_gradient()
    test_masked_rnn_forward_all_cells()
    test_ablation_module_smoke_and_snip_init()
    test_rigl_on_dense_ltc()
    print("All baseline tests passed.")


if __name__ == "__main__":
    main()
