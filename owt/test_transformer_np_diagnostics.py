"""Native auxiliary scoring must use the branch, not final main features."""
import numpy as np
import torch
from owt.test_transformer_np import make_model, batch
from owt.transformer_np_diagnostics import native_score_batch, FeatureMap


def test_native_scoring_reuses_main_and_projects_each_correct_branch():
    torch.manual_seed(84)
    model = make_model(dropout=0.).eval()
    model.backbone.force_fp32_eval = True
    with torch.no_grad():
        for index, (head, branch) in enumerate(zip(model.backbone.neighbor_heads.heads, model.backbone.neighbor_branches)):
            head[-1].weight.normal_()
            branch.block.adaLN_modulation.bias.fill_(.2 + index * .3)
    clean = batch()[0].numpy()
    masked = np.zeros_like(clean, dtype=bool)
    masked[:, 2:8] = True
    canvas = clean.copy()
    canvas[masked] = model.mask_index
    wrong = np.zeros_like(masked)
    rows, arrays = native_score_batch(model, clean, canvas, masked, wrong, 'cpu')
    with torch.inference_mode():
        main, features = model.diagnostic_forward(torch.from_numpy(canvas), torch.zeros(2, 1))
        for offset, head in zip(model.backbone.neighbor_heads.offsets, model.backbone.neighbor_heads.heads):
            direction = 'left' if offset == 1 else 'right'
            for row in range(2):
                targets = np.flatnonzero(arrays[direction+'_prediction'][row] >= 0)
                logits = head(features[offset][row, targets-offset]).clone()
                logits[:, model.mask_index] = -torch.inf
                logp = logits.log_softmax(-1)
                expected = logp.gather(1, torch.tensor(clean[row, targets])[:, None]).squeeze(1).numpy()
                np.testing.assert_allclose(arrays[direction+'_true_logp'][row, targets], expected, rtol=1e-6)
        selected = arrays['scored']
        expected = -main.gather(-1, torch.from_numpy(clean)[..., None]).squeeze(-1).numpy()
        for row, values in enumerate(rows):
            assert np.isclose(values['masked_ce_sum'], expected[row, selected[row]].astype(float).sum())
    assert model.branch_calls == 0
    assert all(values['head_union_targets'] > 0 for values in rows)


def test_native_features_reject_lower_precision():
    import pytest
    with pytest.raises(ValueError, match='FP32'):
        FeatureMap({-1: torch.ones(1, 2, 3).bfloat16(), 1: torch.ones(1, 2, 3)})
