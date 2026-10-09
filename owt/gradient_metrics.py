"""Read-only summaries of the accumulated joint gradient before L2 clipping."""
import torch


GRADIENT_COLUMNS = ['optimizer_step', 'joint_l2', 'shared_trunk_l2',
                    'main_readout_l2', 'neighbor_readouts_l2',
                    'clip_limit', 'estimated_clip_multiplier', 'elapsed_seconds']


@torch.no_grad()
def gradient_norms(module, clip_limit):
    groups = {'shared_trunk_l2': [], 'main_readout_l2': [], 'neighbor_readouts_l2': []}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith('backbone.neighbor_heads.'):
            group = 'neighbor_readouts_l2'
        elif name.startswith('backbone.output_layer.linear.'):
            group = 'main_readout_l2'
        else:
            group = 'shared_trunk_l2'
        groups[group].append(torch.linalg.vector_norm(parameter.grad.detach(), 2))
    # Only scalar reductions are copied to CPU. Gradients and RNG are untouched.
    squared = {name: float(torch.stack(values).double().square().sum()) if values else 0.
               for name, values in groups.items()}
    result = {name: value**.5 for name, value in squared.items()}
    result['joint_l2'] = sum(squared.values())**.5
    limit = float(clip_limit or 0.)
    result['clip_limit'] = limit
    result['estimated_clip_multiplier'] = (
        min(1., limit/(result['joint_l2']+1e-6)) if limit > 0 else 1.)
    return result
