"""Runtime compatibility for retained one-hop VJPs through compiled attention."""
def retain_compiled_backward_buffers():
    import torch._functorch.config as config
    # AOT donated buffers cannot be reused by autograd.grad(retain_graph=True)
    # followed by the optimizer backward. Apply identically to both TT arms.
    config.donated_buffer=False
