"""Runtime-only compatibility helpers for the vendored author release."""

from __future__ import annotations

from typing import Any, Callable


def install_no_cudagraph_compile() -> Callable[..., Any]:
  """Keep torch.compile while removing only CUDA-graph capture.

  The released Zebra model compiles flex attention with ``reduce-overhead``
  and also globally forces Inductor CUDA graphs.  Torch 2.7.1 on the local
  RTX 3090 stack fails inside the CUDA-graph allocator during long validation.
  This shim translates only ``reduce-overhead`` compilation to the default
  Inductor mode and clears the global CUDA-graph flag at compile time.  The
  flex-attention operator, model graph, parameters, and loss are unchanged.
  """
  import torch
  import torch._inductor.config as inductor_config

  original_compile = torch.compile
  if getattr(original_compile, "_dcache_no_cudagraphs", False):
    return original_compile

  def compile_without_cudagraphs(*args, **kwargs):
    if kwargs.get("mode") == "reduce-overhead":
      kwargs = dict(kwargs)
      kwargs["mode"] = "default"
    inductor_config.triton.cudagraphs = False
    return original_compile(*args, **kwargs)

  compile_without_cudagraphs._dcache_no_cudagraphs = True
  compile_without_cudagraphs._dcache_original_compile = original_compile
  torch.compile = compile_without_cudagraphs
  return compile_without_cudagraphs
