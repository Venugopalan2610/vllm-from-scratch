"""tvllm - the model that hosts your parts: a plain-torch Qwen3 ("t" for torch,
as "j" in jvllm is for JAX).

The repo gives you this package, like jvllm/ and tests/helpers.py. You do not
edit it. tvllm/live.py runs your parts from stage 06 (`./vc run`). Read
tvllm/model.py before stage 21, where you write the real runner on it.
"""

from tvllm.model import (DenseReference, LayerWeights, Model, ModelConfig,
                         attend_to_context, causal_mask, linear, load_model)

__all__ = ["DenseReference", "LayerWeights", "Model", "ModelConfig",
           "attend_to_context", "causal_mask", "linear", "load_model"]
