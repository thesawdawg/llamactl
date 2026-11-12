"""In-app guide text shown by GuideScreen."""
from __future__ import annotations

MOE_GUIDE = """\
[b]Running MoE models on limited VRAM[/b]

A mixture-of-experts (MoE) model has many small "experts" per layer but uses only a few of
them for each token. The experts are most of the file, so they can live in RAM while
attention, the shared parts and the context cache stay on the GPU. Generation stays fast
because only the few active experts are read per token; prompt processing slows down more,
because a long prompt touches almost every expert.

[b]What llamactl does[/b]
- The plan keeps the experts of the first N layers in RAM ("CPU MoE layers", --n-cpu-moe N)
  and everything else on the GPU. The Models view shows this as GPU+experts and the plan
  shows how many layers' experts sit in RAM and what one more layer on the GPU would cost.
- The estimate is an estimate. Tune measures it: it moves more experts to RAM when a trial
  runs out of memory, then brings layers back to the GPU while the VRAM headroom allows,
  and keeps the fastest split that fits.
- Auto-fit is turned off while experts are placed by hand, because llama.cpp's fit cannot
  work around a manual placement.

[b]When to tune[/b]
- Tune every MoE model you use. The estimated split is safe, not fast.
- Tune again after changing the context, the KV cache type or the llama.cpp build.

[b]Tensor overrides (-ot)[/b]
- For placements --n-cpu-moe cannot express, for example only the down-projection experts
  in RAM, or the experts of the last layers instead of the first.
- The first matching pattern wins. llamactl writes -ot before --n-cpu-moe so your patterns
  take effect.
- Tune does not touch presets with tensor overrides: remove them to tune.

[b]WSL[/b]
- WSL can only use the memory its VM is given (memory= in .wslconfig), not all of Windows'
  RAM. Experts in RAM count against that limit, and a VM that runs out of memory starts
  swapping, which is far slower than CPU experts.
- llamactl asks once how much RAM Windows has and suggests a memory= value that leaves
  room for Windows. Run wsl --shutdown after editing .wslconfig.
"""

NOLOAD_GUIDE = """\
[b]When a model will not load[/b]

llamactl checks whether your llama.cpp build knows the architecture in a model's GGUF
header before it plans or serves it.

[b]What the marks mean[/b]
- "update llama.cpp": the architecture exists in llama.cpp master but not in your build.
  Rebuild llama.cpp (below); the mark goes away on its own once the new build knows the
  architecture.
- "unsupported": llama.cpp itself (master) does not run this architecture. The model may
  be meant for another runtime such as vLLM, SGLang, TensorRT-LLM or MLX.
  A failed load that logs 'unknown model architecture' is remembered for that build.
- "not a text model": the file is a projector (clip/mmproj), draft or audio model; it
  cannot be served as a text model.
- In Discover, a warning mark on the architecture column means the same for the repo's
  architecture as listed on Hugging Face.

[b]Rebuilding llama.cpp[/b]
@REBUILD@

[b]This check is not all-inclusive[/b]
Passing it does not guarantee a model loads or replies well:
- models only meant for other runtimes (vLLM, SGLang, TensorRT-LLM, MLX) with unofficial
  GGUF conversions may still fail;
- a newer variant released under an old architecture name can fail on older builds;
- chat template or tool-call format of a model may be unusable by this llama.cpp version;
- a missing or incompatible mmproj file breaks vision models;
- a broken or partial download fails regardless of the architecture;
- a quant without kernels for your backend loads but is unusably slow or crashes.
"""


def noload_guide(rebuild: str) -> str:
    """The "When a model will not load" guide page with the rebuild commands filled in.

    Args:
        rebuild: The one-line rebuild command block (`support.rebuild_commands`).

    Returns:
        The guide text.
    """
    return NOLOAD_GUIDE.replace("@REBUILD@", rebuild)
