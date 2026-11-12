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
