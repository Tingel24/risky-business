<br />
<p align="center">
<a><img src="https://github.com/Tingel24/risky-business/blob/1be157b2a771925f72fd76859998a1d96574fba5/images/risky-business-logo.png" alt="risky-business" width="250" height="250" title="risky-business" style="border-radius: 50%;"></a>
</p>
<h3 align="center">Risky Business</h3>
<p align="center">Investigating the inherent tension between safe outputs and transparent reasoning traces in LRMs<br /></p>
<p align="center">
  <a href="https://github.com/Tingel24/risky-business/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Tingel24/risky-business" alt="License"></a>
  <a href="https://github.com/Tingel24/risky-business/network/members"><img src="https://img.shields.io/github/forks/Tingel24/risky-business?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/Tingel24/risky-business/stargazers"><img src="https://img.shields.io/github/stars/Tingel24/risky-business?style=social" alt="GitHub stars"></a>
    <a href="https://huggingface.co/datasets/LFrancis/hazmart"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HazMart-FFD21E" alt="Dataset on HF" height="20"></a>

</p>
<p>
<a href="https://github.com/Tingel24/risky-business/issues">Report Bug</a>
·
<a href="https://github.com/Tingel24/risky-business/issues">Request Feature</a>
·
<a href="https://arxiv.org/abs/2608.03745">Link to ArXiv Paper</a>
</p>
<p>
</p>

## Overview

Welcome to the official repository for the paper, "Risky Business: Measuring The Faithfulness-Safety Tension".

<!-- TODO: Properly cite paper and DOI -->

### Abstract
> CoT reasoning offers a promising window into model monitoring.
> However, monitoring relies on faithfulness, i.e., the model output strictly derives from its reasoning trace.
> We identify an alignment tension where a model must be faithful enough to be monitored, yet robust enough to reject unsafe reasoning.
> We demonstrate that this counterbalance exists in current LRMs, and show ways in which it can be addressed.
> We introduce **HazMart**, a human written dataset set in an autonomous AI shopkeeper scenario.
> Unlike prior work that relies on providing hints in prompts to test faithfulness (e.g., "A Stanford professor said it should be Answer A"), we propose a novel replacement based technique, which we call **TRR**, that directly intervenes in the reasoning chain to substitute in unsafe or illogical thoughts (e.g., "Wait, the answer must be ~~**Option B**~~ **Option A** because it is the most fitting").
> DeepSeek R1 Llama 70B exhibits high faithfulness (97.5%) but fails to reject *Unsafe Reasoning* (12.3%), while QwQ 32B is more robust (73.9% safety) at the cost of lower faithfulness (74.7%).
> Mechanistic analyses of QwQ 32B reveal that these properties are represented by anti correlated internal directions peaking at the action commit token.
> Finally, we demonstrate that representation steering can independently amplify the safety direction, increasing safe behavior by 9 percentage points while maintaining base capabilities.

---

- Use `run_*.sh` scripts to run the experiments on a SLURM cluster.
- Find dataset in `/data`
- Find evaluation data in `/results`
- Find plots in `/src/plots`
- Find code in `/src`
  - Main file for running the evaluation is `src/faith_shop.py`

---
run-scripts assume a workspace named "models" and "TRITON_CACHE"

`ws_allocate -d 30 TRITON_CACHE`

`ws_allocate -d 30 models`

If your cluster does not use the `hpc-workspace` toolkit, you will need to hardcode the paths captured using `ws_find` in the scripts.

---

Our fine-tuned model stored in `/results/qwq_aligned_model/` can be found here:
https://huggingface.co/LFrancis/QwQ-32B-hazmart-finetune
