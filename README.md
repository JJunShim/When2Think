<div align="center">

# When2Think
### Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models

**Jaejun Shim<sup>1</sup>, HyunJin Kim<sup>1</sup>, Young Jin Kim<sup>2</sup>, JinYeong Bak<sup>1</sup>**

<sup>1</sup> TODO: Affiliation &nbsp;&nbsp; <sup>2</sup> TODO: Affiliation

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.19671)
[![HF Paper](https://img.shields.io/badge/🤗-Paper-ffd21e)](https://huggingface.co/papers/2609.19671)
[![Models](https://img.shields.io/badge/🤗-Models-ffd21e)](https://huggingface.co/collections/junshim/when2think)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## 🔥 News

- **[2026-09-17]** Paper released on [arXiv](https://arxiv.org/abs/2609.19671).
- **[2026-09-18]** Featured on [Hugging Face Papers](https://huggingface.co/papers/2609.19671).
- **[2026-09-2X]** `When2Think-1.5B` and `When2Think-ThinkOnly-1.5B` checkpoints released.

## 📌 TL;DR

> Large reasoning models **overthink easy problems and underthink hard ones**.
> **When2Think** is an RL post-training framework that learns *when to think* (System 2) and *when to answer directly* (System 1),
> allocating computation per instance based on problem difficulty.
>
> On **AIME24**, Pass@3 improves by **+10.0%** while token usage drops by **27.9%** vs. the base model.
> On **AIME25**, When2Think reaches **40.0% Pass@3**, outperforming compression-only and routing-only baselines.

<div align="center">
  <img src="assets/teaser.png" width="90%" alt="When2Think teaser">
  <br>
  <em>Figure 1. TODO: accuracy–token trade-off plot or Think/NoThink behavior by difficulty.</em>
</div>

## 💡 Motivation

- **Problem:** Uniform length penalties and rigid routing pay an *efficiency tax*: fewer tokens on easy instances, but accuracy loss on hard ones.
- **Key idea:** Treat efficient reasoning as an **instance-adaptive computation allocation** problem.
- **Our approach:** Shape rewards with per-instance reference statistics (accuracy, token usage) so the model directly answers easy problems and keeps extended reasoning for hard ones.

## ✨ Highlights

- 🧠 **Learns When to Think** — a single model adaptively switches between **NoThink** (System 1) and **Think** (System 2).
- ⚡ **Efficient without the tax** — −27.9% tokens and +10.0% Pass@3 on AIME24 relative to the base model.
- 🪶 **Critic-free & lightweight** — no learned reward model, no online reference-model queries, no critic.
- 📦 **Standalone inference** — no router, verifier, or difficulty estimator needed at test time.

## 🧩 Method

<div align="center">
  <img src="assets/method.png" width="90%" alt="When2Think method overview">
  <br>
  <em>Figure 2. Overview of When2Think. TODO: replace with the method figure.</em>
</div>

- **IDAC (Instance-level Difficulty-Aware Control):** reward shaping that uses pre-computed reference statistics (accuracy and token usage) to regulate reasoning depth per instance.
- **Verifier-based rewards:** verifiable correctness signal (RLVR).
- **BWS (Batch-Wise Standardization):** converts trajectory rewards into standardized advantages for stable **critic-free** PPO-style optimization.
- **Importance sampling:** balances exploration between Think and NoThink modes during training.

## 📊 Main Results

<!-- TODO: paper의 main table 수치로 교체 (Base / compression baselines / routing baselines / Ours) -->

| Method | AIME24 Pass@3 | AIME24 Tokens | AIME25 Pass@3 | MATH-500 | ... |
|---|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Qwen-1.5B (base) | TODO | TODO | TODO | TODO | |
| Compression baseline(s) | TODO | TODO | TODO | TODO | |
| Routing baseline(s) | TODO | TODO | TODO | TODO | |
| **When2Think-1.5B (Ours)** | **base +10.0%** | **base −27.9%** | **40.0%** | **TODO** | |

<div align="center">
  <img src="assets/results.png" width="80%" alt="accuracy-efficiency trade-off">
  <br>
  <em>Figure 3. TODO: accuracy–efficiency trade-off / Think ratio by difficulty.</em>
</div>

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers accelerate
# optional, for fast serving
pip install vllm
```

### Inference (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "junshim/When2Think-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
```

### Serving (vLLM)

```bash
vllm serve "junshim/When2Think-1.5B"
```

The model decides on its own whether to emit an explicit `<think>...</think>` trace (Think) or answer directly (NoThink).

<details>
<summary><b>Parsing reasoning vs. final answer</b></summary>

```python
import re

ASSISTANT_RE = re.compile(r"<｜Assistant｜>(.*?)(?=<｜User｜>|<｜end▁of▁sentence｜>|$)", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def parse_deepseek_r1(text: str) -> list[dict]:
    dialogue = []
    for match in ASSISTANT_RE.finditer(text):
        assistant = match.group(1).strip()
        think = THINK_RE.search(assistant)
        if think:
            reasoning = think.group(1).strip()
            content = (assistant[:think.start()] + assistant[think.end():]).strip()
        else:
            reasoning, content = None, assistant
        dialogue.append({"reasoning": reasoning, "content": content})
    return dialogue
```

</details>

## 📦 Released Resources

| Resource | Description | Link |
|---|---|---|
| `When2Think-1.5B` | Main hybrid reasoning model (post-trained from DeepSeek-R1-Distill-Qwen-1.5B) | [🤗 HF](https://huggingface.co/junshim/When2Think-1.5B) |
| `When2Think-ThinkOnly-1.5B` | TODO: one-line description of this variant | [🤗 HF](https://huggingface.co/junshim/When2Think-ThinkOnly-1.5B) |
| Collection | All When2Think artifacts | [🤗 Collection](https://huggingface.co/collections/junshim/when2think) |

**Training data:** [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)

## 🔁 Reproducing Results

<!-- TODO: 실제 repo 구조에 맞게 수정 -->

```bash
# 1. Pre-compute reference statistics (accuracy / token usage) for IDAC
bash scripts/compute_reference.sh

# 2. RL post-training
bash scripts/train.sh

# 3. Benchmark evaluation (AIME24/25, MATH-500, MMLU-Pro, ...)
bash scripts/eval.sh --model junshim/When2Think-1.5B
```

## 🗺️ Roadmap

- [x] Paper on arXiv
- [x] Model checkpoints (1.5B)
- [ ] Training code
- [ ] Evaluation pipeline
- [ ] Larger model sizes

## 📝 Citation

```bibtex
@misc{shim2026when2think,
  title         = {When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models},
  author        = {Jaejun Shim and HyunJin Kim and Young Jin Kim and JinYeong Bak},
  year          = {2026},
  eprint        = {2609.19671},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.19671}
}
```

## 🙏 Acknowledgements

Built on [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) and trained with [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset). TODO: add RL framework / compute / funding acknowledgements.

## 📬 Contact

Jaejun Shim · TODO: email · or open an issue.

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

⭐ If you find this work useful, please consider starring the repo! ⭐

</div>
