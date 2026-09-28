<div align="center">

# When2Think
### Learning When and How Much to Reason

**Jaejun Shim<sup>1</sup>, HyunJin Kim<sup>1</sup>, Young Jin Kim<sup>2</sup>, JinYeong Bak<sup>1</sup>**

<sup>1</sup> Sungkyunkwan University &nbsp;&nbsp; <sup>2</sup> Microsoft

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.19671)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/JJunShim/When2Think)
[![HF Paper](https://img.shields.io/badge/🤗-Paper-ffd21e)](https://huggingface.co/papers/2609.19671)
[![Models](https://img.shields.io/badge/🤗-Models-ffd21e)](https://huggingface.co/collections/junshim/when2think)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## 🔥 News

- **[2026-09-17]** The [When2Think preprint](https://arxiv.org/abs/2609.19671) is available on arXiv.
- **[2026-09]** Released [When2Think-1.5B](https://huggingface.co/junshim/When2Think-1.5B), the full hybrid checkpoint.
- **[2026-09]** Released [When2Think-ThinkOnly-1.5B](https://huggingface.co/junshim/When2Think-ThinkOnly-1.5B), the THINK-only checkpoint.
- **[2026-09]** Opened the [When2Think Hugging Face Collection](https://huggingface.co/collections/junshim/when2think).

## 📌 TL;DR

> Large reasoning models **overthink easy problems and underthink hard ones**.
> **When2Think** is an RL post-training framework that learns *when to think* (System 2) and *when to answer directly* (System 1),
> allocating computation per instance based on problem difficulty.
>
> On **AIME24**, Pass@3 improves by **+10.0 pp** while token usage drops by **27.9%** vs. the base model.
> On **AIME25**, When2Think reaches **40.0% Pass@3**, outperforming compression-only and routing-only baselines.

## 💡 Motivation

- **Problem:** Uniform length penalties and rigid routing pay an *efficiency tax*: fewer tokens on easy instances, but accuracy loss on hard ones.
- **Key idea:** Treat efficient reasoning as an **instance-adaptive computation allocation** problem.
- **Our approach:** Shape rewards with per-instance reference statistics (accuracy, token usage) so the model directly answers easy problems and keeps extended reasoning for hard ones.

## ✨ Highlights

- 🧠 **Learns When to Think** — a single model adaptively switches between **NoThink** (System 1) and **Think** (System 2).
- ⚡ **Efficient without the tax** — −27.9% tokens and +10.0 pp Pass@3 on AIME24 relative to the base model.
- 🪶 **Critic-free & lightweight** — no learned reward model, no online reference-model queries, no critic.
- 📦 **Standalone inference** — no router, verifier, or difficulty estimator needed at test time.

## 🧩 Method

- **IDAC (Instance-level Difficulty-Aware Control):** reward shaping that uses pre-computed reference statistics (accuracy and token usage) to regulate reasoning depth per instance.
- **Verifier-based rewards:** verifiable correctness signal (RLVR).
- **BWS (Batch-Wise Standardization):** converts trajectory rewards into standardized advantages for stable **critic-free** PPO-style optimization.
- **Importance sampling:** balances exploration between Think and NoThink modes during training.

## 📊 Main Results

### Accuracy and token usage

| Model | GSM-Plus Pass@3 ↑ | GSM-Plus Tokens ↓ | AIME24 Pass@3 ↑ | AIME24 Tokens ↓ | AIME25 Pass@3 ↑ | AIME25 Tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|
| R1-Distill-Qwen | 79.4 | **590** | 46.0 | 14,195 | 32.0 | 12,616 |
| DeepScaleR-Preview | 85.4 | 1,358 | **58.0** | 8,473 | 39.3 | 8,074 |
| AdaptThink | 83.6 | 716 | 44.7 | **5,806** | 30.7 | **6,883** |
| ThinkLess | 85.8 | 1,799 | 46.7 | 11,023 | 33.3 | 11,056 |
| **When2Think** | 85.7 | 1,052 | 56.0 | 10,236 | **40.0** | 9,549 |
| **When2Think-ThinkOnly** | **86.8** | 1,652 | 57.3 | 10,046 | **40.0** | 9,846 |

> Results report Pass@3 accuracy and average generated tokens per response over five independent sampling runs. The strongest value in each displayed column is bolded. Different models may occupy different accuracy-computation operating points.

### Representative comparison with the backbone

| Benchmark | Metric | R1-Distill-Qwen | When2Think | Difference |
|---|---|---:|---:|---:|
| AIME24 | Pass@3 ↑ | 46.0 | **56.0** | **+10.0 pp** |
| AIME24 | Tokens ↓ | 14,195 | **10,236** | **−27.9%** |
| AIME25 | Pass@3 ↑ | 32.0 | **40.0** | **+8.0 pp** |
| AIME25 | Tokens ↓ | 12,616 | **9,549** | **−24.3%** |


## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/JJunShim/When2Think.git
cd When2Think
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
vllm serve junshim/When2Think-1.5B --reasoning-parser deepseek_r1
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

## 📬 Contact

Jaejun Shim (`junshim@skku.edu`) · or open an [issue](https://github.com/JJunShim/When2Think/issues).
Corresponding authors: Young Jin Kim (`youki@microsoft.com`), JinYeong Bak (`jy.bak@skku.edu`).

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

⭐ If you find this work useful, please consider starring the repo! ⭐

</div>
