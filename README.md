<div align="center">

# When2Think
### Learning When and How Much to Reason

**Jaejun Shim<sup>1</sup>, HyunJin Kim<sup>1</sup>, Young Jin Kim<sup>2</sup>, JinYeong Bak<sup>1</sup>**

<sup>1</sup> Sungkyunkwan University &nbsp;&nbsp; <sup>2</sup> Microsoft

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.19671)
[![Code](https://img.shields.io/badge/Repository-GitHub-181717?logo=github&logoColor=white)](https://github.com/JJunShim/When2Think)
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

> Large reasoning models often **overthink easy problems and underthink
> hard ones**. **When2Think** is an RLVR-based post-training framework
> that jointly learns:
>
> 1. **when to reason**, by selecting between THINK and NOTHINK; and
> 2. **how much to reason**, by controlling computation within THINK.
>
> On **AIME24**, When2Think improves Pass@3 from **46.0% to 56.0%**
> while reducing average token usage from **14,195 to 10,236**,
> corresponding to **+10.0 percentage points** and **27.9% fewer
> tokens** relative to the backbone.

## 💡 Motivation

- **Problem:** Uniform length penalties and rigid routing pay an *efficiency tax*: fewer tokens on easy instances, but accuracy loss on hard ones.
- **Key idea:** Treat efficient reasoning as an **instance-adaptive computation allocation** problem.
- **Our approach:** Shape rewards with per-instance reference statistics (accuracy, token usage) so the model directly answers easy problems and keeps extended reasoning for hard ones.

## ✨ Highlights

- 🧠 **Learns When and How Much to Reason** — jointly controls
  THINK/NOTHINK selection and computation within THINK.
- ⚡ **Mitigates the efficiency tax** — improves AIME24 Pass@3 by
  10.0 percentage points while reducing token usage by 27.9%.
- 🪶 **Critic-free & lightweight** — no learned reward model, no online reference-model queries, no critic.
- 📦 **Standalone inference** — no router, verifier, or difficulty estimator needed at test time.
- 🧩 **Beyond hybrid routing** — even when explicit reasoning is always active, difficulty-aware depth control outperforms standard RFT on most evaluated benchmarks.

## 🧩 Method

1. **Reference pre-computation**  
   A reference policy estimates per-instance success and token-cost
   statistics, which are cached for subsequent policy updates.
2. **THINK/NOTHINK exploration**  
   We adopt importance-sampled exploration to maintain coverage of both
   reasoning modes during post-training.
3. **Instance-level Difficulty-Aware Control**  
   IDAC uses cached reference statistics and generated token count to
   construct a correctness-gated efficiency bonus.
4. **Batch-Wise Standardization**  
   BWS converts trajectory rewards into standardized advantages for
   stable critic-free policy optimization.

## 📊 Main Results

### Accuracy and token usage

| Model | GSM-Plus Pass@3 ↑ | Tokens ↓ | AIME24 Pass@3 ↑ | Tokens ↓ | AIME25 Pass@3 ↑ | Tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|
| R1-Distill-Qwen | 79.4 | **590** | 46.0 | 14,195 | 32.0 | 12,616 |
| DeepScaleR-Preview | 85.4 | 1,358 | **58.0** | 8,473 | 39.3 | 8,074 |
| AdaptThink | 83.6 | 716 | 44.7 | **5,806** | 30.7 | **6,883** |
| ThinkLess | 85.8 | 1,799 | 46.7 | 11,023 | 33.3 | 11,056 |
| **When2Think** | 85.7 | 1,052 | 56.0 | 10,236 | **40.0** | 9,549 |
| **When2Think-ThinkOnly** | **86.8** | 1,652 | 57.3 | 10,046 | **40.0** | 9,846 |

> Results report Pass@3 accuracy and average generated tokens per response over five independent sampling runs. The strongest value in each displayed column is bolded. Different models may occupy different accuracy-computation operating points.

### Key Takeaways

- **AIME24:** Pass@3 improves from `46.0` to `56.0`, while average
  token usage decreases from `14,195` to `10,236`.
- **AIME25:** Pass@3 improves from `32.0` to `40.0`, while average
  token usage decreases from `12,616` to `9,549`.
- **THINK-only control remains strong:** The IDAC+BWS variant reaches
  `57.3` Pass@3 on AIME24, showing that within-THINK computation
  control contributes independently of mode selection.

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
# For always-THINK behavior, use:
# model_path = "junshim/When2Think-ThinkOnly-1.5B"
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

### Serving (vLLM)

```bash
vllm serve junshim/When2Think-1.5B --reasoning-parser deepseek_r1
```

## 📦 Released Resources

| Resource | Description | Link |
|---|---|---|
| `When2Think-1.5B` | Hybrid checkpoint that learns THINK/NOTHINK selection and within-THINK computation control | [🤗 Model](https://huggingface.co/junshim/When2Think-1.5B) |
| `When2Think-ThinkOnly-1.5B` | Always-THINK checkpoint that isolates difficulty-aware computation control without hybrid mode selection | [🤗 Model](https://huggingface.co/junshim/When2Think-ThinkOnly-1.5B) |
| `When2Think Collection` | Paper and released model artifacts | [🤗 Collection](https://huggingface.co/collections/junshim/when2think) |

## 📝 Citation

```bibtex
@misc{shim2026when2think,
  title         = {When2Think: Learning When and How Much to Reason},
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
