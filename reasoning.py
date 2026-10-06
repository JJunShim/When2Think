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
