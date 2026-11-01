"""Extract and anonymise the final prose answer so a grader cannot infer the
condition (no code, JSON, tool traces, checker output or method vocabulary)."""
import re

FINAL = re.compile(r"^\s*(?:#+\s*|\*\*)?FINAL ANSWER(?:\*\*)?\s*:?\s*(?:\*\*)?\s*$", re.I | re.M)
FINAL_INLINE = re.compile(r"(?:#+\s*|\*\*)?FINAL ANSWER(?:\*\*)?\s*:\s*", re.I)

METHOD_TERMS = [
    r"R\+\+(?: Core)?", r"\bworld model\b", r"\bformal model\b", r"\bthe model\b", r"\bmy model\b",
    r"\bour model\b", r"\bchecker\b", r"\bverdict\b", r"\bclingo\b", r"\bz3\b", r"\bASP\b", r"\banswer set\b",
    r"\bsimulat\w*\b", r"\bpython\b", r"\bscript\b", r"\bcode\b", r"\btool\b", r"\bfluents?\b",
    r"\bscenario s\d+\b", r"\bclaim \w+\b", r"\bobligations?\b", r"\bderivation\b", r"\bJSON\b", r"\bAST\b",
    r"\bschema\b", r"\bintegrity constraint\b", r"\bper the run\b", r"\bthe run\b", r"\boutput shows\b",
]
TERM_RE = re.compile("|".join(METHOD_TERMS), re.I)
CODE_BLOCK = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`([^`]*)`")
SNAKE_ID = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
CLAIM_ID = re.compile(r"\b(?:c|o|a|x|k|p|s)\d{1,3}\b")
STEP = re.compile(r"\bt\s*=\s*(\d+)\b")


def final_answer(text):
    if not text:
        return ""
    ms = list(FINAL.finditer(text))
    if ms:
        return text[ms[-1].end():].strip()
    ms = list(FINAL_INLINE.finditer(text))
    if ms:
        return text[ms[-1].end():].strip()
    paras = [p for p in text.strip().split("\n\n") if p.strip()]
    return "\n\n".join(paras[-3:])


def anonymise(text):
    t = CODE_BLOCK.sub(" ", text)
    t = INLINE_CODE.sub(lambda m: m.group(1).replace("_", " "), t)
    t = SNAKE_ID.sub(lambda m: m.group(0).replace("_", " "), t)
    t = TERM_RE.sub("[…]", t)
    t = CLAIM_ID.sub("", t)
    t = STEP.sub(r"step \1", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


MAX_WORDS = 600


def grader_text(raw_response):
    words = anonymise(final_answer(raw_response)).split(" ")
    if len(words) > MAX_WORDS:
        words = words[:MAX_WORDS] + ["[truncated]"]
    return " ".join(words)
