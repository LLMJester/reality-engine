"""Reading a JSON object out of model or program output that may have text (and stray braces) around it."""
import json
import re

_DEC = json.JSONDecoder()


def objects(text):
    """Every JSON object that parses at some '{' in the text, in order of position (outer objects only)."""
    out, end = [], -1
    for m in re.finditer(r"\{", text or ""):
        if m.start() < end:
            continue
        try:
            obj, n = _DEC.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
            end = m.start() + n
    return out


def first_object(text):
    objs = objects(text)
    if not objs:
        raise ValueError("no JSON object found")
    return objs[0]


def last_object(text):
    objs = objects(text)
    if not objs:
        raise ValueError("no JSON object found")
    return objs[-1]
