"""Parsing and enforcing inline citations.

The model is asked to end every sentence with markers such as [2] or [1][3] that
point at the numbered sources it was given. This module turns that text into
sentences with resolved sources, and enforces the rule that makes answers
checkable: a sentence that cites nothing valid is not part of the answer.
"""

import re
from dataclasses import dataclass

_MARKER_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
# A sentence ends at terminal punctuation, including any citation markers that follow it.
_SENTENCE = re.compile(r".+?(?:[.!?](?:\s*\[[\d,;\s]+\])*(?=\s|$)|$)", re.DOTALL)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class Sentence:
    text: str
    """The sentence without its citation markers."""
    sources: list[int]
    """1-based source numbers, valid and de-duplicated, in order of appearance."""
    prefix: str = ""
    """List bullet or number that preceded the sentence, if any."""
    starts_line: bool = False


def parse_sentences(answer: str, source_count: int) -> list[Sentence]:
    """Split an answer into sentences and resolve the sources each one cites.

    Markers that point outside the provided sources are ignored: a citation to a
    source that was never shown is not evidence.
    """
    sentences: list[Sentence] = []
    for line in answer.splitlines():
        prefix_match = _LIST_PREFIX.match(line)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = line[len(prefix) :]
        first_in_line = True
        for match in _SENTENCE.finditer(body):
            raw = match.group(0).strip()
            if not raw:
                continue
            sources: list[int] = []
            for group in _MARKER_GROUP.findall(raw):
                for number in re.findall(r"\d+", group):
                    value = int(number)
                    if 1 <= value <= source_count and value not in sources:
                        sources.append(value)
            text = _MARKER_GROUP.sub("", raw)
            text = re.sub(r"\s+([.!?,;:])", r"\1", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            if not re.search(r"\w", text):
                continue
            sentences.append(
                Sentence(
                    text=text,
                    sources=sources,
                    prefix=prefix if first_in_line else "",
                    starts_line=first_in_line,
                )
            )
            first_in_line = False
    return sentences


def render(sentences: list[Sentence]) -> tuple[str, list[int]]:
    """Rebuild the answer with markers renumbered in order of first use.

    Returns the text and the original source numbers in their new order, so marker
    [n] in the text refers to the n-th entry of the returned list.
    """
    order: list[int] = []
    lines: list[str] = []
    for sentence in sentences:
        for source in sentence.sources:
            if source not in order:
                order.append(source)
        markers = "".join(f"[{order.index(source) + 1}]" for source in sentence.sources)
        rendered = f"{sentence.text} {markers}".strip()
        if sentence.starts_line or not lines:
            lines.append(sentence.prefix + rendered)
        else:
            lines[-1] += " " + rendered
    return "\n".join(lines), order
