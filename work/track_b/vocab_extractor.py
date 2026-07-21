"""Extract per-question fault vocabulary from a fault-classification question.

Each fault question embeds its own canonical reason list inside the format spec.
Phase 1 and Phase 2 use slightly different wording (e.g.,
"traffic congestion on port bandwidth" vs "traffic occupying port bandwidth")
and different reason counts (13/7 in Phase 1 vs 19/12 in Phase 2). Hardcoding
the list breaks one phase or the other; extracting per question is authoritative.

Header wording also varies:
    Phase 2: "Routing fault output format: fault-node;destination-IP;fault-reason."
    Phase 1: "Routing fault output format: fault node;destination IP;fault reason."

The regex below is hyphen/space-tolerant and case-insensitive.
"""

from __future__ import annotations

import re

# Tolerate three intro phrasings observed across Phase 1 / Phase 2:
#   "Routing fault output format: ..."
#   "The routing fault output format is: ..."
#   "The output format for routing faults is: ..."
# Header line ends with "fault[\s-]reason." then "Fault reasons include: <items>".
_ROUTING_INTRO = (
    r"(?:Routing\s+fault\s+output\s+format:"
    r"|The\s+routing\s+fault\s+output\s+format\s+is:"
    r"|The\s+output\s+format\s+for\s+routing\s+faults\s+is:)"
)
_PORT_INTRO = (
    r"(?:Port\s+fault\s+output\s+format:"
    r"|The\s+port\s+fault\s+output\s+format\s+is:"
    r"|The\s+output\s+format\s+for\s+port\s+faults\s+is:)"
)
_ROUTING_HEADER = re.compile(
    rf"{_ROUTING_INTRO}[^.]*?fault[\s-]reason\.\s*Fault\s+reasons\s+include:\s*(.*?)"
    r"(?="
    r"Port\s+fault\s+output\s+format:"
    r"|The\s+port\s+fault\s+output\s+format\s+is:"
    r"|The\s+output\s+format\s+for\s+port\s+faults\s+is:"
    r"|if\s+there\s+are\s+multiple\s+fault\s+reasons,"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_PORT_HEADER = re.compile(
    rf"{_PORT_INTRO}[^.]*?fault[\s-]reason\.\s*Fault\s+reasons\s+include:\s*(.*?)"
    r"(?="
    r"if\s+there\s+are\s+multiple\s+fault\s+reasons,"
    r"|Please\s+provide\s+the\s+most\s+specific"
    r"|Routing\s+fault\s+examples:"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_NUMBERED_ITEM = re.compile(r"\(\d+\)\s*([^;)]+?)(?:;|\.|$)")


def _extract_items(block: str) -> list[str]:
    items: list[str] = []
    for m in _NUMBERED_ITEM.finditer(block):
        item = m.group(1).strip()
        item = item.rstrip(" .;")
        if item:
            items.append(item)
    return items


def extract_fault_vocab(question: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (routing_reasons, port_reasons) extracted verbatim from the question.

    Returns empty tuples if the question is not a fault-classification task or
    if the format spec is malformed. Callers should classify the task first.
    """
    routing: list[str] = []
    port: list[str] = []
    rm = _ROUTING_HEADER.search(question)
    if rm:
        routing = _extract_items(rm.group(1))
    pm = _PORT_HEADER.search(question)
    if pm:
        port = _extract_items(pm.group(1))
    return tuple(routing), tuple(port)
