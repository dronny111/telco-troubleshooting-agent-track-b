"""Deterministic constraint extractor for Track B fault questions.

Pulls the free-text symptom suffix's structured signals out of the question
text without invoking the LLM. The extracted fields feed two downstream
consumers:

    1. The Qwen system prompt — as compact JSON ranker context (Step 9).
    2. The XGBoost feature matrix — as binary/scalar features (Step 8):
       target_ip_match, blacklisted_node, denied_command_count (joined later
       from the manifest), disclosed_fault_category_match.

The parser is intentionally conservative: it returns Optional fields and
empty lists rather than guessing. False positives here propagate into the
ranker and the validator, so when a pattern is ambiguous we omit it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .fault_vocab import PROTOCOL_FAMILIES

# ---- Suffix isolation ------------------------------------------------------
# The format-spec boilerplate ends with port-fault example
# "Beta-Node-01;GE1/0/2;interface IP error". Everything after that line is
# the question-specific symptom text. Some questions also include the
# boilerplate sentence:
#   "Please provide the most specific fault reason (the Huawei firewall in
#    this question is not planned to deploy NAT functionality)."
# That sentence is NOT a per-question signal — it appears verbatim in every
# Phase 2 fault question — so we strip it before running pattern detectors.
# The format-spec example block always ends with a line like:
#   "Beta-Node-01;GE1/0/2;interface IP error"
# The marker `;interface IP error` only occurs in that example line — the
# format-spec list itself uses "(2) interface IP error" without a leading
# semicolon — so splitting on this marker is unambiguous. We tolerate both
# real newlines and literal `\n` because the JSON source uses escaped
# newlines that survive into the question string.
_SUFFIX_SPLIT = re.compile(r";\s*interface\s+IP\s+error\b", re.IGNORECASE)
_BOILERPLATE_SENTENCE = re.compile(
    r"Please\s+provide\s+the\s+most\s+specific\s+fault\s+reason\s*\([^)]*\)\.\s*",
    re.IGNORECASE,
)


def extract_suffix(question: str) -> str:
    """Return only the question's free-text symptom suffix.

    Strips the format-spec preamble and the boilerplate "Please provide..."
    sentence. If no split marker is found (e.g., a non-fault question),
    returns the input unchanged.
    """
    parts = _SUFFIX_SPLIT.split(question, maxsplit=1)
    suffix = parts[1] if len(parts) > 1 else question
    # JSON-escaped \n survives as literal backslash+n; normalise so downstream
    # patterns match either real or escaped newlines uniformly.
    suffix = suffix.replace("\\n", "\n")
    suffix = _BOILERPLATE_SENTENCE.sub("", suffix)
    # Drop any leading punctuation/whitespace that remains after the split
    return suffix.lstrip(" \t\n\r,.;").strip()


# ---- Symptom-suffix heuristics ----------------------------------------------

# Phase 2: "From SOURCE, accessing ..." or "From SOURCE(<note>), accessing ..."
_FROM_SOURCE = re.compile(
    r"From\s+([A-Z][A-Za-z0-9_]+)\s*(?:\([^)]*\))?\s*,\s*accessing",
)
# Phase 2: "accessing 10.1.60.2" or "accessing HOST(10.1.60.2)"
_ACCESS_IP = re.compile(
    r"accessing\s+(?:[^,()]*?\()?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\)?",
)
_ACCESS_HOST = re.compile(
    r"accessing\s+(?:[a-z][a-zA-Z\s]*?\s+)?([A-Z][A-Za-z0-9_]+)\s*\(",
)
# Phase 1: "ping 10.1.1.20 from Hermes-Prime-01 is unreachable"
_PING_FROM = re.compile(
    r"ping\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s+from\s+([A-Z][A-Za-z0-9_-]+)\s+is\s+(?:unreachable|interrupted|failing|failed)",
    re.IGNORECASE,
)
# Phase 1: "Beta-Node-01 is addressing GE1/0/2 of Gamma-Node-01"
_ADDRESSING = re.compile(
    r"\b([A-Z][A-Za-z0-9_-]+)\s+is\s+addressing\s+([A-Z][A-Za-z0-9/_.-]+)\s+of\s+([A-Z][A-Za-z0-9_-]+)",
)
# "Limitation: Do not look for faults on X." possibly followed by ", and Y" etc.
_BLACKLIST = re.compile(
    r"Limitation:\s*Do\s+not\s+look\s+for\s+faults?\s+on\s+([^.\n]+)\.",
    re.IGNORECASE,
)
# Bare form without "Limitation:" prefix
_BLACKLIST_BARE = re.compile(
    r"\bDo\s+not\s+look\s+for\s+faults?\s+on\s+([^.\n]+?)\.",
    re.IGNORECASE,
)
# "the Huawei firewall in this question is not planned to deploy NAT functionality"
_VENDOR_EXCLUSION = re.compile(
    r"the\s+([A-Za-z][A-Za-z0-9-]+)\s+(firewall|device|router|switch)\s+in\s+this\s+question\s+is\s+not\s+planned\s+to\s+deploy\s+([A-Za-z0-9-]+)\s+functionality",
    re.IGNORECASE,
)
# Disclosed fault patterns we have actually seen
_VRRP_DUAL_MASTER = re.compile(
    r"([A-Z][A-Za-z0-9_]+)\s+and\s+([A-Z][A-Za-z0-9_]+)\s+on\s+the\s+corresponding\s+(Vlanif\d+)\s+VRRP\s+dual[\s-]master",
)
# Fault node hint: "the fault is on one of the three nodes: A, B, C"
_FAULT_CANDIDATE_LIST = re.compile(
    r"fault\s+is\s+on\s+one\s+of\s+(?:the\s+)?(?:two|three|four|several|the)?\s*nodes?:\s*([^.\n]+?)\.",
    re.IGNORECASE,
)
# Protocol-keyword scan (whole-word match with our small protocol vocabulary)
# Compile once with case-insensitive matching.
_PROTO_PATTERNS = {
    fam: re.compile(rf"\b{re.escape(fam)}\b", re.IGNORECASE) for fam in PROTOCOL_FAMILIES
}
# Disclosed-fault category cues: keywords linked to specific fault reasons.
_FAULT_CATEGORY_CUES: tuple[tuple[str, str], ...] = (
    (r"\bVRRP\s+dual[\s-]master\b", "VRRP-dual-master"),
    (r"\bblackhole\s+route\b", "blackhole route"),
    (r"\bglobal\s+STP\s+not\s+enabled\b", "global STP not enabled"),
    (r"\bport\s+STP\s+not\s+enabled\b", "port STP not enabled"),
    (r"\bMTU\b", "MTU value configuration error"),
    (r"\bARP\s+configuration\b", "ARP configuration error"),
    (r"\bloop(?:back)?\s+IP\s+conflict\b", "loopback IP configuration conflict"),
    (r"\bsecurity\s+policy\s+rule\b", "security policy rule not permitting corresponding users"),
)
_FAULT_CATEGORY_REGEXES = tuple((re.compile(p, re.IGNORECASE), tag) for p, tag in _FAULT_CATEGORY_CUES)


def _split_blacklist_value(raw: str) -> list[str]:
    """Split 'X', 'X and Y', 'X, Y, and Z' into a clean node list."""
    s = raw.strip()
    s = re.sub(r"\s+and\s+", ",", s, flags=re.IGNORECASE)
    parts = [p.strip().rstrip(".") for p in s.split(",")]
    out: list[str] = []
    for p in parts:
        if not p or p.lower() in {"and", ""}:
            continue
        # Keep only tokens that look like a node identifier
        if re.match(r"^[A-Za-z][A-Za-z0-9_-]*$", p):
            out.append(p)
    return out


@dataclass(frozen=True)
class ParsedConstraints:
    source_endpoint: str | None = None
    source_port: str | None = None  # only set by the "X is addressing PORT of Y" pattern
    target_destination_node: str | None = None
    target_destination_host: str | None = None
    target_destination_ip: str | None = None
    blacklisted_nodes: tuple[str, ...] = ()
    disclosed_fault_nodes: tuple[str, ...] = ()
    disclosed_vlanif: str | None = None
    disclosed_fault_categories: tuple[str, ...] = ()
    suspected_protocol_families: tuple[str, ...] = ()
    fault_candidate_nodes: tuple[str, ...] = ()
    vendor_exclusions: tuple[tuple[str, str], ...] = ()  # (vendor, capability)
    suffix: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def parse(question: str) -> ParsedConstraints:
    """Parse a Track B question's free-text constraints.

    Pattern detection runs over the symptom suffix only (everything after the
    format-spec boilerplate). Boilerplate sentences shared across questions
    do not contribute features.
    """
    suffix = extract_suffix(question)

    src = None
    src_port = None
    dest_node = None
    m = _FROM_SOURCE.search(suffix)
    if m:
        src = m.group(1)
    if src is None:
        m = _PING_FROM.search(suffix)
        if m:
            src = m.group(2)
    if src is None:
        m = _ADDRESSING.search(suffix)
        if m:
            src = m.group(1)
            src_port = m.group(2)
            dest_node = m.group(3)

    dest_ip = None
    m = _ACCESS_IP.search(suffix)
    if m:
        dest_ip = m.group(1)
    if dest_ip is None:
        m = _PING_FROM.search(suffix)
        if m:
            dest_ip = m.group(1)

    dest_host = None
    m = _ACCESS_HOST.search(suffix)
    if m:
        dest_host = m.group(1)

    blacklist: list[str] = []
    for rx in (_BLACKLIST, _BLACKLIST_BARE):
        for m in rx.finditer(suffix):
            blacklist.extend(_split_blacklist_value(m.group(1)))
    seen = set()
    deduped_blacklist = []
    for n in blacklist:
        if n not in seen:
            seen.add(n)
            deduped_blacklist.append(n)

    disclosed_nodes: list[str] = []
    disclosed_vlanif: str | None = None
    m = _VRRP_DUAL_MASTER.search(suffix)
    if m:
        disclosed_nodes.extend([m.group(1), m.group(2)])
        disclosed_vlanif = m.group(3)

    fault_candidates: list[str] = []
    m = _FAULT_CANDIDATE_LIST.search(suffix)
    if m:
        fault_candidates = _split_blacklist_value(m.group(1))

    disclosed_categories: list[str] = []
    for rx, tag in _FAULT_CATEGORY_REGEXES:
        if rx.search(suffix):
            disclosed_categories.append(tag)

    proto_families: list[str] = []
    for fam, rx in _PROTO_PATTERNS.items():
        if rx.search(suffix):
            proto_families.append(fam)

    vendor_excl: list[tuple[str, str]] = []
    for m in _VENDOR_EXCLUSION.finditer(suffix):
        vendor_excl.append((m.group(1), m.group(3).upper()))

    return ParsedConstraints(
        source_endpoint=src,
        source_port=src_port,
        target_destination_node=dest_node,
        target_destination_host=dest_host,
        target_destination_ip=dest_ip,
        blacklisted_nodes=tuple(deduped_blacklist),
        disclosed_fault_nodes=tuple(disclosed_nodes),
        disclosed_vlanif=disclosed_vlanif,
        disclosed_fault_categories=tuple(disclosed_categories),
        suspected_protocol_families=tuple(proto_families),
        fault_candidate_nodes=tuple(fault_candidates),
        vendor_exclusions=tuple(vendor_excl),
        suffix=suffix,
    )
