"""Classify each Track B question into one of three answer-shape families.

The output schema differs by family, so the format guard must dispatch on this
classification before validating an LLM draft answer.

Phase 1 and Phase 2 use slightly different punctuation in their format specs
(e.g., "fault-node;destination-IP" vs "fault node;destination IP"); regex
matchers normalise hyphens/spaces and are case-insensitive.
"""

from __future__ import annotations

import re

# fault-node;destination-IP;fault-reason  OR  fault node;destination IP;fault reason
_FAULT_RE = re.compile(
    r"fault[\s-]node\s*;\s*(?:destination[\s-]IP|fault[\s-]port)\s*;\s*fault[\s-]reason",
    re.IGNORECASE,
)
# Path-output cues
_PATH_RE = re.compile(
    r"(?:each\s+path\s+is\s+output\s+on\s+one\s+line"
    r"|give\s+the\s+path"
    r"|connect\s+node\s+names?\s+using\s+the\s+->\s+symbol"
    r"|use\s+the\s+->\s+symbol"
    r"|start[\s-]node\(outbound[\s-]port\)\s*->\s*)",
    re.IGNORECASE,
)
# Topology / link reconstruction cues. Tolerate "Remote" vs "Peer" wording.
_TOPOLOGY_RE = re.compile(
    r"(?:supplement\s+the\s+(?:link\s+information|topology|topology\s+links?)"
    r"|topology\s+links?\s+(?:for|of)"
    r"|local[\s-]end\s+node\s+name\(local[\s-]end\s+port\s+number\)"
    r"|local\s+node\s+name\(local\s+port\s+number\)\s*->\s*(?:remote|peer)\s+node\s+name"
    r"|LocalNodeName\(LocalPortNumber\)\s*->\s*(?:RemoteNodeName|PeerNodeName)"
    r"|UP\s+link\s+connections\s+of\s+this\s+node\s+need\s+to\s+be\s+restored)",
    re.IGNORECASE,
)


def classify(question: str) -> str:
    """Return one of {'fault', 'path', 'topology', 'other'}.

    Precedence: fault > topology > path > other. Fault wins because some fault
    questions also reference paths in the symptom description.
    """
    if _FAULT_RE.search(question):
        return "fault"
    if _TOPOLOGY_RE.search(question):
        return "topology"
    if _PATH_RE.search(question):
        return "path"
    return "other"
