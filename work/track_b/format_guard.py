"""Post-LLM output validator for Track B answers.

Hard-enforces the answer schema dictated by each task family:

    fault     — `node;ip-or-port;reason` lines, reason in the per-question
                vocabulary, English ASCII only, no leading/trailing whitespace,
                no blank lines.
    path      — `node(port)->node(port)->...` per line (or bare `node->node->...`
                when the question allows it).
    topology  — `local_node(local_port)->remote_node(remote_port)` per line.

Format errors zero out a question. The guard returns a structured report so
the caller can either accept normalised output or trigger exactly one re-emit
with a corrective hint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Strict ASCII subset accepted in answer text. Excludes Chinese fullwidth
# punctuation and other Unicode that violates "all symbols use English chars".
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
# Node identifier: letters, digits, underscore, hyphen, dot (for FQDN-like names)
_NODE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
# Port identifier: e.g. GE1/0/1, 10GE1/0/1, GigabitEthernet0/0/0, Vlanif120
_PORT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9/._-]*$")
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

# Fault-line layout: <field>;<field>;<field> with no embedded ; in any field.
_FAULT_LINE = re.compile(r"^([^;]+);([^;]+);([^;]+)$")
# Path token. Accepts three shapes a question may ask for:
#   - bare node:          `Core_SW_01`                  (Phase-1 path style)
#   - node(port):         `Core_SW_01(GE1/0/1)`         (legacy documented schema)
#   - node_iface:         `Core_SW_01_GE1/0/1`          (Phase-2 path style)
# The body character class must allow `/` so interface names like
# `GE1/0/1` and `10GE1/0/1` can sit inside the token without parens.
_PATH_TOKEN = re.compile(r"^([A-Za-z][A-Za-z0-9_./:-]*)(?:\(([^)]+)\))?$")
_NODE_UNDERSCORE_IFACE = re.compile(r"^(.+)_([A-Za-z][A-Za-z0-9/.-]*)$")


@dataclass
class ValidationReport:
    is_valid: bool
    family: str
    line_count: int
    error_count: int
    errors: list[str] = field(default_factory=list)
    normalised: str = ""

    def hint_for_reemit(self) -> str:
        """Concise corrective hint to append to a re-prompt."""
        if self.is_valid:
            return ""
        head = f"Output rejected by format guard ({self.error_count} errors). "
        body = "; ".join(self.errors[:5])
        return head + body + ". Re-emit using the exact required schema."


def _walk_lines(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (raw_nonblank_lines, stripped_lines, whitespace_errors).

    Detects three classes of whitespace deviation per the spec
    ("no blank lines between lines, no extra whitespace before, after, or
    within each line"):
        - blank lines anywhere in the answer
        - leading or trailing whitespace on any line
        - tab characters
    These are reported as errors AND auto-corrected in the stripped output so
    the caller can choose to recover (silent fix) or trigger a re-emit.
    """
    raw_lines: list[str] = []
    stripped: list[str] = []
    errors: list[str] = []
    saw_nonblank = False
    for raw in text.splitlines():
        if raw.strip() == "":
            if saw_nonblank:
                errors.append("blank line in answer body (forbidden)")
            continue
        saw_nonblank = True
        if "\t" in raw:
            errors.append(f"tab character in line: {raw!r}")
        if raw != raw.strip():
            errors.append(f"leading or trailing whitespace: {raw!r}")
        raw_lines.append(raw)
        stripped.append(raw.strip())
    return raw_lines, stripped, errors


def _normalize_text(text: str) -> str:
    _, stripped, _ = _walk_lines(text)
    return "\n".join(stripped)


def _is_valid_ip(s: str) -> bool:
    if not _IPV4.fullmatch(s):
        return False
    return all(0 <= int(p) <= 255 for p in s.split("."))


def _check_fault_line(
    line: str,
    routing_vocab: frozenset[str],
    port_vocab: frozenset[str],
) -> str | None:
    """Validate a single stripped fault line against the schema.

    Whitespace and blank-line issues are caught upstream by `_walk_lines`; this
    function inspects only the schema, vocabulary, and field shapes.
    """
    if _NON_ASCII.search(line):
        return f"non-ASCII characters in line: {line!r}"
    m = _FAULT_LINE.match(line)
    if not m:
        return f"line does not match <field>;<field>;<field>: {line!r}"
    node, mid, reason = m.group(1), m.group(2), m.group(3)
    for label, val in (("node", node), ("middle", mid), ("reason", reason)):
        if val != val.strip():
            return f"surrounding whitespace inside {label} field: {line!r}"
    if not _NODE_ID.match(node):
        return f"invalid node identifier {node!r}: {line!r}"
    is_routing = _is_valid_ip(mid)
    is_port = _PORT_ID.match(mid) is not None
    if not (is_routing or is_port):
        return f"middle field {mid!r} is neither a valid IPv4 nor a port id: {line!r}"
    if is_routing:
        if reason not in routing_vocab:
            return f"reason {reason!r} not in routing-fault vocabulary: {line!r}"
    else:
        if reason not in port_vocab:
            return f"reason {reason!r} not in port-fault vocabulary: {line!r}"
    return None


def _path_token_has_interface(token: str) -> bool:
    m = _PATH_TOKEN.match(token)
    if not m:
        return False
    if m.group(2):
        return True
    um = _NODE_UNDERSCORE_IFACE.match(token)
    if not um:
        return False
    iface = um.group(2).lower()
    return (
        "/" in iface
        or iface.startswith((
            "ge",
            "10ge",
            "xge",
            "eth",
            "gi",
            "fa",
            "te",
            "fo",
            "hu",
            "gigabitethernet",
            "fastethernet",
            "tengigabitethernet",
            "fortygige",
            "hundredgige",
            "vlanif",
            "vbdif",
        ))
    )


def _check_path_line(
    line: str,
    *,
    require_intermediate_interfaces: bool = False,
    forbid_final_interface: bool = False,
) -> str | None:
    if _NON_ASCII.search(line):
        return f"non-ASCII characters in line: {line!r}"
    if "->" not in line:
        return f"path line missing '->' separator: {line!r}"
    tokens = line.split("->")
    if len(tokens) < 2:
        return f"path line has only one segment: {line!r}"
    for t in tokens:
        if not _PATH_TOKEN.match(t):
            return f"invalid path token {t!r}: {line!r}"
    if require_intermediate_interfaces:
        for t in tokens[:-1]:
            if not _path_token_has_interface(t):
                return f"path token missing outbound interface {t!r}: {line!r}"
    if forbid_final_interface and _path_token_has_interface(tokens[-1]):
        return f"destination path token must not include outbound interface: {tokens[-1]!r}"
    return None


def _check_topology_line(line: str) -> str | None:
    if _NON_ASCII.search(line):
        return f"non-ASCII characters in line: {line!r}"
    if "->" not in line:
        return f"topology line missing '->' separator: {line!r}"
    tokens = line.split("->")
    if len(tokens) != 2:
        return f"topology line must have exactly one '->' link: {line!r}"
    for t in tokens:
        if not _PATH_TOKEN.match(t):
            return f"invalid topology token {t!r}: {line!r}"
    return None


def _run(text: str, line_validator) -> tuple[list[str], int, str]:
    _, stripped, ws_errors = _walk_lines(text)
    errors = list(ws_errors)
    for line in stripped:
        err = line_validator(line)
        if err:
            errors.append(err)
    if not stripped:
        errors.append("empty answer")
    return errors, len(stripped), "\n".join(stripped)


def validate_fault(
    text: str,
    routing_vocab: tuple[str, ...] | frozenset[str],
    port_vocab: tuple[str, ...] | frozenset[str],
) -> ValidationReport:
    rv = frozenset(routing_vocab)
    pv = frozenset(port_vocab)
    errors, line_count, normalised = _run(text, lambda line: _check_fault_line(line, rv, pv))
    return ValidationReport(
        is_valid=not errors,
        family="fault",
        line_count=line_count,
        error_count=len(errors),
        errors=errors,
        normalised=normalised,
    )


def validate_path(
    text: str,
    *,
    require_intermediate_interfaces: bool = False,
    forbid_final_interface: bool = False,
) -> ValidationReport:
    errors, line_count, normalised = _run(
        text,
        lambda line: _check_path_line(
            line,
            require_intermediate_interfaces=require_intermediate_interfaces,
            forbid_final_interface=forbid_final_interface,
        ),
    )
    return ValidationReport(
        is_valid=not errors,
        family="path",
        line_count=line_count,
        error_count=len(errors),
        errors=errors,
        normalised=normalised,
    )


def validate_topology(text: str) -> ValidationReport:
    errors, line_count, normalised = _run(text, _check_topology_line)
    return ValidationReport(
        is_valid=not errors,
        family="topology",
        line_count=line_count,
        error_count=len(errors),
        errors=errors,
        normalised=normalised,
    )
