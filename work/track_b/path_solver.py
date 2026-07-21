"""Deterministic path solver from LLDP evidence.

Used by the Phase-2 path runtime to render path answers directly when the
topology evidence is sufficient, without waiting on LLM final emission.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from .topology import parse_lldp_brief

_IPV4 = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")
_DEVICE_TOKEN = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]+)\b")
_PATH_FROM_TO = re.compile(
    r"Path\s+from\s+(.+?)\s+to\s+(.+?)(?:\(|\.)",
    re.IGNORECASE,
)
_BANK22_FROM_TO = re.compile(
    r"from\s+device\s+([A-Za-z][A-Za-z0-9_-]+)\s+to\s+device\s+([A-Za-z][A-Za-z0-9_-]+)\s+with\s+destination\s+IP\s+(\d{1,3}(?:\.\d{1,3}){3})",
    re.IGNORECASE,
)
_CFG_INTF = re.compile(r"^\s*interface\s+(\S+)\s*$", re.MULTILINE)
_TRAILING_NUM_PORT = re.compile(r"(\d+(?:/\d+)+(?:\.\d+)?)$")


@dataclass(frozen=True)
class PathQuestionSpec:
    source_node: str | None
    destination_node: str | None
    destination_ip: str | None
    join_style: str  # "underscore" | "paren"
    require_final_interface: bool
    expand_ge_name: bool


def parse_path_question_spec(question_text: str) -> PathQuestionSpec:
    q = question_text
    source: str | None = None
    dest: str | None = None
    dest_ip: str | None = None

    m = _BANK22_FROM_TO.search(q)
    if m:
        source = m.group(1)
        dest = m.group(2)
        dest_ip = m.group(3)
    else:
        m = _PATH_FROM_TO.search(q)
        if m:
            src_tokens = _DEVICE_TOKEN.findall(m.group(1))
            dst_tokens = _DEVICE_TOKEN.findall(m.group(2))
            if src_tokens:
                source = src_tokens[-1]
            if dst_tokens:
                dest = dst_tokens[-1]
            tail = q[m.start(2): m.end(2) + 64]
            mi = _IPV4.search(tail)
            if mi:
                dest_ip = mi.group(1)

    ql = q.lower()
    join_style = "underscore" if "'_' symbol" in ql or '"_" symbol' in ql else "paren"
    require_final_interface = "end-node(inbound-port)" in ql
    expand_ge_name = "full name gigabitethernet instead of ge" in ql

    return PathQuestionSpec(
        source_node=source,
        destination_node=dest,
        destination_ip=dest_ip,
        join_style=join_style,
        require_final_interface=require_final_interface,
        expand_ge_name=expand_ge_name,
    )


def _normalize_port(port: str, *, expand_ge_name: bool) -> str:
    p = port.strip()
    if expand_ge_name and p.upper().startswith("GE"):
        return "GigabitEthernet" + p[2:]
    return p


def _port_key(port: str) -> str:
    p = port.strip()
    m = _TRAILING_NUM_PORT.search(p)
    if m:
        return m.group(1)
    return p.lower()


def parse_current_configuration_ports(content: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _CFG_INTF.finditer(content):
        name = m.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_interface_brief_ports(content: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in content.splitlines():
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9/._-]*)\s+", line)
        if not m:
            continue
        name = m.group(1).strip()
        if name.lower() in {"interface", "brief"}:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


class PathEvidenceGraph:
    """Stores LLDP-derived directional links with per-side port labels."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, set[tuple[str, str]]]] = {}
        self._port_inventory: dict[str, dict[str, str]] = {}

    def add_link(
        self,
        *,
        src_dev: str,
        src_port: str,
        dst_dev: str,
        dst_port: str,
        bidirectional: bool = True,
    ) -> None:
        self._edges.setdefault(src_dev, {}).setdefault(dst_dev, set()).add((src_port, dst_port))
        if bidirectional:
            self._edges.setdefault(dst_dev, {}).setdefault(src_dev, set()).add((dst_port, src_port))

    def ingest_lldp(self, *, local_device: str, lldp_output: str) -> int:
        rows = parse_lldp_brief(lldp_output)
        n = 0
        for local_if, nbr_if, nbr_dev in rows:
            self.add_link(
                src_dev=local_device,
                src_port=local_if,
                dst_dev=nbr_dev,
                dst_port=nbr_if,
                bidirectional=True,
            )
            n += 1
        return n

    def ingest_port_inventory(
        self,
        *,
        device: str,
        content: str,
        source: str,
    ) -> int:
        if source == "current_config":
            ports = parse_current_configuration_ports(content)
        else:
            ports = parse_interface_brief_ports(content)
        inv = self._port_inventory.setdefault(device, {})
        n = 0
        for port in ports:
            key = _port_key(port)
            prev = inv.get(key)
            if prev is None or len(port) > len(prev):
                inv[key] = port
                n += 1
        return n

    def neighbors(self, node: str) -> list[str]:
        return sorted(self._edges.get(node, {}).keys())

    def get_ports(self, src_dev: str, dst_dev: str) -> tuple[str, str] | None:
        pairs = self._edges.get(src_dev, {}).get(dst_dev)
        if not pairs:
            return None
        return sorted(pairs)[0]

    def canonicalize_port(self, *, device: str, observed: str, expand_ge_name: bool) -> str:
        inv = self._port_inventory.get(device, {})
        exact = inv.get(_port_key(observed))
        if exact:
            return exact
        return _normalize_port(observed, expand_ge_name=expand_ge_name)

    def find_paths(
        self,
        *,
        source: str,
        destination: str,
        max_hops: int = 14,
        max_paths: int = 6,
    ) -> list[list[str]]:
        if source == destination:
            return [[source]]
        out: list[list[str]] = []
        q: deque[list[str]] = deque([[source]])
        shortest_len: int | None = None
        while q:
            path = q.popleft()
            cur = path[-1]
            hop_len = len(path) - 1
            if hop_len > max_hops:
                continue
            if shortest_len is not None and hop_len > shortest_len:
                continue
            if cur == destination:
                shortest_len = hop_len
                out.append(path)
                if len(out) >= max_paths:
                    break
                continue
            for nxt in self.neighbors(cur):
                if nxt in path:
                    continue
                q.append(path + [nxt])
        return out

    def render_path(self, path: list[str], spec: PathQuestionSpec) -> str | None:
        if len(path) < 2:
            return None
        if spec.join_style == "underscore":
            toks: list[str] = []
            for i, node in enumerate(path):
                if i == len(path) - 1:
                    toks.append(node)
                    continue
                ports = self.get_ports(node, path[i + 1])
                if not ports:
                    return None
                out_port = self.canonicalize_port(
                    device=node,
                    observed=ports[0],
                    expand_ge_name=spec.expand_ge_name,
                )
                toks.append(f"{node}_{out_port}")
            return "->".join(toks)

        # parenthesized style
        toks = []
        for i, node in enumerate(path):
            if i == 0:
                ports = self.get_ports(node, path[i + 1])
                if not ports:
                    return None
                out_port = self.canonicalize_port(
                    device=node,
                    observed=ports[0],
                    expand_ge_name=spec.expand_ge_name,
                )
                toks.append(f"{node}({out_port})")
                continue
            if i == len(path) - 1:
                if spec.require_final_interface:
                    ports = self.get_ports(path[i - 1], node)
                    if not ports:
                        return None
                    in_port = self.canonicalize_port(
                        device=node,
                        observed=ports[1],
                        expand_ge_name=spec.expand_ge_name,
                    )
                    toks.append(f"{node}({in_port})")
                else:
                    toks.append(node)
                continue
            in_ports = self.get_ports(path[i - 1], node)
            out_ports = self.get_ports(node, path[i + 1])
            if not in_ports or not out_ports:
                return None
            in_port = self.canonicalize_port(
                device=node,
                observed=in_ports[1],
                expand_ge_name=spec.expand_ge_name,
            )
            out_port = self.canonicalize_port(
                device=node,
                observed=out_ports[0],
                expand_ge_name=spec.expand_ge_name,
            )
            toks.append(f"{node}({in_port})")
            toks.append(f"{node}({out_port})")
        return "->".join(toks)

    def render_paths(
        self,
        *,
        paths: list[list[str]],
        spec: PathQuestionSpec,
    ) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for p in paths:
            line = self.render_path(p, spec)
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)
        return "\n".join(lines)
