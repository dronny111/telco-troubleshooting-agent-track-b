"""Fresh Qwen3.5-35B-A3B inference against Phase 2 Track A scenarios.

Reads `telco_data/Track A/data/Phase_2/test.json`, sends each scenario to the
configured Qwen endpoint with a comprehensive prompt (task + context + data +
options), parses the response for `\\boxed{...}` C-options, and writes
`work/track_a_fresh/result.csv` in the standard `scenario_id, answers` format.

Backend selection mirrors run_submission.py:
  * OPENAI_BASE_URL set → real Qwen (e.g. OpenRouter / vLLM)
  * unset → stub (no useful output; for smoke only)

Env knobs:
  TRACK_A_LIMIT          run only the first N scenarios (0 = all)
  TRACK_A_IDS            comma-sep scenario_ids
  TRACK_A_WORKERS        parallel HTTP workers (default 1)
  TRACK_A_MAX_TOKENS     max output tokens (default 4096 — enough for OpenRouter's
                         hidden reasoning + final answer)
  TRACK_A_OUT_DIR        output directory (default work/track_a_fresh)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_b.qwen_client import ChatResponse, QwenClient, QwenConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "telco_data" / "Track A" / "data" / "Phase_2" / "test.json"
DEFAULT_OUT_DIR = ROOT / "work" / "track_a_fresh"

DOTENV = ROOT / ".env"


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _resolve_env() -> dict[str, str]:
    env = dict(os.environ)
    dotenv = _load_dotenv(DOTENV)
    if not env.get("OPENAI_BASE_URL") and dotenv.get("AGENT_MODEL_URL"):
        env["OPENAI_BASE_URL"] = dotenv["AGENT_MODEL_URL"]
    if not env.get("OPENAI_API_KEY") and dotenv.get("AGENT_API_KEY"):
        env["OPENAI_API_KEY"] = dotenv["AGENT_API_KEY"]
    if not env.get("QWEN_MODEL") and dotenv.get("AGENT_MODEL_NAME"):
        env["QWEN_MODEL"] = dotenv["AGENT_MODEL_NAME"]
    return env


def _qwen_from_env(env: dict[str, str]) -> QwenClient:
    config = QwenConfig(
        base_url=env["OPENAI_BASE_URL"].rstrip("/"),
        api_key=env.get("OPENAI_API_KEY", "EMPTY"),
        model=env.get("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        temperature=float(env.get("QWEN_TEMPERATURE", "0.0")),
        max_tokens=int(env.get("TRACK_A_MAX_TOKENS", env.get("QWEN_MAX_TOKENS", "4096"))),
        timeout_s=float(env.get("QWEN_TIMEOUT_S", "90")),
        retries=int(env.get("QWEN_RETRIES", "1")),
    )
    return QwenClient(config)


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_COPT_RE = re.compile(r"\bC\d+\b")


def parse_answer(content: str | None, allowed_options: set[str]) -> set[str]:
    """Extract C-option set from a Qwen response.

    Strategy:
      1. Find all `\\boxed{...}` blocks; take the LAST one (the final answer).
      2. Pull C-options from that block; intersect with `allowed_options`.
      3. If no boxed block, fall back to the last 200 chars and pull any
         C-options found there.
    """
    if not content:
        return set()
    boxed = _BOXED_RE.findall(content)
    out: set[str] = set()
    if boxed:
        last = boxed[-1]
        for m in _COPT_RE.findall(last):
            tag = m.upper()
            if tag in allowed_options:
                out.add(tag)
    if not out:
        tail = content[-300:]
        for m in _COPT_RE.findall(tail):
            tag = m.upper()
            if tag in allowed_options:
                out.add(tag)
    return out


def _build_prompt(item: dict) -> tuple[str, str]:
    """Return (system, user) prompt strings for one Track A scenario."""
    task = item["task"]
    desc = task.get("description", "")
    options = task.get("options", [])
    allowed_ids = ", ".join(o["id"] for o in options)
    opt_block = "\n".join(f"  {o['id']}: {o.get('label', '')}" for o in options)

    ctx = item.get("context", {})
    if isinstance(ctx, dict):
        ctx_block = "\n".join(f"### {k}\n{v}" for k, v in ctx.items())
    else:
        ctx_block = str(ctx)

    data = item.get("data", {})
    if isinstance(data, dict):
        data_block = "\n".join(f"### {k}\n{v}" for k, v in data.items())
    else:
        data_block = str(data)

    system = (
        "You are an expert 5G/LTE radio access network engineer. Carefully analyze the "
        "drive-test data and network configuration provided, then select the most "
        "appropriate optimization actions from the listed options.\n\n"
        "Output format requirements:\n"
        "- Provide your reasoning, then conclude with a final answer.\n"
        "- The final answer MUST be enclosed in \\boxed{...} on its own line.\n"
        f"- The boxed answer must contain ONLY option IDs from this allowed set: {allowed_ids}.\n"
        "- For multi-answer tasks, list options separated by commas, e.g. \\boxed{C5, C12}.\n"
        "- For single-answer tasks, list exactly one option.\n"
        "- Do not output options that are not in the allowed set."
    )

    tag = item.get("tag", "")
    user_parts = [f"## Task ({tag})\n{desc}", f"## Options\n{opt_block}"]
    if ctx_block.strip():
        user_parts.append(f"## Context\n{ctx_block}")
    if data_block.strip():
        user_parts.append(f"## Data\n{data_block}")
    user_parts.append("Analyse the data, identify the dominant root cause and select the "
                      "optimization action(s) that directly address it. Conclude with the "
                      "final \\boxed{...} answer.")
    return system, "\n\n".join(user_parts)


def _parse_id_selector(raw: str) -> set[str]:
    return {p.strip() for p in raw.split(",") if p.strip()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path,
                   default=Path(os.environ.get("TRACK_A_OUT_DIR", str(DEFAULT_OUT_DIR))))
    p.add_argument("--limit", type=int,
                   default=int(os.environ.get("TRACK_A_LIMIT", "0")))
    p.add_argument("--ids", type=str, default=os.environ.get("TRACK_A_IDS", ""))
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("TRACK_A_WORKERS", "1")))
    p.add_argument("--resume", action="store_true",
                   help="skip scenario_ids already present in out result.csv")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_json) as f:
        scenarios = json.load(f)

    selected = _parse_id_selector(args.ids)
    if selected:
        scenarios = [s for s in scenarios if s["scenario_id"] in selected]
    if args.limit > 0:
        scenarios = scenarios[:args.limit]

    env = _resolve_env()
    if not env.get("OPENAI_BASE_URL"):
        print("ERROR: OPENAI_BASE_URL not set (no real Qwen backend). Aborting.")
        return 1

    qwen = _qwen_from_env(env)
    result_csv = args.out_dir / "result.csv"
    detail_jsonl = args.out_dir / "detail.jsonl"

    completed_ids: set[str] = set()
    if args.resume and result_csv.is_file():
        with open(result_csv) as f:
            for r in csv.DictReader(f):
                completed_ids.add(r.get("scenario_id", ""))

    print(f"==> Track A inference config")
    print(f"   input: {args.input_json}")
    print(f"   out:   {args.out_dir}")
    print(f"   n:     {len(scenarios)}  workers={args.workers}  resume={args.resume}")
    print(f"   base:  {env.get('OPENAI_BASE_URL')}  model={env.get('QWEN_MODEL')}")
    print(f"   resuming, skipping: {len(completed_ids)} already-done")

    import threading
    write_lock = threading.Lock()

    write_header = not (args.resume and result_csv.is_file())
    with open(result_csv, "a" if args.resume else "w", encoding="utf-8", newline="") as csv_f, \
            open(detail_jsonl, "a" if args.resume else "w", encoding="utf-8") as trace_f:
        w = csv.writer(csv_f)
        if write_header:
            w.writerow(["scenario_id", "answers"])
        t0 = time.perf_counter()

        def _process(item):
            sid = item["scenario_id"]
            if sid in completed_ids:
                return None
            allowed = {o["id"] for o in item["task"].get("options", [])}
            system, user = _build_prompt(item)
            try:
                resp: ChatResponse = qwen.chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    temperature=0.0,
                )
                content = resp.content or ""
                picks = parse_answer(content, allowed)
                answers = "|".join(sorted(picks, key=lambda c: int(c[1:])))
                return sid, answers, {
                    "scenario_id": sid,
                    "answers": answers,
                    "finish": resp.finish_reason,
                    "content_len": len(content),
                    "content_tail": content[-300:] if content else "",
                }
            except Exception as e:
                return sid, "", {"scenario_id": sid, "error": str(e)[:200]}

        ran = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process, it): it for it in scenarios}
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                sid, ans, detail = res
                with write_lock:
                    w.writerow([sid, ans])
                    csv_f.flush()
                    trace_f.write(json.dumps(detail) + "\n")
                    trace_f.flush()
                    ran += 1
                    print(f"[{ran:>4}/{len(scenarios)}] sid={sid[:8]} ans={ans!r}")

        dt = time.perf_counter() - t0
        print(f"\nwrote {result_csv}  rows added: {ran}  elapsed: {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
