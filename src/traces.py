"""Turn an agent trace into the tester's trace conditions.

  C5  raw:        every step after the task message, long outputs cut, then trimmed to a budget
                  by keeping the start and the end
  C4  structured: one line per agent action: what it did, a short result, failed?, doubt words
  C3  summary:    a model's short summary of the trace, written once per trace and cached
  X   mismatched: the C5 text of a trace from a different task with a similar length

The system prompt (OpenHands' instructions) and the first user message (the issue, which every
condition already gets) are left out of all versions.
"""

import json
import re

OUTPUT_KEEP = 1500         # chars kept from each tool output in C5 (half from the start, half from the end)
RAW_BUDGET = 60_000        # chars of C5 text; about 15k tokens
RESULT_KEEP = 160          # chars of each tool output in C4
FAILED = re.compile(r"Traceback|\bError\b|\w+Error:|FAILED|\bfailed\b|exit code [1-9]|No such file|command not found")
DOUBT = re.compile(r"\b(not sure|unsure|maybe|might|perhaps|I think|seems|assume|unclear|probably|guess|wonder)\b", re.I)

SUMMARY_PROMPT = """Below is the log of a coding agent working on a GitHub issue. Summarize it for someone \
who will write tests for the agent's patch. Use these headings, a few short bullets each:
Intent (what the agent believed the issue asks for), Changes (what it changed and where), \
Assumptions, Edge cases it considered, Tests it ran and their results, Doubts or unverified points.
Report only what the log shows. Do not judge whether the patch is correct.

AGENT LOG:
{trace}"""


def text_of(content) -> str:
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def calls_of(msg: dict) -> list[tuple[str, dict]]:
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = (c or {}).get("function") or {}
        args = fn.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else (args or {})
        except json.JSONDecodeError:
            args = {"raw": args}
        calls.append((fn.get("name") or "?", args if isinstance(args, dict) else {"raw": args}))
    return calls


def steps(trajectory: list[dict]) -> list[dict]:
    """Messages after the system prompt and the task message."""
    msgs = [m for m in trajectory if isinstance(m, dict)]
    start = next((i + 1 for i, m in enumerate(msgs) if m.get("role") == "user"), 0)
    return msgs[start:]


def cut(text: str, keep: int) -> str:
    if len(text) <= keep:
        return text
    return f"{text[:keep // 2]}\n[... {len(text) - keep} chars cut ...]\n{text[-keep // 2:]}"


def render_raw(trajectory: list[dict], budget: int = RAW_BUDGET) -> str:
    blocks = []
    for m in steps(trajectory):
        role, body = m.get("role"), text_of(m.get("content")).strip()
        if role == "assistant":
            parts = [body] if body else []
            parts += [f"CALL {name}({json.dumps(args, ensure_ascii=False)})" for name, args in calls_of(m)]
            blocks.append("[agent]\n" + "\n".join(parts))
        elif role == "tool":
            blocks.append("[output]\n" + cut(body, OUTPUT_KEEP))
        else:
            blocks.append(f"[{role}]\n" + cut(body, OUTPUT_KEEP))
    text = "\n\n".join(blocks)
    if len(text) <= budget:
        return text
    head, tail = text[:budget // 4], text[-(budget - budget // 4):]
    return f"{head}\n\n[... {len(text) - budget} chars from the middle of the log cut ...]\n\n{tail}"


def action_label(name: str, args: dict) -> str:
    if name == "execute_bash":
        return "bash: " + str(args.get("command", "")).strip().replace("\n", " ")[:200]
    if name == "str_replace_editor":
        return f"{args.get('command', 'edit')} {args.get('path', '')}"
    if name == "think":
        return "think: " + str(args.get("thought", "")).strip().replace("\n", " ")[:300]
    return f"{name}: " + json.dumps(args, ensure_ascii=False)[:200]


def render_structured(trajectory: list[dict]) -> str:
    lines, waiting = [], []  # waiting: steps whose tool output hasn't come yet
    for m in steps(trajectory):
        role = m.get("role")
        if role == "tool":
            if waiting:
                out = text_of(m.get("content")).strip()
                step = waiting.pop(0)
                step["result"] = out.replace("\n", " ")[:RESULT_KEEP]
                step["failed"] = bool(FAILED.search(out))
            continue
        waiting = []
        if role != "assistant":
            continue
        said, calls = text_of(m.get("content")), calls_of(m)
        for name, args in calls:
            doubt = DOUBT.findall(said + " " + str(args.get("thought", "")))
            step = {"action": action_label(name, args), "result": "", "failed": False,
                    "doubt": sorted({d.lower() for d in doubt})}
            lines.append(step)
            waiting.append(step)
        if not calls and said.strip():
            lines.append({"action": "says: " + said.strip().replace("\n", " ")[:300], "result": "",
                          "failed": False, "doubt": sorted({d.lower() for d in DOUBT.findall(said)})})
    out = []
    for i, s in enumerate(lines, 1):
        flags = (" [FAILED]" if s["failed"] else "") + (f" [doubt: {', '.join(s['doubt'])}]" if s["doubt"] else "")
        out.append(f"{i}. {s['action']}{flags}" + (f"\n   -> {s['result']}" if s["result"] else ""))
    return "\n".join(out)


def mismatched(trajectory_id: str, instance_of: dict, lengths: dict) -> str:
    """The trajectory_id from a different task whose trace length is closest."""
    own_task, own_len = instance_of[trajectory_id], lengths[trajectory_id]
    others = [t for t in lengths if instance_of[t] != own_task]
    return min(others, key=lambda t: (abs(lengths[t] - own_len), t))
