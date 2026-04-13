import json
import re
import difflib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


ACTION_RE = re.compile(r"^([A-Za-z]+)\((.*)\)$")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)

CANONICAL_ACTIONS = ["LookAround", "Navigate", "Pick", "Place", "Open", "Close", "Retry"]
ACTION_ALIASES = {
    "look": "LookAround",
    "scan": "LookAround",
    "lookaround": "LookAround",
    "move": "Navigate",
    "goto": "Navigate",
    "go": "Navigate",
    "travel": "Navigate",
    "pickup": "Pick",
    "grab": "Pick",
    "take": "Pick",
    "put": "Place",
    "putdown": "Place",
    "drop": "Place",
    "insert": "Place",
    "unlock": "Open",
    "shut": "Close",
    "repeat": "Retry",
    "retry": "Retry",
}


@dataclass
class ParseResult:
    valid_json: bool
    valid_action: bool
    recovery_action: Optional[str]
    failure_type: Optional[str]
    evidence: Optional[str]
    error: Optional[str]


def _strip_markdown_fence(text: str) -> str:
    raw = text.strip()

    match = JSON_BLOCK_RE.search(raw)
    if match:
        return match.group(1).strip()

    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()

    left = raw.find("{")
    right = raw.rfind("}")
    if left != -1 and right != -1 and left < right:
        candidate = raw[left : right + 1].strip()
        if candidate:
            return candidate

    return raw


def parse_action(action: str) -> Tuple[Optional[str], list]:
    match = ACTION_RE.match(action.strip())
    if not match:
        return None, []
    name = match.group(1)
    args_text = match.group(2).strip()
    if not args_text:
        return name, []
    args = []
    cur = []
    depth = 0
    for ch in args_text:
        if ch == "(":
            depth += 1
            cur.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
            continue
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return name, args


def _canonicalize_action_name(name: str) -> str:
    if not name:
        return name
    lowered = name.strip().lower()
    if lowered in ACTION_ALIASES:
        return ACTION_ALIASES[lowered]

    nearest = difflib.get_close_matches(name, CANONICAL_ACTIONS, n=1, cutoff=0.74)
    if nearest:
        return nearest[0]

    nearest_lower = difflib.get_close_matches(lowered, [a.lower() for a in CANONICAL_ACTIONS], n=1, cutoff=0.74)
    if nearest_lower:
        idx = [a.lower() for a in CANONICAL_ACTIONS].index(nearest_lower[0])
        return CANONICAL_ACTIONS[idx]

    return name


def _normalize_action_string(action: str, enable_repair: bool = True) -> str:
    name, args = parse_action(action)
    if name is None:
        return action
    canon = _canonicalize_action_name(name)
    norm_args = list(args)

    # Heuristic repair for nested Retry(...) outputs:
    # e.g., Retry(Pick(apple)) -> Retry(Pick)
    if enable_repair and canon == "Retry" and norm_args:
        inner = norm_args[0].strip()
        inner_name = _canonicalize_action_name(inner)
        if inner_name not in CANONICAL_ACTIONS:
            parsed_inner_name, _ = parse_action(inner)
            if parsed_inner_name:
                inner_name = _canonicalize_action_name(parsed_inner_name)
        if inner_name in CANONICAL_ACTIONS:
            norm_args = [inner_name]

    return f"{canon}({','.join(norm_args)})"


def validate_action(action: str, candidate_vocab: Dict[str, list], enable_repair: bool = True) -> Tuple[bool, Optional[str]]:
    normalized = _normalize_action_string(action, enable_repair=enable_repair)
    name, args = parse_action(normalized)
    if name is None:
        return False, "action format must be Func(arg1,arg2)"

    if name == "LookAround":
        return (len(args) == 0, "LookAround() takes no args" if len(args) != 0 else None)
    if name == "Navigate":
        ok = len(args) == 1 and args[0] in candidate_vocab.get("targets", [])
        return ok, None if ok else "Navigate(target) target out of vocab"
    if name == "Pick":
        ok = len(args) == 1 and args[0] in candidate_vocab.get("objects", [])
        return ok, None if ok else "Pick(obj) obj out of vocab"
    if name == "Place":
        ok = (
            len(args) == 2
            and args[0] in candidate_vocab.get("objects", [])
            and args[1] in candidate_vocab.get("locations", [])
        )
        return ok, None if ok else "Place(obj,loc) args out of vocab"
    if name == "Open":
        ok = len(args) == 1 and args[0] in candidate_vocab.get("containers", [])
        return ok, None if ok else "Open(container) out of vocab"
    if name == "Close":
        ok = len(args) == 1 and args[0] in candidate_vocab.get("containers", [])
        return ok, None if ok else "Close(container) out of vocab"
    if name == "Retry":
        ok = len(args) == 1 and args[0] in candidate_vocab.get("retryable_actions", [])
        return ok, None if ok else "Retry(action) out of vocab"
    return False, f"unknown action name: {name}"


def parse_prediction(text: str, candidate_vocab: Dict[str, list], enable_repair: bool = True) -> ParseResult:
    cleaned = _strip_markdown_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return ParseResult(False, False, None, None, None, f"invalid json: {exc}")

    if not isinstance(data, dict):
        return ParseResult(True, False, None, None, None, "json must be object")

    recovery_action = data.get("recovery_action")
    failure_type = data.get("failure_type")
    evidence = data.get("evidence")
    if not isinstance(recovery_action, str):
        return ParseResult(True, False, None, None, None, "recovery_action must be string")

    normalized_action = _normalize_action_string(recovery_action, enable_repair=enable_repair)
    valid_action, err = validate_action(normalized_action, candidate_vocab, enable_repair=enable_repair)
    if failure_type is not None and failure_type not in {"F1", "F2", "F3", "F4", "F5"}:
        return ParseResult(True, False, normalized_action, None, evidence, "invalid failure_type")

    return ParseResult(True, valid_action, normalized_action, failure_type, evidence, err)
