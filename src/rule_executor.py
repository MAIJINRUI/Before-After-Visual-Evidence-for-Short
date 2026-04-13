from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from .parser import parse_action


def _to_set(values):
    if values is None:
        return set()
    return set(values)


@dataclass
class ExecResult:
    state: Dict[str, Set[str]]
    executable: bool
    error: Optional[str] = None


class StateExecutor:
    """Lightweight symbolic executor for closed recovery action set A."""

    def normalize_state(self, state: Dict) -> Dict[str, Set[str]]:
        return {
            "visible": _to_set(state.get("visible")),
            "holding": _to_set(state.get("holding")),
            "near": _to_set(state.get("near")),
            "open": _to_set(state.get("open")),
            "in": _to_set(state.get("in")),
        }

    def check_goal(self, state: Dict[str, Set[str]], goal_state: Dict) -> bool:
        has_condition = False
        for pair in goal_state.get("in", []):
            has_condition = True
            if pair not in state["in"]:
                return False
        for obj in goal_state.get("holding", []):
            has_condition = True
            if obj not in state["holding"]:
                return False
        for target in goal_state.get("near", []):
            has_condition = True
            if target not in state["near"]:
                return False
        for container in goal_state.get("open", []):
            has_condition = True
            if container not in state["open"]:
                return False
        # Avoid vacuous truth: if no conditions exist
        # (e.g. LookAround/Retry recovery), report not recovered.
        # Action accuracy still captures correctness for these cases.
        return has_condition

    def apply(self, state: Dict[str, Set[str]], action: str) -> ExecResult:
        """Apply one action and return updated symbolic state + executability."""
        new_state = deepcopy(state)
        name, args = parse_action(action)
        if name is None:
            return ExecResult(new_state, False, "invalid action format")

        if name == "LookAround":
            return ExecResult(new_state, True)

        if name == "Navigate":
            if len(args) != 1:
                return ExecResult(new_state, False, "Navigate expects one arg")
            target = args[0]
            new_state["near"].add(target)
            return ExecResult(new_state, True)

        if name == "Pick":
            if len(args) != 1:
                return ExecResult(new_state, False, "Pick expects one arg")
            obj = args[0]
            if obj in new_state["visible"] and len(new_state["holding"]) == 0:
                new_state["holding"].add(obj)
                return ExecResult(new_state, True)
            return ExecResult(new_state, False, "Pick precondition failed")

        if name == "Place":
            if len(args) != 2:
                return ExecResult(new_state, False, "Place expects two args")
            obj, loc = args
            if obj in new_state["holding"] and loc in new_state["near"]:
                new_state["holding"].remove(obj)
                new_state["in"].add(f"{obj}:{loc}")
                return ExecResult(new_state, True)
            return ExecResult(new_state, False, "Place precondition failed")

        if name == "Open":
            if len(args) != 1:
                return ExecResult(new_state, False, "Open expects one arg")
            container = args[0]
            if container in new_state["near"]:
                new_state["open"].add(container)
                return ExecResult(new_state, True)
            return ExecResult(new_state, False, "Open precondition failed")

        if name == "Close":
            if len(args) != 1:
                return ExecResult(new_state, False, "Close expects one arg")
            container = args[0]
            if container in new_state["open"]:
                new_state["open"].remove(container)
                return ExecResult(new_state, True)
            return ExecResult(new_state, False, "Close precondition failed")

        if name == "Retry":
            if len(args) != 1:
                return ExecResult(new_state, False, "Retry expects one arg")
            return ExecResult(new_state, True)

        return ExecResult(new_state, False, f"unknown action: {name}")
