from typing import Dict

from .rule_executor import StateExecutor


_EXEC = StateExecutor()


def normalize_state(state: Dict):
    return _EXEC.normalize_state(state)


def check_goal(state, goal_state: Dict) -> bool:
    return _EXEC.check_goal(state, goal_state)


def execute_action(state, action: str):
    result = _EXEC.apply(state, action)
    return result.state, result.executable


def run_semi_loop(sample: Dict, predicted_action: str, max_steps: int = 1) -> Dict:
    state = normalize_state(sample["state_before"])
    goal_state = sample["goal_state"]
    success = False
    steps = 0

    if check_goal(state, goal_state):
        return {
            "recovered": True,
            "steps_to_recover": 0,
            "task_completed": True,
            "last_executable": True,
        }

    for _ in range(max_steps):
        steps += 1
        state, executable = execute_action(state, predicted_action)
        success = executable
        if check_goal(state, goal_state):
            return {
                "recovered": True,
                "steps_to_recover": steps,
                "task_completed": True,
                "last_executable": success,
            }

    return {
        "recovered": False,
        "steps_to_recover": None,
        "task_completed": False,
        "last_executable": success,
    }
