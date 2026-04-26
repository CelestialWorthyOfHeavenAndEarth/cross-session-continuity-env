"""
client.py — Step 3: Create Client

EnvClient subclass for the Cross-Session Continuity environment.
Connects over HTTP/WebSocket to the OpenEnv FastAPI server.

Usage:
    from client import ContinuityEnvClient, ContinuityAction

    # Connect to local server
    with ContinuityEnvClient(base_url="http://localhost:7860") as client:
        obs = client.reset(difficulty="easy", seed=42)
        result = client.step(ContinuityAction(tool="run_tests"))
        print(result.observation.output)

    # Connect to HF Space
    with ContinuityEnvClient(
        base_url="https://aswini-kumar-cross-session-continuity-env.hf.space"
    ) as client:
        obs = client.reset(difficulty="medium")
"""

from typing import Dict

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False
    EnvClient = object  # type: ignore[misc,assignment]
    StepResult = None
    State = None

try:
    from models import ContinuityAction, ContinuityObservation
except ImportError:
    from models import ContinuityAction, ContinuityObservation  # type: ignore


class ContinuityEnvClient(EnvClient):  # type: ignore[misc]
    """
    Client for the Cross-Session Continuity RL Environment.

    Wraps the OpenEnv HTTP server with typed Action/Observation classes.

    Session flow:
        1. reset()           → Session 1 starts
        2. step(read_file)   → read starter code
        3. step(write_file)  → write partial implementation
        4. step(run_tests)   → check progress
        5. step(write_handoff) → end Session 1
        6. step(parse_handoff) → Session 2 cold start
        7. step(write_file)  → complete implementation
        8. step(submit)      → scored; done=True
    """

    def _step_payload(self, action: ContinuityAction) -> Dict:
        return {
            "tool":    action.tool,
            "path":    action.path,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> "StepResult":
        obs_data = payload.get("observation", {})
        observation = ContinuityObservation(
            output=obs_data.get("output", ""),
            session=obs_data.get("session", 1),
            passed=obs_data.get("passed", 0),
            total=obs_data.get("total", 0),
            auxiliary_reward=obs_data.get("auxiliary_reward", 0.0),
            error=obs_data.get("error", ""),
            warning=obs_data.get("warning", ""),
            message=obs_data.get("message", ""),
            retries_left=obs_data.get("retries_left", 3),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        from openenv.core.client_types import StepResult as SR
        return SR(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> "State":
        from openenv.core.env_server.types import State as S
        return S(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
