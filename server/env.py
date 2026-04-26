"""
server/env.py

CrossSessionContinuityEnv — main MCPEnvironment subclass.

Implements the Gym-style API:
  env.reset()  → starts episode, returns Session 1 observation
  env.step()   → action → (obs, reward, done, info)
  env.state()  → current env state dict

Two-session architecture:
  Session 1: agent reads code, writes code, runs tests, writes handoff note.
  Session 2: agent calls parse_handoff(), reads, writes, runs tests, submits.

Key enforcement:
  - parse_handoff() MUST be called before any file access in Session 2.
  - HandoffValidator rejects malformed notes (not penalises — retry possible).
  - Filesystem wiped on session transition.
  - Dynamic step limits by difficulty.
  - Retry budget for invalid actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from server.task_generator import TaskGenerator
from server.session_manager import SessionManager
from server.sandbox import Sandbox
from server.rewards.rubric import ContinuityRubric
from server.rewards.auxiliary import AuxiliaryRewarder
from server.handoff_validator import HandoffValidator

try:
    from models import ContinuityAction, ContinuityObservation
except ImportError:
    try:
        from ..models import ContinuityAction, ContinuityObservation
    except ImportError:
        from models import ContinuityAction, ContinuityObservation

# ---------------------------------------------------------------------------
# OpenEnv base class — openenv-core package
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.interfaces import Environment as _EnvBase
    from openenv.core.env_server.types import State
    _HAS_OPENENV = True
except ImportError:
    class State:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
    class _EnvBase:  # type: ignore[no-redef]
        pass
    _HAS_OPENENV = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_LIMITS: Dict[str, int] = {
    "easy":   20,
    "medium": 35,
    "hard":   55,
}


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class Action:
    tool: str
    path: str = ""
    content: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CrossSessionContinuityEnv(_EnvBase):
    """
    RL environment for cross-session coding continuity.

    Inherits from openenv.core.env_server.interfaces.Environment.
    Implements OpenEnv Gym-style: reset / step / state (property) / close.
    Registered tools: read_file, write_file, run_tests,
                      write_handoff, parse_handoff, submit.
    """


    def __init__(self, difficulty: str = "medium"):
        assert difficulty in STEP_LIMITS, f"Invalid difficulty: {difficulty}"
        self.difficulty = difficulty
        self.task_gen    = TaskGenerator(difficulty)
        self.session_mgr = SessionManager()
        self.sandbox     = Sandbox(timeout=10)
        self.rubric      = ContinuityRubric()
        self.aux         = AuxiliaryRewarder()
        self.validator   = HandoffValidator()
        self.step_limit  = STEP_LIMITS[difficulty]

        # Episode state (populated in reset)
        self.task               = None
        self.session            = 1
        self.handoff            = None
        self.step_count         = 0
        self.invalid_action_count = 0
        self.retry_budget       = 3
        self.s1_test_history    = []
        self.s2_edit_history    = []
        self.handoff_parsed     = False
        self.s2_failed_runs     = 0

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a new episode. Returns Session 1 observation."""
        self.task               = self.task_gen.sample(task_id, seed=seed)
        self.session            = 1
        self.handoff            = None
        self.step_count         = 0
        self.invalid_action_count = 0
        self.retry_budget       = 3
        self.s1_test_history    = []
        self.s2_edit_history    = []
        self.handoff_parsed     = False
        self.s2_failed_runs     = 0

        return {
            "session":     1,
            "task":        self.task.description,
            "starter_code": self.task.starter_code,
            "message":     "Session 1 started. Complete what you can, then call write_handoff().",
            "step_limit":  self.step_limit,
        }

    def step(self, action: Action) -> Dict[str, Any]:
        """
        Execute one agent action. Returns observation dict with optional
        'done', 'reward', and 'auxiliary_reward' keys.
        """
        self.step_count += 1

        # Late-step warning in Session 1
        if self.session == 1 and self.step_count > self.step_limit:
            return {
                "warning":  "Step limit reached. Call write_handoff() now or episode terminates.",
                "penalty":  -0.1,
                "session":  1,
                "done":     False,
            }

        # Invalid action guard — soft: deduct small amount, don't terminate early
        if not self._is_valid_action(action):
            self.invalid_action_count += 1
            self.retry_budget -= 1
            if self.retry_budget <= 0:
                return {"done": True, "reward": 0.05, "error": "Retry budget exhausted. Partial credit awarded."}
            return {
                "error":        f"Invalid action '{action.tool}' in session {self.session}. "
                                f"Valid tools: {self._valid_tools()}",
                "retries_left": self.retry_budget,
                "auxiliary_reward": -0.01,   # tiny penalty, not episode-ending
                "done":         False,
            }

        # Dispatch by tool
        return self._dispatch(action)

    @property
    def state(self) -> State:
        """OpenEnv required: return current State object."""
        return State(
            session=self.session,
            step_count=self.step_count,
            step_limit=self.step_limit,
            handoff_written=self.handoff is not None,
            handoff_length=len(self.handoff.split()) if self.handoff else 0,
            difficulty=self.difficulty,
            invalid_actions=self.invalid_action_count,
            task_id=self.task.task_id if self.task else None,
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: Action) -> Dict[str, Any]:
        t = action.tool

        if t == "read_file":
            return self._handle_read_file(action)
        if t == "write_file":
            return self._handle_write_file(action)
        if t == "run_tests":
            return self._handle_run_tests(action)
        if t == "write_handoff":
            return self._handle_write_handoff(action)
        if t == "parse_handoff":
            return self._handle_parse_handoff(action)
        if t == "submit":
            return self._handle_submit(action)

        return {"error": f"Unknown tool: {t}", "done": False}

    def _handle_read_file(self, action: Action) -> Dict[str, Any]:
        if self.session == 2 and not self.handoff_parsed:
            return {"error": "Call parse_handoff() before accessing files in Session 2.", "done": False}
        content = self.task.files.get(action.path, f"File not found: {action.path}")
        return {"output": content, "session": self.session, "done": False}

    def _handle_write_file(self, action: Action) -> Dict[str, Any]:
        prev = self.task.files.get(action.path, "")
        self.task.files[action.path] = action.content
        if self.session == 2:
            self.s2_edit_history.append({
                "path": action.path,
                "prev": prev,
                "new":  action.content,
            })
        return {"output": f"Written to {action.path}", "session": self.session, "done": False}

    def _handle_run_tests(self, action: Action) -> Dict[str, Any]:  # noqa: ARG002
        result = self.sandbox.run_tests(self.task.files, self.task.test_code)
        if self.session == 1:
            self.s1_test_history.append(result.passed)
            aux = self.aux.s1_reward(result, self.task)
            return {
                "output":           result.summary,
                "passed":           result.passed,
                "total":            result.total,
                "auxiliary_reward": aux,
                "session":          1,
                "done":             False,
            }
        else:
            if result.passed == 0:
                self.s2_failed_runs += 1
            return {
                "output":  result.summary,
                "passed":  result.passed,
                "total":   result.total,
                "session": 2,
                "done":    False,
            }

    def _handle_write_handoff(self, action: Action) -> Dict[str, Any]:
        if self.session != 1:
            return {"error": "write_handoff only available in Session 1.", "done": False}

        validation = self.validator.validate(action.content)
        if not validation.valid:
            return {
                "error":    f"Handoff rejected: {validation.reason}",
                "required": self.validator.REQUIRED_SECTIONS,
                "done":     False,
            }

        self.handoff        = action.content
        self.session        = 2
        self.handoff_parsed = False
        self.task           = self.session_mgr.transition(self.task)
        self.retry_budget   = 3   # fresh budget for Session 2

        return {
            "session": 2,
            "message": "Session 2 started. Call parse_handoff() first.",
            "done":    False,
        }

    def _handle_parse_handoff(self, action: Action) -> Dict[str, Any]:  # noqa: ARG002
        if self.session != 2:
            return {"error": "parse_handoff only available in Session 2.", "done": False}
        self.handoff_parsed = True
        return {"output": self.handoff, "session": 2, "done": False}

    def _handle_submit(self, action: Action) -> Dict[str, Any]:  # noqa: ARG002
        if self.session != 2:
            return {"error": "submit only available in Session 2.", "done": False}

        visible = self.sandbox.run_tests(self.task.files, self.task.test_code)
        hidden  = self.sandbox.run_tests(self.task.files, self.task.hidden_test_code)

        reward_breakdown = self.rubric.score(
            visible_results=visible,
            hidden_results=hidden,
            handoff=self.handoff,
            s2_edit_history=self.s2_edit_history,
            s2_failed_runs=self.s2_failed_runs,
            invalid_actions=self.invalid_action_count,
        )

        return {
            "done":    True,
            "reward":  reward_breakdown.total,
            "breakdown": {
                "test_score":       reward_breakdown.test_score,
                "quality_score":    reward_breakdown.quality_score,
                "linearity_score":  reward_breakdown.linearity_score,
                "rewrite_penalty":  reward_breakdown.rewrite_penalty,
                "action_penalty":   reward_breakdown.action_penalty,
            },
            "visible_summary": visible.summary,
            "hidden_summary":  hidden.summary,
        }

    def close(self) -> None:
        """OpenEnv required: teardown. No-op for this environment."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_valid_action(self, action: Action) -> bool:
        s1_tools = {"read_file", "write_file", "run_tests", "write_handoff"}
        s2_tools = {"parse_handoff", "read_file", "write_file", "run_tests", "submit"}
        return action.tool in (s1_tools if self.session == 1 else s2_tools)

    def _valid_tools(self) -> str:
        if self.session == 1:
            return "read_file, write_file, run_tests, write_handoff"
        return "parse_handoff, read_file, write_file, run_tests, submit"
