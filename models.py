"""
models.py — Step 1: Define Types

Action, Observation dataclasses for the Cross-Session Continuity environment.
These extend openenv.core types so the framework can serialize/deserialize them.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ContinuityAction(Action):
    """
    Action for the Cross-Session Continuity environment.

    The agent specifies which tool to call and its arguments.
    """
    tool: str = Field(..., description="Tool name: read_file | write_file | run_tests | write_handoff | parse_handoff | submit")
    path: str = Field(default="", description="File path (for read_file / write_file)")
    content: str = Field(default="", description="File content (for write_file) or handoff note (for write_handoff)")


class ContinuityObservation(Observation):
    """
    Observation returned after each action.

    Provides the agent with rich feedback about the current episode state,
    session, test results, and any errors or warnings.
    """
    output: str = Field(
        default="",
        description="Primary text output of the tool call",
    )
    session: int = Field(
        default=1,
        description="Current session number (1 or 2)",
    )
    passed: int = Field(
        default=0,
        description="Number of tests passed (run_tests only)",
    )
    total: int = Field(
        default=0,
        description="Total number of tests (run_tests only)",
    )
    auxiliary_reward: float = Field(
        default=0.0,
        description="Shaped reward for this step (training signal only)",
    )
    error: str = Field(
        default="",
        description="Error message if action was invalid or rejected",
    )
    warning: str = Field(
        default="",
        description="Warning message (e.g. approaching step limit)",
    )
    message: str = Field(
        default="",
        description="Informational message (e.g. session transition)",
    )
    retries_left: int = Field(
        default=3,
        description="Remaining retry budget for invalid actions",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended",
    )
    reward: float = Field(
        default=0.0,
        description="Final reward (only set when done=True)",
    )
