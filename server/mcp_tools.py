"""
server/mcp_tools.py

OpenEnv MCP tool registry.

OpenEnv discovers tools by scanning for functions decorated with @tool
(or listed in openenv.yaml). This module registers all 6 environment
tools and wires them to the CrossSessionContinuityEnv instance.

Usage (from server/env.py or the MCP server entry point):
    from server.mcp_tools import build_tool_registry
    tools = build_tool_registry(env_instance)
"""

from __future__ import annotations
from typing import Any, Dict


def build_tool_registry(env) -> Dict[str, Any]:
    """
    Return a dict mapping tool_name -> callable, each wrapping env.step().

    Each callable matches the MCP tool signature:
        tool_fn(**kwargs) -> dict
    """

    def read_file(path: str) -> dict:
        """Read a file from the current session's workspace."""
        from server.env import Action
        return env.step(Action(tool="read_file", path=path))

    def write_file(path: str, content: str) -> dict:
        """Write content to a file in the current session's workspace."""
        from server.env import Action
        return env.step(Action(tool="write_file", path=path, content=content))

    def run_tests() -> dict:
        """Run the visible test suite against current file state."""
        from server.env import Action
        return env.step(Action(tool="run_tests"))

    def write_handoff(content: str) -> dict:
        """
        End Session 1 by writing a structured handoff note.

        Required sections: TASK / COMPLETED / REMAINING /
                           KEY FUNCTIONS / EDGE CASES / NEXT STEPS
        Max 400 tokens. Max 5 lines of code in code blocks.
        """
        from server.env import Action
        return env.step(Action(tool="write_handoff", content=content))

    def parse_handoff() -> dict:
        """Retrieve the handoff note at the start of Session 2."""
        from server.env import Action
        return env.step(Action(tool="parse_handoff"))

    def submit() -> dict:
        """Submit the solution. Runs visible + hidden tests. Returns reward."""
        from server.env import Action
        return env.step(Action(tool="submit"))

    return {
        "read_file":     read_file,
        "write_file":    write_file,
        "run_tests":     run_tests,
        "write_handoff": write_handoff,
        "parse_handoff": parse_handoff,
        "submit":        submit,
    }


# ---------------------------------------------------------------------------
# OpenEnv MCP server entry (called by openenv runtime)
# ---------------------------------------------------------------------------

def make_mcp_env(difficulty: str = "medium"):
    """
    Factory called by the OpenEnv runtime to create a fresh env instance
    with its tool registry. Returns (env, tools_dict).
    """
    from server.env import CrossSessionContinuityEnv
    env   = CrossSessionContinuityEnv(difficulty=difficulty)
    tools = build_tool_registry(env)
    return env, tools
