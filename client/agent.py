"""
client/agent.py

Agent loop for the Cross-Session Continuity environment.

Key constraints (enforced):
  - NO imports from server/ — client talks via MCP protocol only in production.
    In local dev/training, the env is passed in directly.
  - Retry logic for invalid actions (retry_budget = 3).
  - Session-aware system prompts.
  - Graceful noop fallback on retry exhaustion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Action (mirrored from server — no import)
# ---------------------------------------------------------------------------

@dataclass
class Action:
    tool: str
    path: str = ""
    content: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

S1_SYSTEM_PROMPT = """\
You are working on a coding task in Session 1.
Your goal: implement as much as possible, then write a handoff note for Session 2.

Available tools: read_file, write_file, run_tests, write_handoff

Call tools in ANY format — just mention the tool name clearly:
  - "read_file solution.py"
  - "I'll write_file to solution.py: <code>"
  - {"tool": "run_tests"}
  - write_file("solution.py", content="...")
  - TOOL: write_handoff\nCONTENT: ...

When you're done with Session 1, call write_handoff with a note containing:
  TASK / COMPLETED / REMAINING / KEY FUNCTIONS / EDGE CASES / NEXT STEPS
Keep it under 400 tokens. No full code dumps (max 5 lines of code).
"""

S2_SYSTEM_PROMPT = """\
You are in Session 2. You have NO memory of Session 1.
Your ONLY information is the handoff note — call parse_handoff first.

Available tools: parse_handoff, read_file, write_file, run_tests, submit

Call tools in ANY format — just mention the tool name:
  - "parse_handoff"
  - "I'll run_tests now"
  - write_file("solution.py", content="...")
  - TOOL: submit

Do NOT rewrite everything from scratch. The note tells you what to build on.
When tests pass, call submit.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    LLM-backed agent for the Cross-Session Continuity environment.

    In training, model and tokenizer are injected (Unsloth Qwen2.5-Coder-7B).
    In eval/demo, model can be replaced with a rule-based stub.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        retry_budget: int = 3,
        max_new_tokens: int = 512,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.retry_budget   = retry_budget
        self.max_new_tokens = max_new_tokens
        self.context: List[Dict] = []

    def act(self, obs: Dict[str, Any]) -> Action:
        """
        Generate an action given the current observation.

        Retries up to retry_budget times if the model output cannot be parsed.
        Falls back to a noop action on exhaustion.
        """
        prompt = self._build_prompt(obs)

        for attempt in range(self.retry_budget):
            response = self._generate(prompt)
            action   = self._parse_action(response)

            if action is not None:
                self.context.append({"obs": obs, "action": action, "response": response})
                return action

            # Build a retry prompt with the failed response
            prompt = self._build_retry_prompt(prompt, response, attempt)

        # Graceful fallback — noop so the episode doesn't crash
        return Action(tool="noop", content="")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        system = S1_SYSTEM_PROMPT if obs.get("session", 1) == 1 else S2_SYSTEM_PROMPT
        obs_text = self._format_obs(obs)
        return f"{system}\n\nObservation:\n{obs_text}\n\nAction:"

    def _build_retry_prompt(self, prev_prompt: str, failed_response: str, attempt: int) -> str:
        return (
            f"{prev_prompt}\n\n"
            f"[Attempt {attempt + 1} failed. Output was not a valid tool call.]\n"
            f"Failed output: {failed_response[:200]}\n\n"
            "Please output a valid tool call in the format:\n"
            "TOOL: <tool_name>\nPATH: <path (if applicable)>\nCONTENT:\n<content>\n\n"
            "Action:"
        )

    def _generate(self, prompt: str) -> str:
        """
        Generate a response from the model.

        In training: uses self.model + self.tokenizer (Unsloth/HF).
        In stub mode (model=None): returns empty string for override.
        """
        if self.model is None or self.tokenizer is None:
            return ""   # stub — caller must override or inject model

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            max_length=None,          # suppress transformers max_length warning
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt from the output
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return decoded.strip()

    @staticmethod
    def _parse_action(response: str) -> Optional[Action]:
        """
        Flexible parser — accepts any format that mentions a tool name.

        Handles all of these:
          TOOL: write_file\\nPATH: solution.py\\nCONTENT: ...   (strict)
          write_file("solution.py", content="...")              (function-call)
          {"tool": "write_file", "path": "solution.py"}        (JSON)
          I'll write to solution.py: ...                       (natural language)
          ```write_file\\nsolution.py\\n...```                  (markdown)
          run_tests                                             (bare word)
        """
        if not response:
            return None

        TOOLS = [
            "write_handoff", "parse_handoff",  # check longer names first
            "read_file", "write_file", "run_tests", "submit",
        ]

        resp_lower = response.lower()

        # ── Find which tool is mentioned ────────────────────────────────────
        found_tool = None
        for t in TOOLS:
            if t.replace("_", " ") in resp_lower or t in resp_lower:
                found_tool = t
                break

        if found_tool is None:
            return None

        # ── Extract path (flexible) ──────────────────────────────────────────
        path = ""
        if found_tool in ("read_file", "write_file"):
            # Try explicit PATH: label first
            m = re.search(r"path[:\s]+([^\s\n\"']+\.py)", response, re.IGNORECASE)
            if m:
                path = m.group(1).strip()
            else:
                # Fall back: any .py filename mentioned
                m = re.search(r"([\w/_-]+\.py)", response)
                path = m.group(1) if m else "solution.py"

        # ── Extract content (flexible) ───────────────────────────────────────
        content = ""
        if found_tool in ("write_file", "write_handoff"):
            # Try CONTENT: label
            m = re.search(r"content[:\s]*\n(.*)", response, re.IGNORECASE | re.DOTALL)
            if m:
                content = m.group(1).strip()
            else:
                # Try markdown code block
                m = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
                if m:
                    content = m.group(1).strip()
                else:
                    # Use everything after the tool name as content
                    idx = resp_lower.find(found_tool) + len(found_tool)
                    content = response[idx:].strip(" :\n\t")

        return Action(tool=found_tool, path=path, content=content)

    @staticmethod
    def _format_obs(obs: Dict[str, Any]) -> str:
        parts = []
        for key, val in obs.items():
            if key in ("done", "reward"):
                continue
            if isinstance(val, dict):
                for k, v in val.items():
                    parts.append(f"[{key}/{k}]\n{v}")
            else:
                parts.append(f"[{key}]\n{val}")
        return "\n\n".join(parts)
