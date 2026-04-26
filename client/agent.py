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
Complete as much as possible within your step limit.
When approaching the limit, call write_handoff() with a structured note:

Required sections (all mandatory):
  TASK:          one sentence — what the overall task is
  COMPLETED:     bullet list — fully implemented + test-verified items
  REMAINING:     bullet list — what Session 2 must still implement
  KEY FUNCTIONS: function/class names, signatures, brief purpose
  EDGE CASES:    constraints or tricky logic discovered in Session 1
  NEXT STEPS:    ordered list — what Session 2 should do first

Constraints:
  - Max 400 tokens in handoff note.
  - Max 5 lines of code in code blocks.
  - All 6 sections must be present.
  - You have a retry budget of 3 for invalid actions — use it wisely.

Available tools: read_file, write_file, run_tests, write_handoff
"""

S2_SYSTEM_PROMPT = """\
You are in Session 2. You have NO memory of Session 1.
Your ONLY information about what was done is the handoff note.

Start by calling parse_handoff() to retrieve the note.
Then use the note to continue the task.
Do NOT rewrite everything from scratch — the note tells you what to build on.

Available tools: parse_handoff, read_file, write_file, run_tests, submit
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
        Parse LLM output into an Action.

        Expected format:
          TOOL: write_file
          PATH: solution.py
          CONTENT:
          <content lines>
        """
        if not response:
            return None

        tool_match = re.search(r"TOOL:\s*(\w+)", response, re.IGNORECASE)
        if not tool_match:
            return None

        tool = tool_match.group(1).strip().lower()
        path_match    = re.search(r"PATH:\s*(.+)", response, re.IGNORECASE)
        content_match = re.search(r"CONTENT:\s*\n(.*)", response, re.IGNORECASE | re.DOTALL)

        path    = path_match.group(1).strip()    if path_match    else ""
        content = content_match.group(1).strip() if content_match else response

        return Action(tool=tool, path=path, content=content)

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
