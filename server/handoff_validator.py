"""
server/handoff_validator.py

Validates structured handoff notes written by the agent at the end of Session 1.
Enforces a 6-section required format. Rejects (not penalises) invalid notes so
the agent can retry within its retry budget.
"""

from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""


class HandoffValidator:
    """
    Validates that the agent's handoff note:
      1. Contains all 6 required section headers.
      2. Does not embed large code blocks (prevents code-dump gaming).
      3. Does not exceed the token budget (prevents padding gaming).
    """

    REQUIRED_SECTIONS = [
        "TASK:",
        "COMPLETED:",
        "REMAINING:",
        "KEY FUNCTIONS:",
        "EDGE CASES:",
        "NEXT STEPS:",
    ]
    MAX_CODE_BLOCK_LINES = 5   # prevents "just paste the code" shortcut
    MAX_TOKENS = 400            # hard ceiling — forces compression

    def validate(self, content: str) -> ValidationResult:
        if not content or not content.strip():
            return ValidationResult(valid=False, reason="Handoff note is empty.")

        for section in self.REQUIRED_SECTIONS:
            if section not in content:
                return ValidationResult(
                    valid=False,
                    reason=f"Missing required section: '{section}'. "
                           f"All required: {self.REQUIRED_SECTIONS}",
                )

        code_lines = self._count_code_block_lines(content)
        if code_lines > self.MAX_CODE_BLOCK_LINES:
            return ValidationResult(
                valid=False,
                reason=(
                    f"Code block too long ({code_lines} lines, max {self.MAX_CODE_BLOCK_LINES}). "
                    "Use function signatures only, not full implementations."
                ),
            )

        token_count = len(content.split())
        if token_count > self.MAX_TOKENS:
            return ValidationResult(
                valid=False,
                reason=(
                    f"Handoff too long ({token_count} tokens, max {self.MAX_TOKENS}). "
                    "Be more concise."
                ),
            )

        return ValidationResult(valid=True)

    def _count_code_block_lines(self, content: str) -> int:
        in_block = False
        count = 0
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
            elif in_block:
                count += 1
        return count
