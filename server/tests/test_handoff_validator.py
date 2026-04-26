"""
server/tests/test_handoff_validator.py

Unit tests for HandoffValidator.
"""

import pytest
from server.handoff_validator import HandoffValidator


@pytest.fixture
def validator():
    return HandoffValidator()


def _valid_handoff(extra: str = "") -> str:
    return (
        "TASK: Implement a stack.\n"
        "COMPLETED:\n- push() done\n- pop() done\n"
        "REMAINING:\n- __repr__ method\n- iterable support\n"
        "KEY FUNCTIONS:\n- Stack.push(val)\n- Stack.pop() -> val\n"
        "EDGE CASES:\n- pop on empty raises IndexError\n"
        "NEXT STEPS:\n1. Add __repr__\n2. Add __iter__\n3. Run all tests\n"
        + extra
    )


class TestHandoffValidator:

    def test_valid_handoff(self, validator):
        result = validator.validate(_valid_handoff())
        assert result.valid is True

    def test_missing_section(self, validator):
        bad = _valid_handoff().replace("TASK:", "SUBJECT:")
        result = validator.validate(bad)
        assert result.valid is False
        assert "TASK:" in result.reason

    def test_all_missing_sections(self, validator):
        for section in validator.REQUIRED_SECTIONS:
            bad = _valid_handoff().replace(section, section.replace(":", ""))
            result = validator.validate(bad)
            assert result.valid is False

    def test_empty_handoff(self, validator):
        result = validator.validate("")
        assert result.valid is False

    def test_code_block_too_long(self, validator):
        code = "```python\n" + "x = 1\n" * 10 + "```\n"
        result = validator.validate(_valid_handoff(extra=code))
        assert result.valid is False
        assert "Code block" in result.reason

    def test_code_block_at_limit(self, validator):
        code = "```python\n" + "x = 1\n" * 5 + "```\n"  # exactly 5 — allowed
        result = validator.validate(_valid_handoff(extra=code))
        assert result.valid is True

    def test_token_limit_exceeded(self, validator):
        long_extra = (" wordX" * 450)   # 450 extra tokens, well over 400 limit
        result = validator.validate(_valid_handoff(extra=long_extra))
        assert result.valid is False
        assert "too long" in result.reason

    def test_token_count_at_limit(self, validator):
        # Valid handoff is ~80 tokens — well within 400
        result = validator.validate(_valid_handoff())
        assert result.valid is True
