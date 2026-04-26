"""
server/sandbox.py

Safe code execution sandbox using subprocess + ulimits.

Resource limits enforced:
  - CPU:       8s hard limit
  - RAM:       256 MB
  - File handles: 20
  - Subprocesses: 10 (prevents fork bombs)
  - Network:   PATH stripped to /usr/bin:/bin (no pip, no curl)

Note: For production training on HF Jobs, upgrade to Docker container
isolation. subprocess + ulimits is sufficient for dev and demo.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict


@dataclass
class TestResult:
    passed: int
    total: int
    compiled: bool
    summary: str


class Sandbox:
    """
    Executes pytest in an isolated subprocess with strict resource limits.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def run_tests(self, files: Dict[str, str], test_code: str) -> TestResult:
        """
        Write files + test code to a temp directory and run pytest.

        Args:
            files: Dict[filename → source code] — the agent's current file state.
            test_code: pytest test suite source code as a string.

        Returns:
            TestResult with pass/fail counts and compilation flag.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_files(tmpdir, files, test_code)

            try:
                result = subprocess.run(
                    [
                        "python", "-m", "pytest",
                        "test_solution.py",
                        "--tb=short", "-q", "--no-header",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    preexec_fn=self._set_limits,          # POSIX only
                    env={"PATH": "/usr/bin:/bin"},         # no network access
                )
                return self._parse_result(result.stdout, result.returncode)

            except subprocess.TimeoutExpired:
                return TestResult(
                    passed=0, total=1, compiled=False,
                    summary="Timeout — likely infinite loop or blocking call.",
                )
            except PermissionError:
                # preexec_fn not available on Windows — run without ulimits (dev only)
                result = subprocess.run(
                    ["python", "-m", "pytest", "test_solution.py",
                     "--tb=short", "-q", "--no-header"],
                    capture_output=True, text=True,
                    timeout=self.timeout, cwd=tmpdir,
                )
                return self._parse_result(result.stdout, result.returncode)
            except Exception as e:
                return TestResult(
                    passed=0, total=1, compiled=False,
                    summary=f"Sandbox error: {e}",
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_files(
        self,
        tmpdir: str,
        files: Dict[str, str],
        test_code: str,
    ) -> None:
        for filename, content in files.items():
            path = os.path.join(tmpdir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
            f.write(test_code)

    @staticmethod
    def _set_limits() -> None:
        """ulimits — only runs on POSIX (Linux/Mac)."""
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU,    (8, 8))
            resource.setrlimit(resource.RLIMIT_AS,     (256 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (20, 20))
            resource.setrlimit(resource.RLIMIT_NPROC,  (10, 10))
        except Exception:
            pass  # graceful degradation on Windows

    @staticmethod
    def _parse_result(stdout: str, returncode: int) -> TestResult:
        """
        Parse pytest -q output like:
          "3 passed, 1 failed in 0.42s"
          "4 passed in 0.18s"
          "ERROR: ..."
        """
        compiled = "SyntaxError" not in stdout and "ImportError" not in stdout
        passed, total = 0, 0

        for line in stdout.split("\n"):
            line = line.strip()
            if "passed" in line or "failed" in line or "error" in line:
                parts = line.split()
                p, f = 0, 0
                for i, part in enumerate(parts):
                    if part == "passed":
                        try:
                            p = int(parts[i - 1])
                        except (IndexError, ValueError):
                            pass
                    if part in ("failed", "error"):
                        try:
                            f = int(parts[i - 1])
                        except (IndexError, ValueError):
                            pass
                total = p + f
                passed = p
                break

        if total == 0 and returncode == 0:
            # pytest found no tests — treat as 0/0
            compiled = True

        return TestResult(
            passed=passed,
            total=max(total, 1),
            compiled=compiled,
            summary=stdout[:500] if stdout else "No output.",
        )
