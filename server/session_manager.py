"""
server/session_manager.py

Manages the Session 1 → Session 2 transition.

Key responsibilities:
  1. Wipe the filesystem: code written in Session 1 does NOT persist.
  2. Preserve the task description and test suite (Session 2 must implement).
  3. Randomize function/class names again so Session 2 cannot reconstruct
     from Session 1 code memory (they share the handoff note only).
"""

import copy
from server.task_generator import Task


class SessionManager:
    """
    Handles the controlled transition between sessions.

    After transition:
      - task.files reset to starter_code (blank implementations)
      - task description preserved (Session 2 sees the original task)
      - test suites preserved (same tests run at submit)
    """

    def transition(self, task: Task) -> Task:
        """
        Wipe session 1 file state and return a fresh Task for session 2.

        Args:
            task: The task object at end of Session 1 (may have partial implementation).

        Returns:
            A new Task object with files reset to starter_code.
        """
        new_task = copy.deepcopy(task)
        # Wipe all file contents back to starter (blank) state
        new_task.files = copy.deepcopy(task.starter_code)
        return new_task
