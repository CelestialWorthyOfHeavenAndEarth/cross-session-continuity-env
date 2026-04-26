"""
server/task_generator.py

Generates coding tasks with:
  1. Name randomization per episode seed — prevents session 2 reconstructing from
     pretrained knowledge without reading the handoff note.
  2. Injected hidden adversarial tests — prevents visible-test overfitting.
  3. Handoff-critical calibration — tasks are designed so session 1 cannot
     fully finish within the step limit.
"""

import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Canonical → variant name bank (expanded per task template)
# ---------------------------------------------------------------------------

NAME_BANK: Dict[str, List[str]] = {
    # Data structures
    "merge_intervals":    ["combine_ranges", "fuse_spans", "join_segments"],
    "Stack":              ["Accumulator", "PushPop", "LifoStore"],
    "push":               ["enqueue_item", "add_entry", "store_val"],
    "pop":                ["dequeue_item", "remove_entry", "fetch_val"],
    # Rate limiting
    "RateLimiter":        ["ThrottleGuard", "RequestBucket", "AccessGate"],
    "is_allowed":         ["check_permit", "can_proceed", "gate_request"],
    # LRU Cache
    "LRUCache":           ["BoundedCache", "EvictStore", "MruVault"],
    "get":                ["fetch", "retrieve", "lookup"],
    "put":                ["store", "insert", "upsert"],
    # Data processing
    "process_data":       ["transform_records", "handle_payload", "digest_input"],
    "normalize":          ["standardize", "rescale", "calibrate"],
    # Graph
    "TopologicalSort":    ["DependencyOrder", "DAGResolver", "LayerSorter"],
    "add_edge":           ["link_nodes", "connect_dep", "wire_pair"],
    # Retry
    "RetryExecutor":      ["FaultTolerant", "BackoffRunner", "ResilienceWrap"],
    "execute":            ["run_with_retry", "attempt_call", "safe_invoke"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: str
    description: str
    starter_code: Dict[str, str]          # filename → source
    test_code: str                         # visible pytest suite
    hidden_test_code: str                  # adversarial hidden suite
    files: Dict[str, str] = field(default_factory=dict)  # runtime state


# ---------------------------------------------------------------------------
# Task templates (inline — no external files required for dev/demo)
# ---------------------------------------------------------------------------

TASK_TEMPLATES = {
    # -----------------------------------------------------------------------
    # EASY tasks
    # -----------------------------------------------------------------------
    "easy_merge_intervals": {
        "difficulty": "easy",
        "description": (
            "Implement merge_intervals(intervals: list[list[int]]) -> list[list[int]] "
            "in solution.py. The function receives a list of [start, end] intervals "
            "and must return a merged list with no overlaps. "
            "Session 1 should implement the sort + sweep logic and pass visible tests. "
            "Session 2 must handle edge cases (empty input, single interval, touching "
            "but non-overlapping intervals)."
        ),
        "starter_code": {
            "solution.py": (
                "def merge_intervals(intervals):\n"
                "    # TODO: implement\n"
                "    pass\n"
            )
        },
        "test_code": (
            "from solution import merge_intervals\n\n"
            "def test_basic():\n"
            "    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n\n"
            "def test_overlapping():\n"
            "    assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n\n"
            "def test_no_overlap():\n"
            "    assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]\n"
        ),
        "hidden_test_code": (
            "from solution import merge_intervals\n\n"
            "def test_empty():\n"
            "    assert merge_intervals([]) == []\n\n"
            "def test_single():\n"
            "    assert merge_intervals([[5,5]]) == [[5,5]]\n"
        ),
    },

    "easy_stack": {
        "difficulty": "easy",
        "description": (
            "Implement a Stack class in solution.py with push(val), pop() -> val, "
            "peek() -> val, is_empty() -> bool, and size() -> int. "
            "pop() and peek() on empty stack should raise IndexError. "
            "Session 1: implement and pass visible tests. "
            "Session 2: add __repr__ and make the class iterable."
        ),
        "starter_code": {
            "solution.py": (
                "class Stack:\n"
                "    def __init__(self):\n"
                "        # TODO\n"
                "        pass\n"
            )
        },
        "test_code": (
            "from solution import Stack\n\n"
            "def test_push_pop():\n"
            "    s = Stack()\n"
            "    s.push(1); s.push(2)\n"
            "    assert s.pop() == 2\n\n"
            "def test_empty_pop():\n"
            "    import pytest\n"
            "    s = Stack()\n"
            "    with pytest.raises(IndexError):\n"
            "        s.pop()\n\n"
            "def test_size():\n"
            "    s = Stack()\n"
            "    s.push(10); s.push(20)\n"
            "    assert s.size() == 2\n"
        ),
        "hidden_test_code": (
            "from solution import Stack\n\n"
            "def test_repr():\n"
            "    s = Stack()\n"
            "    s.push(1)\n"
            "    assert '1' in repr(s)\n\n"
            "def test_iterable():\n"
            "    s = Stack()\n"
            "    for v in [3,2,1]:\n"
            "        s.push(v)\n"
            "    assert list(s) == [1, 2, 3]\n"
        ),
    },

    "easy_running_median": {
        "difficulty": "easy",
        "description": (
            "Implement RunningMedian in solution.py. It must support add(num) and "
            "get_median() -> float. Uses two heaps internally. "
            "Session 1: implement heap-based median, pass visible tests. "
            "Session 2: add reset() and from_list(nums) classmethod."
        ),
        "starter_code": {
            "solution.py": (
                "class RunningMedian:\n"
                "    def __init__(self):\n"
                "        # TODO: two-heap approach\n"
                "        pass\n"
            )
        },
        "test_code": (
            "from solution import RunningMedian\n\n"
            "def test_basic():\n"
            "    rm = RunningMedian()\n"
            "    rm.add(1); rm.add(2); rm.add(3)\n"
            "    assert rm.get_median() == 2.0\n\n"
            "def test_even():\n"
            "    rm = RunningMedian()\n"
            "    rm.add(1); rm.add(2)\n"
            "    assert rm.get_median() == 1.5\n"
        ),
        "hidden_test_code": (
            "from solution import RunningMedian\n\n"
            "def test_reset():\n"
            "    rm = RunningMedian()\n"
            "    rm.add(5)\n"
            "    rm.reset()\n"
            "    rm.add(1)\n"
            "    assert rm.get_median() == 1.0\n\n"
            "def test_from_list():\n"
            "    rm = RunningMedian.from_list([3, 1, 2])\n"
            "    assert rm.get_median() == 2.0\n"
        ),
    },

    # -----------------------------------------------------------------------
    # MEDIUM tasks
    # -----------------------------------------------------------------------
    "medium_rate_limiter": {
        "difficulty": "medium",
        "description": (
            "Implement a token-bucket RateLimiter in solution.py. "
            "Constructor: RateLimiter(rate: int, capacity: int). "
            "is_allowed(n_tokens=1) -> bool: returns True if n_tokens can be consumed, "
            "refilling at 'rate' tokens per second. "
            "Use time.monotonic() for timestamps. "
            "Session 1: core token-bucket logic + visible tests. "
            "Session 2: add burst_remaining() -> int and thread-safety via threading.Lock."
        ),
        "starter_code": {
            "solution.py": (
                "import time\n\n"
                "class RateLimiter:\n"
                "    def __init__(self, rate: int, capacity: int):\n"
                "        # TODO\n"
                "        pass\n"
            )
        },
        "test_code": (
            "import time\n"
            "from solution import RateLimiter\n\n"
            "def test_basic_allow():\n"
            "    rl = RateLimiter(10, 10)\n"
            "    assert rl.is_allowed() is True\n\n"
            "def test_exhaustion():\n"
            "    rl = RateLimiter(1, 3)\n"
            "    assert rl.is_allowed(3) is True\n"
            "    assert rl.is_allowed() is False\n\n"
            "def test_refill():\n"
            "    rl = RateLimiter(10, 10)\n"
            "    rl.is_allowed(10)\n"
            "    time.sleep(0.2)\n"
            "    assert rl.is_allowed(2) is True\n\n"
            "def test_over_capacity():\n"
            "    rl = RateLimiter(5, 5)\n"
            "    assert rl.is_allowed(6) is False\n\n"
            "def test_zero_tokens():\n"
            "    rl = RateLimiter(5, 5)\n"
            "    assert rl.is_allowed(0) is True\n"
        ),
        "hidden_test_code": (
            "import threading, time\n"
            "from solution import RateLimiter\n\n"
            "def test_burst_remaining():\n"
            "    rl = RateLimiter(10, 10)\n"
            "    rl.is_allowed(4)\n"
            "    assert rl.burst_remaining() == 6\n\n"
            "def test_thread_safe():\n"
            "    rl = RateLimiter(100, 100)\n"
            "    results = []\n"
            "    def task():\n"
            "        results.append(rl.is_allowed(10))\n"
            "    threads = [threading.Thread(target=task) for _ in range(10)]\n"
            "    for t in threads: t.start()\n"
            "    for t in threads: t.join()\n"
            "    assert results.count(True) == 10\n"
        ),
    },

    "medium_lru_cache": {
        "difficulty": "medium",
        "description": (
            "Implement LRUCache(capacity: int) in solution.py. "
            "get(key) -> int: return value or -1 if not present. "
            "put(key, value): insert, evicting LRU entry if at capacity. "
            "Both O(1) using dict + doubly-linked list. "
            "Session 1: core get/put + visible tests. "
            "Session 2: add keys() -> list (in MRU→LRU order) and "
            "clear() method."
        ),
        "starter_code": {
            "solution.py": (
                "class LRUCache:\n"
                "    def __init__(self, capacity: int):\n"
                "        # TODO: doubly-linked list + dict\n"
                "        pass\n"
            )
        },
        "test_code": (
            "from solution import LRUCache\n\n"
            "def test_basic():\n"
            "    c = LRUCache(2)\n"
            "    c.put(1, 1); c.put(2, 2)\n"
            "    assert c.get(1) == 1\n"
            "    c.put(3, 3)\n"
            "    assert c.get(2) == -1\n\n"
            "def test_overwrite():\n"
            "    c = LRUCache(2)\n"
            "    c.put(1, 10); c.put(1, 20)\n"
            "    assert c.get(1) == 20\n\n"
            "def test_capacity_one():\n"
            "    c = LRUCache(1)\n"
            "    c.put(1, 1); c.put(2, 2)\n"
            "    assert c.get(1) == -1\n"
            "    assert c.get(2) == 2\n\n"
            "def test_miss():\n"
            "    c = LRUCache(3)\n"
            "    assert c.get(99) == -1\n\n"
            "def test_no_eviction_under_cap():\n"
            "    c = LRUCache(5)\n"
            "    for i in range(5):\n"
            "        c.put(i, i*10)\n"
            "    for i in range(5):\n"
            "        assert c.get(i) == i*10\n"
        ),
        "hidden_test_code": (
            "from solution import LRUCache\n\n"
            "def test_keys_order():\n"
            "    c = LRUCache(3)\n"
            "    c.put(1,1); c.put(2,2); c.put(3,3)\n"
            "    c.get(1)\n"
            "    assert c.keys()[0] == 1\n\n"
            "def test_clear():\n"
            "    c = LRUCache(3)\n"
            "    c.put(1,1); c.put(2,2)\n"
            "    c.clear()\n"
            "    assert c.get(1) == -1\n"
        ),
    },

    # -----------------------------------------------------------------------
    # HARD tasks
    # -----------------------------------------------------------------------
    "hard_topological_sort": {
        "difficulty": "hard",
        "description": (
            "Implement a TopologicalSort class in solution.py for a DAG. "
            "add_edge(u, v): add directed edge u→v. "
            "sort() -> list: return topological order (raise CycleError if cycle). "
            "has_path(src, dst) -> bool: BFS/DFS reachability. "
            "Also implement parallel_layers() -> list[list]: return nodes grouped "
            "by execution layer (Kahn's algorithm variant). "
            "Session 1: add_edge + sort + CycleError + visible tests. "
            "Session 2: has_path + parallel_layers + hidden tests."
        ),
        "starter_code": {
            "solution.py": (
                "from collections import defaultdict, deque\n\n"
                "class CycleError(Exception):\n"
                "    pass\n\n"
                "class TopologicalSort:\n"
                "    def __init__(self):\n"
                "        self.graph = defaultdict(list)\n"
                "        self.nodes = set()\n\n"
                "    def add_edge(self, u, v):\n"
                "        # TODO\n"
                "        pass\n\n"
                "    def sort(self):\n"
                "        # TODO: Kahn's algorithm\n"
                "        pass\n"
            )
        },
        "test_code": (
            "import pytest\n"
            "from solution import TopologicalSort, CycleError\n\n"
            "def test_linear():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','b'); ts.add_edge('b','c')\n"
            "    order = ts.sort()\n"
            "    assert order.index('a') < order.index('b') < order.index('c')\n\n"
            "def test_cycle():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','b'); ts.add_edge('b','a')\n"
            "    with pytest.raises(CycleError):\n"
            "        ts.sort()\n\n"
            "def test_diamond():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','b'); ts.add_edge('a','c')\n"
            "    ts.add_edge('b','d'); ts.add_edge('c','d')\n"
            "    order = ts.sort()\n"
            "    assert order[0] == 'a' and order[-1] == 'd'\n\n"
            "def test_isolated_node():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','b')\n"
            "    order = ts.sort()\n"
            "    assert set(order) == {'a','b'}\n\n"
            "def test_empty():\n"
            "    ts = TopologicalSort()\n"
            "    assert ts.sort() == []\n\n"
            "def test_single_node():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('x','x')\n"
            "    with pytest.raises(CycleError):\n"
            "        ts.sort()\n\n"
            "def test_large_dag():\n"
            "    ts = TopologicalSort()\n"
            "    for i in range(9):\n"
            "        ts.add_edge(str(i), str(i+1))\n"
            "    order = ts.sort()\n"
            "    assert order == [str(i) for i in range(10)]\n\n"
            "def test_multi_root():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','c'); ts.add_edge('b','c')\n"
            "    order = ts.sort()\n"
            "    assert order.index('a') < order.index('c')\n"
            "    assert order.index('b') < order.index('c')\n"
        ),
        "hidden_test_code": (
            "from solution import TopologicalSort\n\n"
            "def test_has_path_true():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','b'); ts.add_edge('b','c')\n"
            "    assert ts.has_path('a','c') is True\n\n"
            "def test_has_path_false():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','b')\n"
            "    assert ts.has_path('b','a') is False\n\n"
            "def test_parallel_layers():\n"
            "    ts = TopologicalSort()\n"
            "    ts.add_edge('a','c'); ts.add_edge('b','c')\n"
            "    layers = ts.parallel_layers()\n"
            "    assert set(layers[0]) == {'a','b'}\n"
            "    assert layers[-1] == ['c']\n"
        ),
    },
}

# Holdout tasks (simplified for eval only)
HOLDOUT_TEMPLATES = {
    "holdout_two_sum":     TASK_TEMPLATES["easy_merge_intervals"],   # placeholder
    "holdout_word_count":  TASK_TEMPLATES["easy_stack"],
    "holdout_retry_exec":  TASK_TEMPLATES["medium_rate_limiter"],
}


# ---------------------------------------------------------------------------
# TaskGenerator
# ---------------------------------------------------------------------------

class TaskGenerator:
    """
    Samples tasks from the template bank, applies name randomization per seed,
    and injects hidden adversarial tests.
    """

    def __init__(self, difficulty: str = "medium"):
        assert difficulty in {"easy", "medium", "hard"}, f"Invalid difficulty: {difficulty}"
        self.difficulty = difficulty
        self._bank = {
            k: v for k, v in TASK_TEMPLATES.items()
            if v["difficulty"] == difficulty
        }
        self._holdout = HOLDOUT_TEMPLATES

    def sample(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Task:
        if seed is not None:
            random.seed(seed)

        if task_id and task_id in TASK_TEMPLATES:
            template = copy.deepcopy(TASK_TEMPLATES[task_id])
            chosen_id = task_id
        else:
            chosen_id = random.choice(list(self._bank.keys()))
            template = copy.deepcopy(self._bank[chosen_id])

        task = self._build_task(chosen_id, template)
        task = self._randomize_names(task)
        return task

    def sample_holdout(self, task_id: Optional[str] = None) -> Task:
        """Sample from holdout set (never used in training)."""
        if task_id and task_id in self._holdout:
            template = copy.deepcopy(self._holdout[task_id])
            chosen_id = task_id
        else:
            chosen_id = random.choice(list(self._holdout.keys()))
            template = copy.deepcopy(self._holdout[chosen_id])
        return self._build_task(chosen_id, template)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_task(task_id: str, template: dict) -> Task:
        return Task(
            task_id=task_id,
            difficulty=template["difficulty"],
            description=template["description"],
            starter_code=dict(template["starter_code"]),
            test_code=template["test_code"],
            hidden_test_code=template["hidden_test_code"],
            files=dict(template["starter_code"]),   # runtime mutable copy
        )

    @staticmethod
    def _randomize_names(task: Task) -> Task:
        """
        Randomly remap canonical names to episode-specific variants.
        This prevents Session 2 from reconstructing the solution purely from
        pretrained knowledge without reading the handoff note.
        """
        mapping = {}
        for canonical, variants in NAME_BANK.items():
            mapping[canonical] = random.choice(variants)

        def apply(text: str) -> str:
            for canon, variant in mapping.items():
                text = text.replace(canon, variant)
            return text

        task.description = apply(task.description)
        task.files = {k: apply(v) for k, v in task.files.items()}
        task.starter_code = {k: apply(v) for k, v in task.starter_code.items()}
        task.test_code = apply(task.test_code)
        # Hidden tests use canonical names — not randomized (consistent eval)
        return task
