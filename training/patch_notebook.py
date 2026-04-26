"""
Adds a ScriptedAgent class to the training notebook so the full
pipeline can be validated without waiting for a real LLM to learn
the action format.

Run: python training/patch_notebook.py
"""
import json, re, os

NB_PATH = os.path.join(os.path.dirname(__file__), "train_grpo.ipynb")

SCRIPTED_CELL = """\
# ── SCRIPTED AGENT (pipeline smoke-test) ─────────────────────────────────────
# When SCRIPTED_AGENT=True the LLM is bypassed; a rule-based agent
# plays out a valid Session-1 + Session-2 trajectory so we can verify
# rewards > 0 and all downstream cells (eval, plots) work correctly.

import sys
sys.path.insert(0, '.')
from server.env import Action

class ScriptedAgent:
    \"\"\"
    Deterministic agent that always:
      S1: read_file → write_file (partial impl) → run_tests → write_handoff
      S2: parse_handoff → read_file → write_file (full impl) → run_tests → submit
    Rewards will be real (not 0) because it writes actual code.
    \"\"\"
    HANDOFF = (
        "TASK: implement the function as described.\\n"
        "COMPLETED:\\n- read starter code\\n- wrote partial implementation\\n"
        "REMAINING:\\n- edge cases and final testing\\n"
        "KEY FUNCTIONS:\\n- see solution.py\\n"
        "EDGE CASES:\\n- empty input returns sensible default\\n"
        "NEXT STEPS:\\n1. read_file solution.py\\n2. complete edge cases\\n3. run_tests\\n4. submit\\n"
    )
    # Simple implementations that pass most visible tests
    IMPLEMENTATIONS = {
        "merge_intervals": (
            "def merge_intervals(intervals):\\n"
            "    if not intervals: return []\\n"
            "    intervals = sorted(intervals, key=lambda x: x[0])\\n"
            "    merged = [intervals[0]]\\n"
            "    for s, e in intervals[1:]:\\n"
            "        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)\\n"
            "        else: merged.append([s, e])\\n"
            "    return merged\\n"
        ),
        "Stack": (
            "class Stack:\\n"
            "    def __init__(self): self._data = []\\n"
            "    def push(self, v): self._data.append(v)\\n"
            "    def pop(self):\\n"
            "        if not self._data: raise IndexError('empty')\\n"
            "        return self._data.pop()\\n"
            "    def peek(self):\\n"
            "        if not self._data: raise IndexError('empty')\\n"
            "        return self._data[-1]\\n"
            "    def is_empty(self): return len(self._data) == 0\\n"
            "    def size(self): return len(self._data)\\n"
            "    def __repr__(self): return f'Stack({self._data})'\\n"
            "    def __iter__(self): return iter(reversed(self._data))\\n"
        ),
        "RateLimiter": (
            "import time, threading\\n"
            "class RateLimiter:\\n"
            "    def __init__(self, rate, capacity):\\n"
            "        self._rate = rate; self._cap = capacity\\n"
            "        self._tokens = float(capacity); self._last = time.monotonic()\\n"
            "        self._lock = threading.Lock()\\n"
            "    def _refill(self):\\n"
            "        now = time.monotonic()\\n"
            "        self._tokens = min(self._cap, self._tokens + (now - self._last)*self._rate)\\n"
            "        self._last = now\\n"
            "    def is_allowed(self, n=1):\\n"
            "        with self._lock:\\n"
            "            self._refill()\\n"
            "            if n > self._cap: return False\\n"
            "            if self._tokens >= n:\\n"
            "                self._tokens -= n; return True\\n"
            "            return False\\n"
            "    def burst_remaining(self): self._refill(); return int(self._tokens)\\n"
        ),
    }

    def __init__(self):
        self._step = 0
        self._session = 1
        self._impl = None

    def _pick_impl(self, task):
        desc = task.description
        for key, code in self.IMPLEMENTATIONS.items():
            if key.lower() in desc.lower():
                return code
        return "def solution(*args, **kwargs): pass\\n"

    def act(self, obs):
        self._step += 1
        session = obs.get("session", self._session)

        if session == 1:
            if self._step == 1:
                return Action(tool="read_file", path="solution.py")
            if self._step == 2:
                # We'll write a real impl — but we don't have env here,
                # so just write a generic stub and let write_handoff carry info
                return Action(tool="write_file", path="solution.py",
                              content="# partial — see handoff\\ndef placeholder(): pass\\n")
            if self._step == 3:
                return Action(tool="run_tests")
            # write_handoff on step 4+
            return Action(tool="write_handoff", content=self.HANDOFF)
        else:
            s2 = self._step - 4  # steps within session 2
            if s2 <= 0:
                return Action(tool="parse_handoff")
            if s2 == 1:
                return Action(tool="read_file", path="solution.py")
            if s2 == 2:
                # Write a full working implementation
                impl = list(self.IMPLEMENTATIONS.values())[0]
                return Action(tool="write_file", path="solution.py", content=impl)
            if s2 == 3:
                return Action(tool="run_tests")
            return Action(tool="submit")

    def reset(self):
        self._step = 0
        self._session = 1

print("ScriptedAgent defined — SCRIPTED_AGENT =", SCRIPTED_AGENT)
"""

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Build the new cell
new_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": [line + "\n" for line in SCRIPTED_CELL.strip().splitlines()],
    "outputs": [],
    "execution_count": None,
}

# Insert after the config cell (index 1) and the warning cell (index 2)
# i.e., after cell 2 (0-indexed)
INSERT_AFTER = 2   # after warning-suppression cell
nb["cells"].insert(INSERT_AFTER + 1, new_cell)

# Patch the training loop cell: replace "agent = Agent(...)" with conditional
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "FastLanguageModel.for_training(model)" in src and "agent = Agent(" in src:
        new_src = src.replace(
            "FastLanguageModel.for_training(model)\nagent = Agent(model=model, tokenizer=tokenizer, max_new_tokens=256 if FAST_MODE else 512)",
            "if not SCRIPTED_AGENT:\n    FastLanguageModel.for_training(model)\n"
            "agent = ScriptedAgent() if SCRIPTED_AGENT else Agent(model=model, tokenizer=tokenizer, max_new_tokens=256 if FAST_MODE else 512)",
        )
        cell["source"] = [line + "\n" for line in new_src.rstrip().splitlines()]
        print("Patched training cell")

    # Patch eval agent cell similarly
    if "FastLanguageModel.for_inference(model)" in src:
        new_src = src.replace(
            "FastLanguageModel.for_inference(model)",
            "if not SCRIPTED_AGENT:\n    FastLanguageModel.for_inference(model)",
        )
        cell["source"] = [line + "\n" for line in new_src.rstrip().splitlines()]
        print("Patched eval cell")

    # patch each ablation + training loop that calls agent.act to also reset scripted agent
    if "for ep_idx in range(EPISODES_EPOCH)" in src:
        new_src = src.replace(
            "for ep_idx in range(EPISODES_EPOCH):",
            "for ep_idx in range(EPISODES_EPOCH):\n        if SCRIPTED_AGENT: agent.reset()",
        )
        cell["source"] = [line + "\n" for line in new_src.rstrip().splitlines()]
        print("Patched episode loop")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook patched: {NB_PATH}")
print("Run: git add training/train_grpo.ipynb && git commit -m 'feat: scripted agent for pipeline smoke-test'")
