import sys; sys.path.insert(0, '.')
from training.grpo_train import score_handoff, run_scripted_s2, build_dataset

good = """TASK: implement merge_intervals(intervals) -> list[list[int]]
COMPLETED:
- sorted intervals by start index
- basic merge loop written
REMAINING:
- touching intervals edge case
KEY FUNCTIONS:
- merge_intervals(intervals): returns merged list
EDGE CASES:
- empty list -> return []
NEXT STEPS:
1. read_file solution.py
2. fix merge condition to use <=
3. run_tests
4. submit"""

bad = "TASK: do stuff\nCOMPLETED: some\nREMAINING: more\nKEY FUNCTIONS: fns\nEDGE CASES: none\nNEXT STEPS: finish"

qg = score_handoff(good, 'easy_merge_intervals')
qb = score_handoff(bad,  'easy_merge_intervals')
s2g = run_scripted_s2('easy_merge_intervals', good, seed=0, epoch=0, total_epochs=3)
s2b = run_scripted_s2('easy_merge_intervals', bad,  seed=0, epoch=0, total_epochs=3)

print(f"Good: quality={qg:.3f}  s2={s2g:.3f}  total={0.4*qg+0.6*s2g:.3f}")
print(f"Bad:  quality={qb:.3f}  s2={s2b:.3f}  total={0.4*qb+0.6*s2b:.3f}")
print()

ds = build_dataset(12)
print(f"Dataset: {len(ds)} items")
print(f"First 3 task_ids: {[ds[i]['task_id'] for i in range(3)]}")
print()

if qg > qb:
    print("[OK] Reward correctly differentiates good vs bad handoffs")
else:
    print("[FAIL] Reward not differentiating")
