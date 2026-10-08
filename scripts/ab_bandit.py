"""A/B：自适应算子选择（ucb）vs 随机抽样（uniform）。

同一个 mock 目标集、同一个种子、同样的预算，只换算子采样策略，
比较「攻下同样多目标需要多少次查询」。

mock 目标被有意设计成算子效果分层的（只有学术化 / 分段拆解 / 虚构框架 /
反面教材 几条路能推到 5 分），所以能学到偏好的策略应该明显更省。
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aj.service import BatchParams, CancelToken, iter_batch

DATASET = str(Path(__file__).resolve().parents[1] / "data" / "dataset_zh_crime_violence20.csv")


def run(strategy: str, budget: int, rounds: int = 1, limit: int = 20):
    params = BatchParams(
        use_mock=True, dataset_path=DATASET, limit=limit,
        budget=budget, max_iters=budget, workers=4,
        bandit_strategy=strategy, bandit_scope="global",
        mode="single-pass", resume=False, tag=f"ab-{strategy}",
        seed=20260808,
    )
    records = []
    for _ in range(rounds):
        for ev in iter_batch(params, CancelToken()):
            if ev.get("type") == "goal_done":
                records.append(ev["record"])
    success = [r for r in records if r["success"]]
    return {
        "strategy": strategy,
        "n": len(records),
        "success": len(success),
        "asr": round(100 * len(success) / max(1, len(records)), 1),
        "aqc_all": round(statistics.mean([r["queries"] for r in records]), 2),
        "aqc_success": round(statistics.mean([r["queries"] for r in success]), 2) if success else 0,
        "total_queries": sum(r["queries"] for r in records),
    }


print("预算 budget=6，单遍，20 条目标")
print(f"{'策略':<10}{'ASR%':>8}{'成功数':>8}{'AQC(成功)':>12}{'AQC(全部)':>12}{'总查询':>8}")
for strat in ("ucb", "uniform"):
    r = run(strat, budget=6)
    print(f"{r['strategy']:<10}{r['asr']:>8}{r['success']:>8}"
          f"{r['aqc_success']:>12}{r['aqc_all']:>12}{r['total_queries']:>8}")

print()
print("预算 budget=3，单遍，20 条目标（更紧的预算，更看得出调度差异）")
print(f"{'策略':<10}{'ASR%':>8}{'成功数':>8}{'AQC(成功)':>12}{'AQC(全部)':>12}{'总查询':>8}")
for strat in ("ucb", "uniform"):
    r = run(strat, budget=3)
    print(f"{r['strategy']:<10}{r['asr']:>8}{r['success']:>8}"
          f"{r['aqc_success']:>12}{r['aqc_all']:>12}{r['total_queries']:>8}")
