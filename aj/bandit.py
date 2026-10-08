"""自适应算子选择：把「哪个变异算子有用」当成一个多臂老虎机来学。

为什么需要它
------------
方向信号在本方法里只有裁判的 1–5 分一个。如果按固定规则分派算子，规则能
依据的也只能是这一个信号，等于把「哪些算子值得先试」写死成先验，遇到不同
分布的目标模型就不灵了。

这里的做法：记录**每个变异算子历史上带来的裁判分**，按 bandit 采样。
信号 100% 来自裁判的真实反馈，不引入任何额外模型，也不消耗额外查询——
每条候选本来就要查一次目标、判一次分，我们只是把结果记账下来。

收益怎么算
----------
单纯用子代绝对分做 reward 会把「底本本来就 4 分」的算子捧上天，也会把
「底本 1 分、算子救到 3 分」的算子埋没。所以 reward 取两半：

    reward = 0.5 * (子代分 / 5)            # 绝对质量
           + 0.5 * (0.5 + (子代分 - 底本分) / 6)   # 相对改进

底本分未知时（首轮 TDI 起点）用当前束内已知最好分做基准。

选择策略
--------
- ucb（默认）：mean + c·sqrt(ln(t+1)/n)，没试过的算子视为 +∞，
  保证冷启动阶段每个算子至少被采样一次；
- epsilon：以 ε 概率随机探索，否则取历史均值最高的；
- uniform：等概率随机，作为对照基线（用来衡量"学"到底有没有带来增益）。
"""
from __future__ import annotations

import math
import random
import threading
from typing import Any, Dict, List, Optional, Sequence

from .operators import MUTATION_OPERATORS, OPERATOR_NAMES


def compute_reward(child_score: int, base_score: Optional[float]) -> float:
    """把一次「底本 → 子代」的裁判分变化折算成 [0,1] 的收益。"""
    try:
        child = float(child_score)
    except (TypeError, ValueError):
        return 0.0
    if base_score is None:
        base = 1.0
    else:
        try:
            base = float(base_score)
        except (TypeError, ValueError):
            base = 1.0
    absolute = max(0.0, min(1.0, child / 5.0))
    relative = max(0.0, min(1.0, 0.5 + (child - base) / 6.0))
    return 0.5 * absolute + 0.5 * relative


class OperatorBandit:
    """算子收益账本。可以被整批评测共享（scope=global）或每条目标独立。"""

    def __init__(
        self,
        names: Optional[Sequence[str]] = None,
        strategy: str = "ucb",
        c: float = 1.4,
        epsilon: float = 0.15,
        prior_mean: float = 0.5,
        ema: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.names: List[str] = list(names or OPERATOR_NAMES)
        self.strategy = strategy if strategy in ("ucb", "epsilon", "uniform") else "ucb"
        self.c = float(c)
        self.epsilon = float(epsilon)
        self.prior_mean = float(prior_mean)
        # ema > 0：用指数滑动平均，让近期表现权重更高（目标/模型切换后能跟上）
        self.ema = float(ema)
        self.rng = rng or random.Random(20260808)

        self.n: Dict[str, int] = {name: 0 for name in self.names}
        self.mean: Dict[str, float] = {name: self.prior_mean for name in self.names}
        self.sum: Dict[str, float] = {name: 0.0 for name in self.names}
        self.last: Dict[str, Optional[float]] = {name: None for name in self.names}
        self.t = 0
        self.updates = 0
        # 批量并发时多条目标共享同一份账本，记账必须加锁
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    def update(self, name: str, reward: float) -> None:
        if name not in self.n:
            return
        try:
            r = float(reward)
        except (TypeError, ValueError):
            return
        r = max(0.0, min(1.0, r))
        with self._lock:
            self.n[name] += 1
            self.t += 1
            self.updates += 1
            self.sum[name] += r
            if self.ema > 0:
                self.mean[name] = (1.0 - self.ema) * self.mean[name] + self.ema * r
            else:
                self.mean[name] = self.sum[name] / self.n[name]
            self.last[name] = r

    # ------------------------------------------------------------------ #
    def ucb(self, name: str) -> float:
        if self.n[name] == 0:
            return float("inf")
        if self.strategy == "uniform":
            return self.mean[name]
        bonus = self.c * math.sqrt(math.log(self.t + 2.0) / self.n[name])
        return self.mean[name] + bonus

    def _order(self, pool: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """按当前策略给候选算子排序（高分在前）。"""
        if self.strategy == "uniform":
            shuffled = list(pool)
            self.rng.shuffle(shuffled)
            return shuffled
        if self.strategy == "epsilon" and self.rng.random() < self.epsilon:
            shuffled = list(pool)
            self.rng.shuffle(shuffled)
            return shuffled
        fresh = [o for o in pool if self.n[o["name"]] == 0]
        tried = [o for o in pool if self.n[o["name"]] > 0]
        if fresh:
            self.rng.shuffle(fresh)
        tried.sort(key=lambda o: self.ucb(o["name"]), reverse=True)
        return fresh + tried

    def select(self, k: int = 3, pool: Optional[List[Dict[str, str]]] = None,
               exclude: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
        """采样 k 个算子（不放回）。"""
        candidates = list(pool) if pool is not None else list(MUTATION_OPERATORS)
        if exclude:
            blocked = set(exclude)
            candidates = [o for o in candidates if o["name"] not in blocked]
        if not candidates:
            return []
        ordered = self._order(candidates)
        return ordered[: max(1, min(k, len(ordered)))]

    # ------------------------------------------------------------------ #
    def ranking(self) -> List[Dict[str, Any]]:
        rows = []
        for name in self.names:
            rows.append({
                "name": name,
                "family": next((o["family"] for o in MUTATION_OPERATORS
                                if o["name"] == name), ""),
                "n": self.n[name],
                "mean": round(self.mean[name], 4),
                "ucb": round(self.ucb(name), 4) if self.n[name] else None,
                "last": round(self.last[name], 4) if self.last[name] is not None else None,
            })
        rows.sort(key=lambda r: (-r["mean"], -r["n"]))
        return rows

    def top(self, k: int = 5) -> List[Dict[str, Any]]:
        return self.ranking()[:k]

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "c": self.c,
            "epsilon": self.epsilon,
            "ema": self.ema,
            "t": self.t,
            "updates": self.updates,
            "n": dict(self.n),
            "mean": {k: round(v, 4) for k, v in self.mean.items()},
            "sum": {k: round(v, 4) for k, v in self.sum.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], rng=None) -> "OperatorBandit":
        obj = cls(
            names=list((data.get("n") or {}).keys()) or None,
            strategy=data.get("strategy", "ucb"),
            c=data.get("c", 1.4),
            epsilon=data.get("epsilon", 0.15),
            ema=data.get("ema", 0.0),
            rng=rng,
        )
        for name, cnt in (data.get("n") or {}).items():
            if name in obj.n:
                obj.n[name] = int(cnt)
        for name, val in (data.get("sum") or {}).items():
            if name in obj.sum:
                obj.sum[name] = float(val)
        for name in obj.names:
            obj.mean[name] = (obj.sum[name] / obj.n[name]) if obj.n[name] else obj.prior_mean
        obj.t = int(data.get("t", sum(obj.n.values())))
        obj.updates = int(data.get("updates", obj.t))
        return obj
