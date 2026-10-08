"""引擎装配：把「攻击侧 LLM + 目标模型 + 裁判模型 + 算子账本」绑成一个引擎。

三个模型角色可以指向同一个网关下的不同模型，也可以全部同一个：

    attacker_model  做 TDI 逆向推理与变异改写（温度高，要多样性）
    target_model    被攻击的目标（温度按被测系统的正常设置）
    judge_model     裁判打分（温度 0，要稳定）

链路上的角色就这三个，方向信号统一由裁判给出。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from .bandit import OperatorBandit
from .judge import Judge
from .llm import build_client
from .runner import AttackJudgeRunner
from .tdi import TDIInitializer


@dataclass
class EngineConfig:
    # ---- 连接 ----
    base_url: str = ""
    api_key: str = ""
    no_proxy: str = ""
    use_mock: bool = True

    # ---- 三个模型角色 ----
    target_model: str = "qwen3-next-80b-a3b-instruct"
    attacker_model: str = ""
    judge_model: str = ""
    judge_mode: str = "llm"          # llm | heuristic
    judge_confirm: bool = True       # 4 分以上是否做一次二分类复核

    # ---- 采样 ----
    max_tokens: int = 2048
    target_temperature: float = 0.7
    attacker_temperature: float = 0.9
    judge_temperature: float = 0.0
    timeout: float = 180.0
    retries: int = 3

    # ---- 搜索 ----
    budget: int = 24
    max_iters: int = 24
    beam_width: int = 3
    p_iter: int = 8
    probe_width: int = 1
    operators_per_round: int = 3
    operator_family: str = "all"     # all | wrap | induce
    hard_restart_at: int = 8
    seed: int = 42

    # ---- 算子选择 ----
    bandit_strategy: str = "ucb"     # ucb | epsilon | uniform
    bandit_c: float = 1.4
    bandit_epsilon: float = 0.15
    bandit_ema: float = 0.0
    bandit_scope: str = "global"     # global: 整批共享一份账本 | goal: 每条目标独立

    def attacker(self) -> str:
        return self.attacker_model or self.target_model

    def judge(self) -> str:
        return self.judge_model or self.target_model


class AttackJudgeEngine:
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.cfg = config or EngineConfig()
        self.target_client: Any = None
        self.attacker_client: Any = None
        self.judge_client: Any = None
        self.judge: Optional[Judge] = None
        self.bandit: Optional[OperatorBandit] = None
        self._rng = random.Random(self.cfg.seed)

    # ------------------------------------------------------------------ #
    def build(self) -> "AttackJudgeEngine":
        cfg = self.cfg
        common = dict(
            base_url=cfg.base_url, api_key=cfg.api_key, no_proxy=cfg.no_proxy,
            timeout=cfg.timeout, retries=cfg.retries, seed=cfg.seed,
        )
        self.target_client = build_client(
            cfg.use_mock, model=cfg.target_model, max_tokens=cfg.max_tokens,
            temperature=cfg.target_temperature, label="target", **common)
        self.attacker_client = build_client(
            cfg.use_mock, model=cfg.attacker(), max_tokens=1024,
            temperature=cfg.attacker_temperature, label="attacker", **common)
        judge_mode = cfg.judge_mode
        if judge_mode == "llm" and not cfg.use_mock and not cfg.judge():
            judge_mode = "heuristic"
        if judge_mode == "llm":
            self.judge_client = build_client(
                cfg.use_mock, model=cfg.judge(), max_tokens=4096,
                temperature=cfg.judge_temperature, label="judge", **common)
            self.judge = Judge(mode="llm", judge_call=self.judge_client.chat,
                               confirm=cfg.judge_confirm)
        else:
            self.judge = Judge(mode="heuristic")
        self.bandit = OperatorBandit(
            strategy=cfg.bandit_strategy, c=cfg.bandit_c,
            epsilon=cfg.bandit_epsilon, ema=cfg.bandit_ema,
            rng=random.Random(cfg.seed + 7),
        )
        return self

    # ------------------------------------------------------------------ #
    def new_runner(
        self,
        goal: str,
        goal_id: str = "",
        seed_prompts: Optional[Sequence[str]] = None,
        strong_induce: bool = False,
        seed: Optional[int] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AttackJudgeRunner:
        cfg = self.cfg
        sd = cfg.seed if seed is None else seed
        tdi = TDIInitializer(call=self.attacker_client.chat, seed=sd,
                             strong_induce=strong_induce)
        bandit = self.bandit
        if cfg.bandit_scope == "goal" or bandit is None:
            bandit = OperatorBandit(
                strategy=cfg.bandit_strategy, c=cfg.bandit_c,
                epsilon=cfg.bandit_epsilon, ema=cfg.bandit_ema,
                rng=random.Random(sd + 7),
            )
        kwargs: Dict[str, Any] = dict(
            goal=goal, goal_id=goal_id, tdi=tdi,
            mutator=self.attacker_client.chat,
            target_call=self.target_client.chat,
            judge=self.judge, bandit=bandit,
            query_budget=cfg.budget, max_iters=cfg.max_iters,
            beam_width=cfg.beam_width, p_iter=cfg.p_iter,
            probe_width=cfg.probe_width,
            operators_per_round=cfg.operators_per_round,
            operator_family=cfg.operator_family,
            hard_restart_at=cfg.hard_restart_at,
            seed=sd, seed_prompts=list(seed_prompts or []),
            on_event=on_event, should_stop=should_stop,
        )
        if overrides:
            kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return AttackJudgeRunner(**kwargs)

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "config": {
                "target_model": self.cfg.target_model,
                "attacker_model": self.cfg.attacker(),
                "judge_model": self.cfg.judge(),
                "judge_mode": self.cfg.judge_mode,
                "use_mock": self.cfg.use_mock,
            },
            "clients": {},
        }
        for key in ("target_client", "attacker_client", "judge_client"):
            client = getattr(self, key, None)
            if client is not None and hasattr(client, "stats"):
                out["clients"][key] = client.stats()
        if self.judge is not None:
            out["judge"] = self.judge.stats()
        if self.bandit is not None:
            out["bandit"] = self.bandit.to_dict()
            out["bandit"]["ranking"] = self.bandit.ranking()
        return out
