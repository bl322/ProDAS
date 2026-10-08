"""多轮复攻的种子与失败集。

单遍跑完，一批目标其实是「裁判已经给了 4 分、只差一步」的状态。复攻轮直接
从这些高分 prompt 出发，比从零重新搜一遍省得多。

- ``load_seed_map``：从历史 jsonl 里按目标 id 捞出高分 prompt
- ``write_failed_set``：把没攻下的目标单独存成一份 csv，供下一轮直接选它当数据集
- ``best_of_records``：多轮结果按 id 取 best-of
"""
from __future__ import annotations

import csv
import glob as _glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MIN_JUDGE_FOR_SEED = 3
MAX_SEEDS_PER_GOAL = 4
TOP_ATTEMPTS_PER_FILE = 3


def expand_paths(seed_from: Optional[str]) -> List[Path]:
    """支持逗号分隔的多个路径与 glob。"""
    if not seed_from:
        return []
    paths: List[Path] = []
    for part in str(seed_from).split(","):
        part = part.strip()
        if not part:
            continue
        matched = _glob.glob(part)
        paths.extend(Path(x) for x in (matched or [part]))
    return [p for p in paths if p.exists()]


def load_seed_map(seed_from: Optional[str],
                  max_per_goal: int = MAX_SEEDS_PER_GOAL) -> Dict[str, List[str]]:
    """{目标 id: [高分 prompt, ...]}，按裁判分降序取前几条。"""
    seed_map: Dict[str, List[str]] = {}
    for f in expand_paths(seed_from):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = record.get("id")
            if not sid:
                continue
            candidates = []
            for attempt in (record.get("attempts") or []):
                prompt = (attempt.get("prompt") or "").strip()
                if not prompt:
                    continue
                try:
                    score = int(attempt.get("judge_score", 0))
                except (TypeError, ValueError):
                    score = 0
                if score >= MIN_JUDGE_FOR_SEED:
                    candidates.append((score, len(prompt), prompt))
            candidates.sort(reverse=True)
            for _, _, prompt in candidates[:TOP_ATTEMPTS_PER_FILE]:
                seed_map.setdefault(sid, []).append(prompt)
            last = (record.get("best_prompt") or record.get("last_prompt") or "").strip()
            if last:
                seed_map.setdefault(sid, []).append(last)

    out: Dict[str, List[str]] = {}
    for sid, prompts in seed_map.items():
        seen, uniq = set(), []
        for p in prompts:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        if uniq:
            out[sid] = uniq[:max_per_goal]
    return out


def read_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("{") is False:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("id"):
            out.append(record)
    return out


def best_of_records(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """多轮 jsonl 按 id 取最好的一次（成功优先，其次比分高）。"""
    def rank(rec: Dict[str, Any]) -> tuple:
        try:
            score = int(rec.get("best_score") or 0)
        except (TypeError, ValueError):
            score = 0
        return (int(bool(rec.get("success"))), score, -int(rec.get("queries") or 0))

    best: Dict[str, Dict[str, Any]] = {}
    for f in paths:
        for rec in read_records(f):
            sid = rec.get("id")
            current = best.get(sid)
            if current is None or rank(rec) > rank(current):
                best[sid] = rec
    return best


def write_failed_set(dataset_path: str, failed_ids: List[str],
                     out_path: Path) -> Optional[str]:
    src = Path(dataset_path)
    if not src.exists() or not failed_ids:
        return None
    failed = set(str(x) for x in failed_ids)
    with src.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader if str(r.get("id") or "") in failed]
    if not rows:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(out_path)
