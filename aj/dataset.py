"""数据集加载：csv / jsonl 都支持，自动识别文本列与领域列。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TEXT_COLUMN_CANDIDATES = [
    "goal", "query", "question_zh", "text", "prompt", "instruction",
    "original_instruction", "seed_text", "content",
]
PRIMARY_DOMAIN_CANDIDATES = ["primary_domain", "一级领域", "source_domain", "domain"]
SECONDARY_DOMAIN_CANDIDATES = ["secondary_domain", "二级领域", "subcategory", "subdomain"]


def _first_nonempty(row: Dict[str, object], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _record_id(row: Dict[str, object], fallback: int) -> str:
    for candidate in ("id", "ID", "_id"):
        value = row.get(candidate)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return str(fallback)


class DatasetLoader:
    def __init__(self, dataset_path: Path,
                 candidate_columns: Optional[Sequence[str]] = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.candidate_columns = list(candidate_columns or TEXT_COLUMN_CANDIDATES)

    def load_records(self, limit: Optional[int] = None,
                     offset: int = 0) -> List[Dict[str, str]]:
        suffix = self.dataset_path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            records = self._load_jsonl()
        else:
            records = self._load_csv()
        if offset:
            records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return records

    def _load_csv(self) -> List[Dict[str, str]]:
        with self.dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{self.dataset_path} 没有表头")
            column = self._detect_column(reader.fieldnames)
            out: List[Dict[str, str]] = []
            for idx, row in enumerate(reader):
                rec = self._build(row, idx, column)
                if rec:
                    out.append(rec)
        return out

    def _load_jsonl(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        with self.dataset_path.open("r", encoding="utf-8-sig") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{self.dataset_path} 第 {idx + 1} 行不是 JSON 对象")
                rec = self._build(row, idx)
                if rec:
                    out.append(rec)
        return out

    def _detect_column(self, header: Sequence[str]) -> str:
        lowered = {name.strip().lower(): name for name in header}
        for candidate in self.candidate_columns:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        raise ValueError(
            f"在 {self.dataset_path} 里找不到可用的文本列，试过："
            f"{', '.join(self.candidate_columns)}"
        )

    def _build(self, row: Dict[str, object], fallback_idx: int,
               text_column: Optional[str] = None) -> Optional[Dict[str, str]]:
        if text_column is not None:
            raw = row.get(text_column)
            text = str(raw).strip() if raw is not None else ""
            source = text_column
        else:
            text, source = "", ""
            for candidate in self.candidate_columns:
                raw = row.get(candidate)
                if raw is None:
                    continue
                candidate_text = str(raw).strip()
                if candidate_text:
                    text, source = candidate_text, candidate
                    break
        if not text:
            return None
        return {
            "id": _record_id(row, fallback_idx),
            "goal": text,
            "primary_domain": _first_nonempty(row, PRIMARY_DOMAIN_CANDIDATES) or "unknown",
            "secondary_domain": _first_nonempty(row, SECONDARY_DOMAIN_CANDIDATES) or "unknown",
            "source_column": source or "unknown",
        }
