"""统一计划源抽象。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PlanItem:
    key: str
    title: str
    body: str
    source_path: str
    line_number: int
    labels: list[str]


class PlanSource(Protocol):
    """统一的计划输入源接口。"""

    def collect_items(self) -> list[PlanItem]:
        ...

    def has_item_key(self, item_key: str) -> bool:
        ...

    def get_item_by_key(self, item_key: str) -> PlanItem | None:
        ...
