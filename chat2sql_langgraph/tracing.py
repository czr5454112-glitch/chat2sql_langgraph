from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SQLTraceEvent:
    sql: str
    result: Any


@dataclass
class SQLTrace:
    """记录“实际执行过的 SQL”及其结果（路线 A）。"""

    events: list[SQLTraceEvent] = field(default_factory=list)

    def clear(self) -> None:
        self.events.clear()

    def record(self, sql: str, result: Any) -> None:
        self.events.append(SQLTraceEvent(sql=sql, result=result))

    @property
    def last_sql(self) -> Optional[str]:
        return self.events[-1].sql if self.events else None

    @property
    def last_result(self) -> Any:
        return self.events[-1].result if self.events else None

