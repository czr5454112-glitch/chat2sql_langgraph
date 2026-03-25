from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

from .config import Settings
from .tracing import SQLTraceEvent


MERGE_SYSTEM = """你是一个 SQLite 3 的「SQL 整合」专家（后处理阶段，不再调用任何外部工具）。

你会收到：
1) 用户本轮的自然语言问题；
2) 智能体为回答该问题而**依次实际执行**过的多条 SQL（仅 SELECT/WITH 类查询）及其返回摘要；
3) 智能体基于这些查询写出的**最终自然语言答案**。

你的任务：产出**恰好一条**可在 SQLite 3 中成功执行的 SQL，使其查询结果能够**支撑**最终答案中的关键结论（数字、排序、分组、列表等）。这是业界常见的「SQL consolidation / replay query」做法：用单条可执行语句概括本轮分析逻辑。

硬性规则：
- 只输出数据查询：允许 WITH/SELECT；严禁 INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE 等写操作。
- 若多条原 SQL 语义上可合并：优先用 CTE（WITH a AS (...), b AS (...) ...）表达先后依赖关系。
- 若语义上难以无损合并为单行集（例如两个完全独立的问题），可用 UNION ALL 合并，并增加一列（如 step 或 part）标明每一段对应哪一步；列名与类型需自洽。
- 必须只使用原 SQL 中已出现过的表名与列名；不要臆造字段。
- 若确实无法用单条 SQL 忠实地表达最终答案所依赖的全部信息，仍给出**最接近**的单条查询，并在 SQL 开头用一行注释说明取舍，格式：-- MERGE_NOTE: ...
- 输出格式：只输出一个 Markdown 的 ```sql 代码块```，代码块内为纯 SQL，不要其它解释文字。"""


@dataclass
class MergeSqlResult:
    merged_sql: Optional[str]
    notes: str
    exec_ok: bool
    exec_error: Optional[str]
    result_preview: Optional[str]


def make_sql_database(settings: Settings) -> SQLDatabase:
    """与 agent_graph 中一致的 SQLite 连接方式。"""
    db_path = str(Path(settings.sqlite_db_path).expanduser())
    engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(db_path))
    return SQLDatabase(engine)


def extract_sql_from_llm_text(text: str) -> Optional[str]:
    """从模型输出中提取 SQL：优先 ```sql```，否则识别以 WITH/SELECT 开头的整段。"""
    if not text:
        return None
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
        return s if s else None
    t = text.strip()
    if re.search(r"(?is)\A\s*(WITH|SELECT)\b", t):
        return t
    return None


def _truncate(s: Any, max_chars: int) -> str:
    t = str(s)
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "..."


def _evidence_block(
    events: List[SQLTraceEvent],
    result_cap: int,
) -> str:
    lines: List[str] = []
    for i, ev in enumerate(events, start=1):
        lines.append(f"--- 第 {i} 条实际执行的 SQL ---")
        lines.append(ev.sql.strip())
        lines.append(f"--- 第 {i} 条返回摘要（可能被截断） ---")
        lines.append(_truncate(ev.result, result_cap))
    return "\n".join(lines)


def _run_sql_safe(db: SQLDatabase, sql: str) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        out = db.run(sql)
        return True, None, _truncate(out, 8000)
    except Exception as e:
        return False, str(e), None


def merge_sql_round(
    llm: ChatOpenAI,
    db: SQLDatabase,
    events: List[SQLTraceEvent],
    user_question: str,
    final_answer: str,
    max_retries: int = 1,
    evidence_result_cap: int = 2500,
) -> MergeSqlResult:
    """
    将本轮多条已执行 SQL 合并为单条，并在同一 SQLDatabase 上执行校验。
    """
    if not events:
        return MergeSqlResult(
            merged_sql=None,
            notes="本轮没有 sql_db_query 执行记录，跳过合并。",
            exec_ok=False,
            exec_error=None,
            result_preview=None,
        )

    evidence = _evidence_block(events, evidence_result_cap)
    user_payload = (
        f"【用户问题】\n{user_question.strip()}\n\n"
        f"【最终自然语言答案】\n{final_answer.strip()}\n\n"
        f"【依次执行过的 SQL 与结果】\n{evidence}\n"
    )

    messages: List[Any] = [
        SystemMessage(content=MERGE_SYSTEM),
        HumanMessage(content=user_payload),
    ]
    resp = llm.invoke(messages)
    text = getattr(resp, "content", "") or ""
    sql = extract_sql_from_llm_text(str(text))
    if not sql:
        return MergeSqlResult(
            merged_sql=None,
            notes=f"模型未产出可解析的 SQL。原始输出已截断保存。\n{_truncate(text, 2000)}",
            exec_ok=False,
            exec_error=None,
            result_preview=None,
        )

    ok, err, preview = _run_sql_safe(db, sql)
    if ok:
        return MergeSqlResult(
            merged_sql=sql,
            notes="已生成合并 SQL 并通过执行校验。",
            exec_ok=True,
            exec_error=None,
            result_preview=preview,
        )

    attempts = 0
    last_err = err
    while attempts < max_retries:
        attempts += 1
        messages.append(AIMessage(content=str(text)))
        messages.append(
            HumanMessage(
                content=(
                    "上一条合并 SQL 在本机 SQLite 上执行失败。\n"
                    f"错误信息：{last_err}\n\n"
                    "请根据错误修正 SQL，仍只输出一个 ```sql 代码块```，不要其它文字。"
                )
            )
        )
        resp2 = llm.invoke(messages)
        text2 = getattr(resp2, "content", "") or ""
        sql2 = extract_sql_from_llm_text(str(text2))
        if not sql2:
            return MergeSqlResult(
                merged_sql=sql,
                notes="合并 SQL 执行失败；重试时模型未返回可解析 SQL。",
                exec_ok=False,
                exec_error=last_err,
                result_preview=None,
            )
        sql = sql2
        ok, err, preview = _run_sql_safe(db, sql)
        if ok:
            return MergeSqlResult(
                merged_sql=sql,
                notes=f"经 {attempts} 次执行反馈后通过校验。",
                exec_ok=True,
                exec_error=None,
                result_preview=preview,
            )
        last_err = err

    return MergeSqlResult(
        merged_sql=sql,
        notes="合并 SQL 在执行校验阶段仍失败（已达最大重试次数）。",
        exec_ok=False,
        exec_error=last_err,
        result_preview=None,
    )
