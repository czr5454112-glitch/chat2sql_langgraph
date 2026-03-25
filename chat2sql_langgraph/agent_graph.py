from __future__ import annotations

import sqlite3
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from typing_extensions import NotRequired, TypedDict

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import create_engine

from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph

from .config import Settings
from .sql_merge import merge_sql_round
from .tracing import SQLTrace


class TrackedQuerySQLDatabaseTool(QuerySQLDatabaseTool):
    """在每次执行 sql_db_query 时，记录真实 SQL 与返回结果。"""

    def __init__(self, *args: Any, trace: SQLTrace, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._trace = trace

    def _run(self, query: str, run_manager=None) -> str:  # type: ignore[override]
        result = super()._run(query, run_manager=run_manager)
        self._trace.record(query, result)
        return result


class TrackedSQLDatabaseToolkit(SQLDatabaseToolkit):
    def __init__(self, *args: Any, trace: SQLTrace, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._trace = trace

    def get_tools(self) -> list[BaseTool]:  # type: ignore[override]
        tools = super().get_tools()
        replaced: list[BaseTool] = []
        for tool in tools:
            if isinstance(tool, QuerySQLDatabaseTool):
                replaced.append(
                    TrackedQuerySQLDatabaseTool(
                        db=self.db,
                        trace=self._trace,
                        description=tool.description,
                        name=tool.name,
                    )
                )
            else:
                replaced.append(tool)
        return replaced


def _last_human_question(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human":
            c = getattr(m, "content", "")
            if c:
                return str(c)
    return ""


def _last_ai_answer(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) or getattr(m, "type", None) == "ai":
            c = getattr(m, "content", "")
            if c is not None and str(c).strip():
                return str(c)
    return ""


class AgentState(TypedDict):
    # 复用 LangGraph 的 messages 状态约定，便于 create_react_agent 接管。
    messages: List[BaseMessage]
    # 后处理：将本轮多条 sql_db_query 合并为单条（可选）
    merged_sql: NotRequired[Optional[str]]
    merge_notes: NotRequired[Optional[str]]
    merge_exec_ok: NotRequired[Optional[bool]]
    merge_exec_error: NotRequired[Optional[str]]
    merge_result_preview: NotRequired[Optional[str]]


def build_langgraph_sql_agent(settings: Settings) -> Tuple[Any, SQLTrace]:
    """
    返回：
    - compiled graph（可用 invoke）
    - SQLTrace（路线 A：抓取最终实际执行 SQL）
    """

    llm = ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
    )

    # Windows 中文/空格路径：直接用 sqlite3 连接更稳
    db_path = str(Path(settings.sqlite_db_path).expanduser())
    engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(db_path))
    db = SQLDatabase(engine)

    trace = SQLTrace()
    toolkit = TrackedSQLDatabaseToolkit(db=db, llm=llm, trace=trace)
    tools = toolkit.get_tools()

    # === LangChain SQL agent 核心逻辑（System Prompt）→ 适配 LangGraph ReAct ===
    # 说明：
    # - LangChain 的 SQL agent 通过 SQL_PREFIX + 工具描述来约束行为（限 top_k、禁止 DML、先 list_tables 再 schema、
    #   执行前必须 query_checker、禁止 SELECT *、只基于工具返回信息作答等）。
    # - 这里将核心约束重构为单一 SystemMessage，供 LangGraph 的 ReAct agent 使用。
    top_k = int(os.getenv("CHAT2SQL_TOP_K", "10").strip() or "10")
    dialect = getattr(toolkit, "dialect", "SQL")
    system_message = SystemMessage(
        content=(
            "你是一个用于与 SQL 数据库交互的智能体。\n"
            "给定用户问题，你需要：先用工具了解可查询的表与 schema，"
            f"再编写语法正确的 {dialect} SQL 去执行，最后根据查询结果回答。\n\n"
            "### 核心约束（必须遵守）\n"
            f"1) 除非用户明确要求更多，否则查询最多返回 {top_k} 行（使用 LIMIT）。\n"
            "2) 可以按相关列 ORDER BY，返回最有代表性的结果。\n"
            "3) 禁止 SELECT *：不要查询某张表的所有列，只查询与问题相关的列。\n"
            "4) 只能使用你被提供的工具；最终回答只能基于工具返回的信息，禁止臆测。\n"
            "5) 执行查询前必须先用 query checker 工具检查 SQL；若执行报错，改写 SQL 后重试。\n"
            "6) 严禁任何 DML/DDL 写操作：INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE 等。\n"
            "7) 如果问题与数据库明显无关，直接回答“我不知道”。\n\n"
            "### 推荐流程（更可靠）\n"
            "A) 先调用 sql_db_list_tables 确认可用表。\n"
            "B) 再对最相关的表调用 sql_db_schema（或等价的 info 工具）获取字段与样例。\n"
            "C) 基于 schema 生成 SQL；先调用 query checker；确认后再调用 sql_db_query。\n"
            "D) 若遇到 Unknown column/字段不确定，回到 B) 再查 schema，禁止猜字段。\n\n"
            "### Schema 摘要\n"
            "对话里如果出现以 [SCHEMA_SUMMARY] 开头的摘要，它来自工具的真实输出（表名/数量权威）。"
            "请优先参考该摘要选择表名。\n"
        )
    )

    def _find_tool(name: str) -> Optional[BaseTool]:
        for t in tools:
            if getattr(t, "name", None) == name:
                return t
        return None

    list_tables_tool = _find_tool("sql_db_list_tables")

    def _run_tool(tool: BaseTool, tool_input: Any) -> str:
        # 不同版本 tool 输入格式可能不同，这里做一次兼容尝试
        try:
            return str(tool.invoke(tool_input))
        except Exception:
            return str(tool.invoke({"tool_input": tool_input}))

    def _tables_from_list_output(raw: str) -> list[str]:
        # LangChain SQL 工具通常用 ", " 分隔；这里做一个稳健拆分
        parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
        return [p for p in parts if p]

    answer_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message,
        debug=settings.show_sql,
    )

    def _bootstrap_node(state: AgentState) -> AgentState:
        # 若对话里已经生成过摘要，则跳过（避免重复 list_tables）
        if any("[SCHEMA_SUMMARY]" in str(getattr(m, "content", "")) for m in state["messages"]):
            return state

        if list_tables_tool is None:
            # fallback：如果工具名变化，至少不阻塞 answer 阶段
            return state

        raw = _run_tool(list_tables_tool, "")
        tables = _tables_from_list_output(raw)
        summary = (
            "[SCHEMA_SUMMARY]\n"
            f"tables_count={len(tables)}\n"
            "tables=" + ", ".join(tables)
        )
        # 注入系统指令 + schema 摘要。系统指令只注入一次，避免对话膨胀。
        already_has_system = any(
            getattr(m, "type", None) == "system" or m.__class__.__name__ == "SystemMessage"
            for m in state["messages"]
        )
        base = state["messages"]
        if not already_has_system:
            base = base + [system_message]
        new_messages = base + [SystemMessage(content=summary)]
        return {"messages": new_messages}

    def _answer_node(state: AgentState) -> AgentState:
        res = answer_agent.invoke({"messages": state["messages"]})
        return {"messages": res.get("messages", state["messages"])}

    def _merge_sql_node(state: AgentState) -> AgentState:
        """ReAct 之后的后处理：SQL 轨迹合并 + 同库执行校验（不调用工具）。"""
        if not settings.enable_sql_merge:
            return {
                "merge_notes": "已关闭 SQL 合并（enable_sql_merge=False）。",
                "merged_sql": None,
                "merge_exec_ok": None,
                "merge_exec_error": None,
                "merge_result_preview": None,
            }
        msgs = state["messages"]
        user_q = _last_human_question(msgs)
        final_ans = _last_ai_answer(msgs)
        merge_res = merge_sql_round(
            llm=llm,
            db=db,
            events=list(trace.events),
            user_question=user_q,
            final_answer=final_ans,
            max_retries=max(0, int(settings.sql_merge_max_retries)),
        )
        return {
            "merged_sql": merge_res.merged_sql,
            "merge_notes": merge_res.notes,
            "merge_exec_ok": merge_res.exec_ok,
            "merge_exec_error": merge_res.exec_error,
            "merge_result_preview": merge_res.result_preview,
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("bootstrap", _bootstrap_node)
    workflow.add_node("answer", _answer_node)
    workflow.add_node("merge_sql", _merge_sql_node)
    workflow.set_entry_point("bootstrap")
    workflow.add_edge("bootstrap", "answer")
    workflow.add_edge("answer", "merge_sql")
    workflow.add_edge("merge_sql", END)

    graph = workflow.compile()
    return graph, trace


def build_langchain_sql_agent(settings: Settings) -> Tuple[Any, SQLTrace]:
    """
    用 LangChain 预设的 SQL Agent（create_sql_agent + openai-tools）来获得更贴近“经典 SQL agent”的编排。

    返回：
    - agent executor（可用 invoke，输入 key 为 input）
    - SQLTrace（抓取最终实际执行 SQL）
    """
    llm = ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
    )

    db_path = str(Path(settings.sqlite_db_path).expanduser())
    engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(db_path))
    db = SQLDatabase(engine)

    trace = SQLTrace()
    toolkit = TrackedSQLDatabaseToolkit(db=db, llm=llm, trace=trace)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=settings.show_sql,
        agent_type="openai-tools",
        return_intermediate_steps=False,
    )
    return agent, trace

