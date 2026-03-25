from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from rich.console import Console
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .agent_graph import build_langchain_sql_agent, build_langgraph_sql_agent
from .config import Settings
from .sql_merge import merge_sql_round, make_sql_database


console = Console()


def _load_settings() -> Settings:
    # 尽量避免 Windows 控制台中文乱码
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY（请在 .env 中配置）")

    sqlite_path = os.getenv("SQLITE_DB_PATH", "").strip()
    if not sqlite_path:
        raise RuntimeError("缺少环境变量 SQLITE_DB_PATH（请在 .env 中配置）")

    show_sql = os.getenv("CHAT2SQL_SHOW_SQL", "0").strip() in {"1", "true", "True", "yes", "YES"}

    enable_sql_merge = os.getenv("CHAT2SQL_SQL_MERGE", "1").strip() not in {
        "0",
        "false",
        "False",
        "no",
        "NO",
    }
    merge_retries_raw = os.getenv("CHAT2SQL_SQL_MERGE_RETRIES", "1").strip() or "1"
    try:
        sql_merge_max_retries = int(merge_retries_raw)
    except ValueError:
        sql_merge_max_retries = 1

    return Settings(
        deepseek_api_key=api_key,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
        sqlite_db_path=sqlite_path,
        show_sql=show_sql,
        enable_sql_merge=enable_sql_merge,
        sql_merge_max_retries=sql_merge_max_retries,
    )


def _truncate(obj: Any, max_chars: int) -> str:
    text = str(obj)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _make_merge_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
    )


def _print_round_sql_trace(console: Console, trace: Any, result_max_chars: int) -> None:
    if not getattr(trace, "events", None):
        console.print("\n[dim]本轮无 sql_db_query 执行记录。[/dim]")
        return
    console.print("\n[bold]本轮执行的 SQL（按顺序，来自 sql_db_query）[/bold]")
    for i, ev in enumerate(trace.events, start=1):
        console.print(f"\n[dim]--- {i} ---[/dim]")
        console.print(ev.sql)
        console.print(f"[dim]返回摘要（可能被截断）:[/dim] {_truncate(ev.result, result_max_chars)}")


def _print_merged_sql_section(
    console: Console,
    *,
    merged_sql: Optional[str],
    merge_notes: Optional[str],
    merge_exec_ok: Optional[bool],
    merge_exec_error: Optional[str],
    merge_result_preview: Optional[str],
    result_max_chars: int,
) -> None:
    if merge_notes:
        console.print(f"\n[dim]合并说明：{merge_notes}[/dim]")
    if merged_sql:
        console.print("\n[bold]合并后的单条 SQL（后处理生成，已尝试同库执行校验）[/bold]")
        console.print(merged_sql)
    elif merge_notes and "已关闭 SQL 合并" in merge_notes:
        pass
    else:
        console.print("\n[bold yellow]未生成可用的合并 SQL。[/bold yellow]")
    if merge_exec_ok is True:
        console.print("\n[bold green]合并 SQL 执行：成功[/bold green]")
    elif merge_exec_ok is False:
        console.print("\n[bold red]合并 SQL 执行：失败[/bold red]")
        if merge_exec_error:
            console.print(f"[red]{merge_exec_error}[/red]")
    if merge_result_preview:
        console.print("\n[bold]合并 SQL 执行结果预览[/bold]")
        console.print(_truncate(merge_result_preview, result_max_chars))


def _print_langgraph_verbose(state: Any, max_chars: int) -> None:
    # state: {"messages": [...]}
    messages = state.get("messages", []) if isinstance(state, dict) else []
    if not messages:
        return

    console.print("\n[dim]> Entering LangGraph Agent chain...[/dim]")

    # 通过 messages 中的 AIMessage.tool_calls + ToolMessage 来模拟“Invoking / responded”
    pending_calls: dict[str, dict[str, Any]] = {}
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args")
                tc_id = tc.get("id") or tc.get("tool_call_id")
                if name and tc_id:
                    pending_calls[tc_id] = {"name": name, "args": args}
                    console.print(f"[bold cyan]Invoking:[/bold cyan] `{name}` with `{args}`")
            continue

        if getattr(m, "type", None) == "tool" or m.__class__.__name__ == "ToolMessage":
            tc_id = getattr(m, "tool_call_id", None)
            meta = pending_calls.pop(tc_id, None) if tc_id else None
            header = "[green]responded[/green]"
            if meta and meta.get("name"):
                header = f"[green]responded ({meta['name']})[/green]"
            console.print(f"{header}:")
            console.print(_truncate(getattr(m, "content", ""), max_chars))

    console.print("[dim]> Finished chain.[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="chat2sql_langgraph")
    parser.add_argument(
        "--engine",
        type=str,
        default="langgraph",
        choices=["langchain", "langgraph"],
        help="选择执行引擎：langgraph=LangGraph SQL Agent（默认，支持 --viz）；langchain=预设 SQL agent",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="单次提问模式：提供问题后执行一次并退出",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="输出 LangGraph 的 Mermaid 流程图，然后退出",
    )
    parser.add_argument(
        "--viz-file",
        type=str,
        default=None,
        help="把 Mermaid 流程图保存到指定文件（例如 viz/graph.mmd）",
    )
    args = parser.parse_args()

    settings = _load_settings()
    if args.engine == "langgraph":
        agent, trace = build_langgraph_sql_agent(settings)
    else:
        agent, trace = build_langchain_sql_agent(settings)

    result_max_chars = int(os.getenv("CHAT2SQL_RESULT_MAX_CHARS", "4000").strip() or "4000")

    if args.query:
        trace.clear()
        if args.engine == "langgraph":
            state = agent.invoke({"messages": [HumanMessage(content=args.query)]})
            if settings.show_sql:
                _print_langgraph_verbose(state, result_max_chars)
            messages = state.get("messages", [])
            output = messages[-1].content if messages else state
        else:
            result = agent.invoke({"input": args.query})
            output = result.get("output", result) if isinstance(result, dict) else result

        console.print("[bold]最终答案[/bold]")
        console.print(output)

        _print_round_sql_trace(console, trace, result_max_chars)

        if args.engine == "langgraph" and isinstance(state, dict):
            _print_merged_sql_section(
                console,
                merged_sql=state.get("merged_sql"),
                merge_notes=state.get("merge_notes"),
                merge_exec_ok=state.get("merge_exec_ok"),
                merge_exec_error=state.get("merge_exec_error"),
                merge_result_preview=state.get("merge_result_preview"),
                result_max_chars=result_max_chars,
            )
        elif args.engine == "langchain" and settings.enable_sql_merge:
            mllm = _make_merge_llm(settings)
            mdb = make_sql_database(settings)
            mr = merge_sql_round(
                mllm,
                mdb,
                list(trace.events),
                user_question=args.query,
                final_answer=str(output),
                max_retries=max(0, settings.sql_merge_max_retries),
            )
            _print_merged_sql_section(
                console,
                merged_sql=mr.merged_sql,
                merge_notes=mr.notes,
                merge_exec_ok=mr.exec_ok,
                merge_exec_error=mr.exec_error,
                merge_result_preview=mr.result_preview,
                result_max_chars=result_max_chars,
            )
        elif args.engine == "langchain" and not settings.enable_sql_merge:
            console.print("\n[dim]已关闭 SQL 合并（CHAT2SQL_SQL_MERGE=0）。[/dim]")

        if trace.last_result is not None:
            console.print("\n[bold]最后一次 sql_db_query 的原始返回（可能被截断）[/bold]")
            console.print(_truncate(trace.last_result, result_max_chars))
        return

    if args.viz:
        if args.engine != "langgraph":
            console.print("[bold yellow]提示：[/bold yellow] --viz 仅在 --engine langgraph 下可用。")
            console.print("请改用：python -m chat2sql_langgraph.cli --engine langgraph --viz")
            return
        base_graph = agent.get_graph()
        mermaid = base_graph.draw_mermaid()
        if args.viz_file:
            Path(args.viz_file).parent.mkdir(parents=True, exist_ok=True)
            Path(args.viz_file).write_text(mermaid, encoding="utf-8")
            console.print(f"Mermaid 已保存到：{args.viz_file}")
        console.print(mermaid)
        return

    console.print("[bold]chat2sql_langgraph[/bold]：输入自然语言问题，回车发送；输入 exit 退出。")
    messages: list[HumanMessage] = []
    while True:
        try:
            q = console.input("\n[bold cyan]你[/bold cyan]> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n已退出。")
            return

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            console.print("已退出。")
            return

        trace.clear()
        if args.engine == "langgraph":
            # 传入完整消息序列，保证多轮对话上下文
            state = agent.invoke({"messages": messages + [HumanMessage(content=q)]})
            if settings.show_sql:
                _print_langgraph_verbose(state, result_max_chars)
            messages = state.get("messages", messages)
            output = messages[-1].content if messages else state
        else:
            # LangChain 预设 SQL agent 更偏“单轮任务”，这里保持每轮独立以复刻原项目效果
            result = agent.invoke({"input": q})
            output = result.get("output", result) if isinstance(result, dict) else result

        console.print("\n[bold green]最终答案[/bold green]")
        console.print(output)

        _print_round_sql_trace(console, trace, result_max_chars)

        if args.engine == "langgraph" and isinstance(state, dict):
            _print_merged_sql_section(
                console,
                merged_sql=state.get("merged_sql"),
                merge_notes=state.get("merge_notes"),
                merge_exec_ok=state.get("merge_exec_ok"),
                merge_exec_error=state.get("merge_exec_error"),
                merge_result_preview=state.get("merge_result_preview"),
                result_max_chars=result_max_chars,
            )
        elif args.engine == "langchain" and settings.enable_sql_merge:
            mllm = _make_merge_llm(settings)
            mdb = make_sql_database(settings)
            mr = merge_sql_round(
                mllm,
                mdb,
                list(trace.events),
                user_question=q,
                final_answer=str(output),
                max_retries=max(0, settings.sql_merge_max_retries),
            )
            _print_merged_sql_section(
                console,
                merged_sql=mr.merged_sql,
                merge_notes=mr.notes,
                merge_exec_ok=mr.exec_ok,
                merge_exec_error=mr.exec_error,
                merge_result_preview=mr.result_preview,
                result_max_chars=result_max_chars,
            )
        elif args.engine == "langchain" and not settings.enable_sql_merge:
            console.print("\n[dim]已关闭 SQL 合并（CHAT2SQL_SQL_MERGE=0）。[/dim]")

        if trace.last_result is not None:
            console.print("\n[bold]最后一次 sql_db_query 的原始返回（可能被截断）[/bold]")
            console.print(_truncate(trace.last_result, result_max_chars))


if __name__ == "__main__":
    main()

