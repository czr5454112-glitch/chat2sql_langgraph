from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from langchain_core.messages import HumanMessage

from .agent_graph import build_langchain_sql_agent, build_langgraph_sql_agent
from .config import Settings


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

    return Settings(
        deepseek_api_key=api_key,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
        sqlite_db_path=sqlite_path,
        show_sql=show_sql,
    )


def _truncate(obj: Any, max_chars: int) -> str:
    text = str(obj)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


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

        if trace.last_sql:
            console.print("\n[bold]SQL（最终执行版本）[/bold]")
            console.print(trace.last_sql)
        if trace.last_result is not None:
            console.print("\n[bold]查询结果（原始返回，可能被截断）[/bold]")
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

        if trace.last_sql:
            console.print("\n[bold]SQL（最终执行版本）[/bold]")
            console.print(trace.last_sql)
        if trace.last_result is not None:
            console.print("\n[bold]查询结果（原始返回，可能被截断）[/bold]")
            console.print(_truncate(trace.last_result, result_max_chars))


if __name__ == "__main__":
    main()

