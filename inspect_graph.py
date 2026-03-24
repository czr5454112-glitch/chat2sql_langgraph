from __future__ import annotations

from dotenv import dotenv_values

from chat2sql_langgraph.agent_graph import build_langgraph_sql_agent
from chat2sql_langgraph.config import Settings


def main() -> None:
    conf = dotenv_values("C:/PROGRAMING/chat2sql_langgraph/.env")
    settings = Settings(
        deepseek_api_key=conf.get("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=conf.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        deepseek_model=conf.get("DEEPSEEK_MODEL", "deepseek-chat"),
        sqlite_db_path=conf.get("SQLITE_DB_PATH", ""),
        show_sql=False,
    )
    graph, _trace = build_langgraph_sql_agent(settings)
    print("compiled_type:", type(graph))
    # 输出所有可能的可视化/导出方法名
    keys = ["mermaid", "viz", "draw", "image", "dot", "graph"]
    cands = [m for m in dir(graph) if any(k in m.lower() for k in keys)]
    print("candidates:", cands)

    base_graph = graph.get_graph() if hasattr(graph, "get_graph") else None
    if base_graph is not None:
        print("base_graph_type:", type(base_graph))
        base_cands = [m for m in dir(base_graph) if any(k in m.lower() for k in keys)]
        print("base_graph_candidates:", base_cands)

        if hasattr(base_graph, "draw_mermaid"):
            mm = base_graph.draw_mermaid()
            print("mermaid_len:", len(mm))
            print("mermaid_head:", mm[:400].replace("\n", "\\n"))



if __name__ == "__main__":
    main()

