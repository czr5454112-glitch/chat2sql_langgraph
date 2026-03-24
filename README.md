# chat2sql_langgraph

这是一个 **LangGraph 架构**的本地 SQLite 自然语言查询项目：

- 你用中文提问
- 系统调用 DeepSeek（OpenAI 兼容接口）
- 由 LangGraph 驱动 SQL 工具（list tables / schema / execute query）
- 最终输出：**最终答案 / SQL（最终执行版本）/ 查询结果**

## 运行前准备

1. 在 `chat2sql_langgraph/.env` 配置：
   - `DEEPSEEK_API_KEY`
   - `SQLITE_DB_PATH`（你的 `.db` 绝对路径）

2. 创建环境与安装依赖：

```bash
conda create -n chat2sql_langgraph python=3.11 -y
conda activate chat2sql_langgraph
pip install -r requirements.txt
```

## 运行方式

交互式：

```powershell
cd C:\PROGRAMING\chat2sql_langgraph
& "C:\Users\38908\.conda\envs\chat2sql_langgraph\python.exe" -m chat2sql_langgraph.cli
```

单次提问并退出：

```powershell
cd C:\PROGRAMING\chat2sql_langgraph
& "C:\Users\38908\.conda\envs\chat2sql_langgraph\python.exe" -m chat2sql_langgraph.cli --query "这个数据库有哪些表？"
```

## LangGraph 可视化（Mermaid）

运行：

```powershell
cd C:\PROGRAMING\chat2sql_langgraph
& "C:\Users\38908\.conda\envs\chat2sql_langgraph\python.exe" -m chat2sql_langgraph.cli --viz
```

如果想保存到文件：

```powershell
& "C:\Users\38908\.conda\envs\chat2sql_langgraph\python.exe" -m chat2sql_langgraph.cli --viz-file "viz/chat2sql.mmd"
```

