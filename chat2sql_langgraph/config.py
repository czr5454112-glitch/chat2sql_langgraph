from __future__ import annotations

from pydantic import BaseModel


class Settings(BaseModel):
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    sqlite_db_path: str
    show_sql: bool = False
    # 本轮结束后是否合并多条 SQL 为单条，并在同一数据库上执行校验
    enable_sql_merge: bool = True
    sql_merge_max_retries: int = 1

