from __future__ import annotations

from pydantic import BaseModel


class Settings(BaseModel):
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    sqlite_db_path: str
    show_sql: bool = False

