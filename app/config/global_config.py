import os
from pathlib import Path
from typing import Any, Dict

import yaml

class GlobalConfig:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
        return cls._instance

    def load(self):
        """加载父目录下的 config.yml，并加载.env文件"""
        # 定位路径
        rag_root = Path(__file__).resolve().parent.parent.parent
        config_path = rag_root / "config.yml"
        print(f"GLOBAL CONFIG: Attempting to load configuration from {config_path}...")

        if not config_path.exists():
            raise FileNotFoundError(f"GLOBAL CONFIG: could not find {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}
            print(f"GLOBAL CONFIG: loaded config.yml successfully.")

        from dotenv import load_dotenv
        env_path = rag_root / ".env"
        if env_path.exists():
            override = self.get("env_override", False)
            load_dotenv(dotenv_path=env_path, override=override)
            print(f"GLOBAL CONFIG: loaded .env from {env_path} with override={override}.")
        else:
            print(f"GLOBAL CONFIG: no .env file found at {env_path}, skipping environment variable loading.")
        print("GLOBAL CONFIG: configuration loading complete.")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持嵌套（可选扩展）"""
        return self._config.get(key, default)

    @property
    def all(self):
        return self._config

global_config = GlobalConfig()