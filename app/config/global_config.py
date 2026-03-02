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
        """加载父目录下的 config.yml"""
        # 定位路径
        config_path = Path(__file__).resolve().parent.parent.parent / "config.yml"

        if not config_path.exists():
            raise FileNotFoundError(f"未找到配置文件: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}
            print(f"成功从 {config_path} 加载配置")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持嵌套（可选扩展）"""
        return self._config.get(key, default)

    @property
    def all(self):
        return self._config

global_config = GlobalConfig()