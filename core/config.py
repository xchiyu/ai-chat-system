# core/config.py
import yaml
import os

class ConfigManager:
    """支持通过点号分隔的键获取配置值，例如 config_manager.get('llm.api_key')"""
    def __init__(self, config_path="config.yaml"):
        self._config = self._load_config(config_path)

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        """支持点号分隔的键，如 'llm.api_key'"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_all(self):
        """返回完整配置字典"""
        return self._config

# 全局单例，项目启动时加载
config_manager = ConfigManager("config.yaml")

# 保留原有 load_config 函数，以便兼容旧代码
def load_config(config_path="config.yaml"):
    return config_manager.get_all()