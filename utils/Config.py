import configparser
import ast
import os
from typing import Any, Dict, Optional


class Config:
    """配置管理类，用于读取和管理超参数"""

    def __init__(self, config_path: str = "config.ini"):
        """
        初始化配置类

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = configparser.ConfigParser()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        self.config.read(config_path, encoding='utf-8')

    def _parse_value(self, value: str) -> Any:
        """
        智能解析配置值的类型

        Args:
            value: 字符串形式的配置值

        Returns:
            解析后的值（可能是int, float, bool, list, dict等）
        """
        value = value.strip()

        # 处理布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 处理None
        if value.lower() == 'none':
            return None

        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # 尝试解析为列表、字典等Python字面量
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        # 如果以上都不是，返回字符串
        return value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            section: 配置节
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        try:
            value = self.config.get(section, key)
            return self._parse_value(value)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if default is not None:
                return default
            raise

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取整个配置节

        Args:
            section: 配置节名称

        Returns:
            包含该节所有配置的字典
        """
        if section not in self.config:
            raise KeyError(f"配置节不存在: {section}")

        items = self.config.items(section)
        return {key: self._parse_value(value) for key, value in items}

    @property
    def loss_weights(self) -> Dict[str, float]:
        """获取损失权重配置"""
        return self.get_section('LOSS_WEIGHTS')

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """获取超参数配置"""
        return self.get_section('HYPERPARAMETERS')


