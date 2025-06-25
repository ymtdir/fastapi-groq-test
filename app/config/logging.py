"""
ログ設定管理

アプリケーション全体のログ設定を管理します。
将来的にここにログレベル、フォーマット、出力先などの設定を追加予定。
"""

import logging
import os
from typing import Dict, Any


class LoggingConfig:
    """ログ設定クラス

    将来的にログレベル、フォーマット、ハンドラーなどの設定を管理。
    """

    # 基本設定（将来拡張予定）
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """ログ設定辞書を取得

        Returns:
            Dict[str, Any]: ログ設定辞書
        """
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": cls.LOG_FORMAT,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": cls.LOG_LEVEL,
                "handlers": ["default"],
            },
        }


# ログ設定インスタンス
logging_config = LoggingConfig()
