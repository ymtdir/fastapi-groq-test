"""
設定管理パッケージ

アプリケーション全体の設定を管理します。
環境変数、ログ設定、外部サービス設定などを統一的に管理します。
"""

from .settings import settings

__all__ = ["settings"]
