"""
アプリケーション設定管理

環境変数からアプリケーションの設定を読み込み、
各モジュールで利用できるようにします。
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """アプリケーション設定クラス

    環境変数から各種設定値を取得し、アプリケーション全体で共有する。
    設定されていない必須項目がある場合は、各サービスで適切にエラーハンドリングされる。
    """

    # Groq設定
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # プリザンター設定
    PLEASANTER_BASE_URL = os.getenv("PLEASANTER_BASE_URL", "http://localhost")
    PLEASANTER_API_KEY = os.getenv("PLEASANTER_API_KEY")


# グローバル設定インスタンス
settings = Settings()
