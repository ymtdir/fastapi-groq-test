"""
チャット処理サービス

Groq APIを使用したチャット機能を提供します。
Llama3-8b-8192モデルを使用してシンプルなメッセージ応答を行います。
"""

from groq import Groq
import httpx
from typing import Optional, Dict, Any
from ..config import settings
import datetime


class ChatService:
    """チャット処理サービスクラス

    Groq APIを使用したチャット機能を提供するサービスクラス。
    シンプルなメッセージ応答を行います。
    """

    def __init__(self):
        """ChatServiceの初期化

        Groqクライアントを初期化します。
        """
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEYが設定されていません")

        self.groq_client = Groq(
            api_key=settings.GROQ_API_KEY,
        )

    async def process_message(self, message: str) -> str:
        """メッセージを処理してGROQから応答を取得

        Args:
            message (str): ユーザーからのメッセージ

        Returns:
            str: GROQからの応答メッセージ

        Raises:
            Exception: GROQ APIエラーが発生した場合
        """
        if not message.strip():
            raise ValueError("メッセージが空です")

        try:
            # GROQにメッセージを送信
            print(f"[{datetime.datetime.now()}] Groq API呼び出し開始")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model="llama3-8b-8192",  # GROQの利用可能なモデル
                temperature=0.7,
                max_tokens=1024,
            )
            print(f"[{datetime.datetime.now()}] Groq API呼び出し完了")

            # 応答を取得
            reply = chat_completion.choices[0].message.content
            return reply

        except Exception as e:
            raise Exception(f"GROQ APIエラー: {str(e)}")


def get_chat_service() -> ChatService:
    """ChatServiceのインスタンスを取得

    Returns:
        ChatService: ChatServiceのインスタンス
    """
    return ChatService()
