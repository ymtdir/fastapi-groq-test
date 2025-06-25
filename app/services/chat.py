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

        # DocumentServiceを追加
        from .documents import DocumentService

        self.document_service = DocumentService()

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
            # 1. 関連文書を検索
            similar_docs = await self.document_service.search_similar_documents(
                query=message, n_results=3
            )

            # 2. コンテキストを作成
            context = "\n".join([doc["document"] for doc in similar_docs])

            # 3. RAGプロンプトを作成
            prompt = f"""以下の情報だけを参考にして、質問に答えてください。
参考情報がない場合は、「申し訳ございませんが、その情報は見つかりませんでした」と答えてください。

参考情報:
{context}

質問: {message}"""

            # 4. GROQに送信
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024,
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            raise Exception(f"RAG処理エラー: {str(e)}")


def get_chat_service() -> ChatService:
    """ChatServiceのインスタンスを取得

    Returns:
        ChatService: ChatServiceのインスタンス
    """
    return ChatService()
