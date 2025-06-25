"""
チャット機能APIスキーマ

APIリクエスト・レスポンスのデータ構造を定義します。
Pydanticモデルを使用して、データのバリデーションとドキュメンテーションを自動生成します。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """チャットリクエストモデル"""

    message: str = Field(
        ..., description="ユーザーからのメッセージ", example="こんにちは"
    )


class ChatResponse(BaseModel):
    """チャットレスポンスモデル"""

    reply: str = Field(
        ...,
        description="AIからの応答メッセージ",
        example="こんにちは！何かお手伝いできることはありますか？",
    )
