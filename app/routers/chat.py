"""
チャット機能APIルーター

チャット関連のAPIエンドポイントを定義します。
"""

import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from ..services.chat import ChatService, get_chat_service
from ..schemas.chat import ChatRequest, ChatResponse
import datetime

logger = logging.getLogger(__name__)

# チャット機能用のルーター
router = APIRouter(
    prefix="/api/chat",
    tags=["チャット機能"],
    responses={404: {"description": "Not found"}},
)


@router.post("", response_model=ChatResponse)
async def receive_chat(
    chat_request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """チャットメッセージ処理エンドポイント

    クライアントからのメッセージを受信し、GROQで応答を生成して返します。
    GROQ APIキーが未設定の場合は、エラーメッセージを返却します。

    Args:
        chat_request (ChatRequest): チャットリクエスト
        chat_service (ChatService): DIで注入されるチャットサービス

    Returns:
        ChatResponse: {"reply": "応答メッセージ"} 形式のレスポンス
    """
    logger.info(f"チャットAPI呼び出し開始: /api/chat")
    logger.debug(
        f"受信メッセージ: {chat_request.message[:100]}{'...' if len(chat_request.message) > 100 else ''}"
    )

    try:
        # ChatServiceを使用してGROQにメッセージを送信
        logger.debug("ChatService呼び出し開始")
        chat_reply = await chat_service.process_message(chat_request.message)
        logger.debug("ChatService呼び出し完了")
        logger.info("チャットAPI処理成功")

        return ChatResponse(reply=chat_reply)

    except ValueError as e:
        logger.warning(f"バリデーションエラー: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"チャットAPI処理エラー: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"エラーが発生しました: {str(e)}"}
        )
