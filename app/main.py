from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .chat.service import ChatService, get_chat_service
from .chat.schema import ChatRequest, ChatResponse
import datetime

# === 開発用コマンド ===
# uvicorn app.main:app --reload

# FastAPIアプリケーションの作成
app = FastAPI(
    title="GROQ Chat API",
    description="FastAPIとGROQを使用したチャットAPI",
    version="1.0.0",
)

# CORS設定 - 全てのオリジンからのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なドメインに制限することを推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
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
    print(f"[{datetime.datetime.now()}] 処理開始: /api/chat")
    try:
        # ChatServiceを使用してGROQにメッセージを送信
        print(f"[{datetime.datetime.now()}] ChatService呼び出し開始")
        chat_reply = await chat_service.process_message(chat_request.message)
        print(f"[{datetime.datetime.now()}] ChatService呼び出し完了")

        return ChatResponse(reply=chat_reply)

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"エラーが発生しました: {str(e)}"}
        )


@app.get("/")
async def root():
    return {"message": "GROQ Chat API is running"}
