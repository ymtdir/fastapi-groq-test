from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat_router, documents_router

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

# ルーターを登録
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/")
async def root():
    return {"message": "GROQ Chat API is running"}
