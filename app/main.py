import logging.config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat_router, documents_router
from .config.logging import LoggingConfig

# === 開発用コマンド ===
# uvicorn app.main:app --reload

# ログ設定を初期化
logging.config.dictConfig(LoggingConfig.get_logging_config())
logger = logging.getLogger(__name__)

# FastAPIアプリケーションの作成
app = FastAPI(
    title="GROQ Chat API",
    description="FastAPIとGROQを使用したチャットAPI",
    version="1.0.0",
)

logger.info("FastAPIアプリケーション初期化完了")

# CORS設定 - 全てのオリジンからのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なドメインに制限することを推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS設定完了")

# ルーターを登録
app.include_router(chat_router)
app.include_router(documents_router)

logger.info("ルーター登録完了")


@app.get("/")
async def root():
    logger.debug("ルートエンドポイントへのアクセス")
    return {"message": "GROQ Chat API is running"}
