from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .chat.service import ChatService, get_chat_service
from .chat.schema import ChatRequest, ChatResponse
from .vector.service import VectorService, get_vector_service
from .vector.schema import (
    AddDocumentRequest,
    AddDocumentResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    GetAllDocumentsResponse,
    CollectionInfoResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DeleteAllDocumentsResponse,
)
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


@app.post("/api/documents/add", response_model=AddDocumentResponse)
async def add_document(
    request: AddDocumentRequest,
    vector_service: VectorService = Depends(get_vector_service),
) -> AddDocumentResponse:
    """文書をベクトル化してDBに保存

    受け取った文書をベクトル化し、ChromaDBに保存します。

    Args:
        request (AddDocumentRequest): 文書追加リクエスト（id、title、textを含む）
        vector_service (VectorService): DIで注入されるベクトル化サービス

    Returns:
        AddDocumentResponse: ベクトルIDと特徴量を含む保存結果
    """
    print(f"[{datetime.datetime.now()}] 文書追加処理開始")
    try:
        result = await vector_service.add_document(
            id=request.id, title=request.title, text=request.text
        )

        print(f"[{datetime.datetime.now()}] 文書追加処理完了: {result['vector_id']}")

        return AddDocumentResponse(embedding=result["embedding"])

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"文書の保存に失敗しました: {str(e)}"}
        )


@app.post("/api/documents/search", response_model=SearchDocumentsResponse)
async def search_documents(
    request: SearchDocumentsRequest,
    vector_service: VectorService = Depends(get_vector_service),
) -> SearchDocumentsResponse:
    """類似文書を検索

    クエリに類似した文書をベクトルDBから検索します。

    Args:
        request (SearchDocumentsRequest): 検索リクエスト
        vector_service (VectorService): DIで注入されるベクトル化サービス

    Returns:
        SearchDocumentsResponse: 検索結果
    """
    print(f"[{datetime.datetime.now()}] 文書検索処理開始")
    try:
        results = await vector_service.search_similar_documents(
            query=request.query, n_results=request.n_results
        )

        print(f"[{datetime.datetime.now()}] 文書検索処理完了: {len(results)}件")

        return SearchDocumentsResponse(results=results)

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"文書の検索に失敗しました: {str(e)}"}
        )


@app.get("/api/documents/all", response_model=GetAllDocumentsResponse)
async def get_all_documents(
    vector_service: VectorService = Depends(get_vector_service),
) -> GetAllDocumentsResponse:
    """保存されている全ての文書を取得（デバッグ用）"""
    try:
        documents = await vector_service.get_all_documents()
        return GetAllDocumentsResponse(documents=documents, count=len(documents))
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"文書の取得に失敗しました: {str(e)}"}
        )


@app.get("/api/documents/info", response_model=CollectionInfoResponse)
async def get_collection_info(
    vector_service: VectorService = Depends(get_vector_service),
) -> CollectionInfoResponse:
    """コレクションの情報を取得"""
    try:
        info = await vector_service.get_collection_info()
        return CollectionInfoResponse(**info)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"情報の取得に失敗しました: {str(e)}"}
        )


@app.delete("/api/documents/all", response_model=DeleteAllDocumentsResponse)
async def delete_all_documents(
    vector_service: VectorService = Depends(get_vector_service),
) -> DeleteAllDocumentsResponse:
    """保存されている全ての文書を削除"""
    try:
        print(f"[{datetime.datetime.now()}] 全文書削除処理開始")

        result = await vector_service.delete_all_documents()

        print(
            f"[{datetime.datetime.now()}] 全文書削除処理完了: {result['deleted_count']}件削除"
        )

        return DeleteAllDocumentsResponse(
            success=result["success"],
            deleted_count=result["deleted_count"],
            message=f"{result['deleted_count']}件の文書を削除しました",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"全文書の削除に失敗しました: {str(e)}"}
        )


@app.delete("/api/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str, vector_service: VectorService = Depends(get_vector_service)
) -> DeleteDocumentResponse:
    """指定されたIDの文書を削除"""
    try:
        success = await vector_service.delete_document(document_id)
        return DeleteDocumentResponse(
            success=success, message="文書が正常に削除されました"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"文書の削除に失敗しました: {str(e)}"}
        )


@app.get("/")
async def root():
    return {"message": "GROQ Chat API is running"}
