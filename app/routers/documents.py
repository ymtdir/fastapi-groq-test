"""
文書管理機能APIルーター

文書の追加、検索、取得、削除に関するAPIエンドポイントを定義します。
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from ..services.documents import VectorService, get_vector_service
from ..schemas.documents import (
    AddDocumentRequest,
    AddDocumentResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    GetAllDocumentsResponse,
    CollectionInfoResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DeleteAllDocumentsResponse,
    GetDocumentResponse,
)
import datetime

# 文書管理機能用のルーター
router = APIRouter(
    prefix="/api/documents",
    tags=["文書管理機能"],
    responses={404: {"description": "Not found"}},
)


@router.post("", response_model=AddDocumentResponse)
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
        AddDocumentResponse: 特徴量を含む保存結果
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


@router.post("/search", response_model=SearchDocumentsResponse)
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


@router.get("", response_model=GetAllDocumentsResponse)
async def get_all_documents(
    vector_service: VectorService = Depends(get_vector_service),
) -> GetAllDocumentsResponse:
    """保存されている全ての文書を取得"""
    try:
        documents = await vector_service.get_all_documents()
        return GetAllDocumentsResponse(documents=documents, count=len(documents))
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"文書の取得に失敗しました: {str(e)}"}
        )


@router.get("/info", response_model=CollectionInfoResponse)
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


@router.delete("", response_model=DeleteAllDocumentsResponse)
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


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
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


@router.get("/{document_id}", response_model=GetDocumentResponse)
async def get_document(
    document_id: str,
    vector_service: VectorService = Depends(get_vector_service),
) -> GetDocumentResponse:
    """指定されたIDの文書を取得"""
    try:
        document = await vector_service.get_document(document_id)
        return GetDocumentResponse(**document)
    except Exception as e:
        return JSONResponse(
            status_code=404, content={"error": f"文書が見つかりません: {str(e)}"}
        )
