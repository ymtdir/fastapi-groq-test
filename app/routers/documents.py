"""
文書管理機能APIルーター

文書の追加、検索、取得、削除に関するAPIエンドポイントを定義します。
"""

import logging
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

logger = logging.getLogger(__name__)

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
    logger.info(f"文書追加API呼び出し: ID={request.id}")
    logger.debug(f"文書タイトル: {request.title}")

    try:
        result = await vector_service.add_document(
            id=request.id, title=request.title, text=request.text
        )

        logger.info(f"文書追加API処理成功: ID={result['vector_id']}")

        return AddDocumentResponse(embedding=result["embedding"])

    except Exception as e:
        logger.error(f"文書追加API処理エラー: {str(e)}", exc_info=True)
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
    logger.info(
        f"文書検索API呼び出し: クエリ={request.query[:50]}{'...' if len(request.query) > 50 else ''}"
    )
    logger.debug(f"検索結果数: {request.n_results}")

    try:
        results = await vector_service.search_similar_documents(
            query=request.query, n_results=request.n_results
        )

        logger.info(f"文書検索API処理成功: {len(results)}件の文書が見つかりました")

        return SearchDocumentsResponse(results=results)

    except Exception as e:
        logger.error(f"文書検索API処理エラー: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"文書の検索に失敗しました: {str(e)}"}
        )


@router.get("", response_model=GetAllDocumentsResponse)
async def get_all_documents(
    vector_service: VectorService = Depends(get_vector_service),
) -> GetAllDocumentsResponse:
    """保存されている全ての文書を取得"""
    logger.info("全文書取得API呼び出し")

    try:
        documents = await vector_service.get_all_documents()
        logger.info(f"全文書取得API処理成功: {len(documents)}件の文書を取得")
        return GetAllDocumentsResponse(documents=documents, count=len(documents))
    except Exception as e:
        logger.error(f"全文書取得API処理エラー: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"文書の取得に失敗しました: {str(e)}"}
        )


@router.get("/info", response_model=CollectionInfoResponse)
async def get_collection_info(
    vector_service: VectorService = Depends(get_vector_service),
) -> CollectionInfoResponse:
    """コレクションの情報を取得"""
    logger.debug("コレクション情報取得API呼び出し")

    try:
        info = await vector_service.get_collection_info()
        logger.debug("コレクション情報取得API処理成功")
        return CollectionInfoResponse(**info)
    except Exception as e:
        logger.error(f"コレクション情報取得API処理エラー: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"情報の取得に失敗しました: {str(e)}"}
        )


@router.delete("", response_model=DeleteAllDocumentsResponse)
async def delete_all_documents(
    vector_service: VectorService = Depends(get_vector_service),
) -> DeleteAllDocumentsResponse:
    """保存されている全ての文書を削除"""
    logger.warning("全文書削除API呼び出し")

    try:
        result = await vector_service.delete_all_documents()

        logger.warning(f"全文書削除API処理完了: {result['deleted_count']}件削除")

        return DeleteAllDocumentsResponse(
            success=result["success"],
            deleted_count=result["deleted_count"],
            message=f"{result['deleted_count']}件の文書を削除しました",
        )
    except Exception as e:
        logger.error(f"全文書削除API処理エラー: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"全文書の削除に失敗しました: {str(e)}"}
        )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str, vector_service: VectorService = Depends(get_vector_service)
) -> DeleteDocumentResponse:
    """指定されたIDの文書を削除"""
    logger.info(f"文書削除API呼び出し: ID={document_id}")

    try:
        success = await vector_service.delete_document(document_id)
        logger.info(f"文書削除API処理成功: ID={document_id}")
        return DeleteDocumentResponse(
            success=success, message="文書が正常に削除されました"
        )
    except Exception as e:
        logger.error(
            f"文書削除API処理エラー: ID={document_id}, エラー={str(e)}", exc_info=True
        )
        return JSONResponse(
            status_code=500, content={"error": f"文書の削除に失敗しました: {str(e)}"}
        )


@router.get("/{document_id}", response_model=GetDocumentResponse)
async def get_document(
    document_id: str,
    vector_service: VectorService = Depends(get_vector_service),
) -> GetDocumentResponse:
    """指定されたIDの文書を取得"""
    logger.debug(f"個別文書取得API呼び出し: ID={document_id}")

    try:
        document = await vector_service.get_document(document_id)
        logger.debug(f"個別文書取得API処理成功: ID={document_id}")
        return GetDocumentResponse(**document)
    except Exception as e:
        logger.warning(f"個別文書取得API処理失敗: ID={document_id}, エラー={str(e)}")
        return JSONResponse(
            status_code=404, content={"error": f"文書が見つかりません: {str(e)}"}
        )
