"""
ベクトル化機能APIスキーマ

文書の追加と検索のためのリクエスト・レスポンスモデルを定義します。
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class AddDocumentRequest(BaseModel):
    """文書追加リクエストモデル"""

    id: str = Field(..., description="文書の一意ID", example="vector-000")
    title: str = Field(..., description="文書のタイトル", example="重要な文書")
    text: str = Field(
        ..., description="ベクトル化するテキスト", example="これは重要な文書です。"
    )


class AddDocumentResponse(BaseModel):
    """文書追加レスポンスモデル"""

    embedding: List[float] = Field(..., description="テキストの特徴量（ベクトル）")


class SearchDocumentsRequest(BaseModel):
    """文書検索リクエストモデル"""

    query: str = Field(..., description="検索クエリ", example="使用方法について")
    n_results: int = Field(5, description="取得する結果数", example=5)


class SearchDocumentsResponse(BaseModel):
    """文書検索レスポンスモデル"""

    results: List[Dict[str, Any]] = Field(..., description="検索結果")


class GetAllDocumentsResponse(BaseModel):
    """全文書取得レスポンスモデル"""

    documents: List[Dict[str, Any]] = Field(..., description="全文書のリスト")
    count: int = Field(..., description="文書の総数")


class GetDocumentResponse(BaseModel):
    """個別文書取得レスポンスモデル"""

    id: str = Field(..., description="文書のID")
    title: str = Field(..., description="文書のタイトル")
    text: str = Field(..., description="文書の内容")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    embedding: List[float] = Field(..., description="文書の特徴量（ベクトル）")


class CollectionInfoResponse(BaseModel):
    """コレクション情報レスポンスモデル"""

    name: str = Field(..., description="コレクション名")
    metadata: Dict[str, Any] = Field(..., description="コレクションのメタデータ")
    count: int = Field(..., description="文書の総数")


class DeleteDocumentRequest(BaseModel):
    """文書削除リクエストモデル"""

    document_id: str = Field(..., description="削除する文書のID")


class DeleteDocumentResponse(BaseModel):
    """文書削除レスポンスモデル"""

    success: bool = Field(..., description="削除成功フラグ")
    message: str = Field(..., description="処理結果メッセージ")


class DeleteAllDocumentsResponse(BaseModel):
    """全文書削除レスポンスモデル"""

    success: bool = Field(..., description="削除成功フラグ")
    deleted_count: int = Field(..., description="削除された文書数")
    message: str = Field(..., description="処理結果メッセージ")
