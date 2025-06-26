"""
ベクトル化サービス

テキストの特徴量を算出し、ChromaDBに保存する機能を提供します。
"""

import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uuid
import datetime
from ..config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """文書管理サービスクラス

    テキストをベクトルに変換し、ChromaDBに保存する機能を提供します。
    文書の追加、検索、取得、削除などの文書管理機能を包括的に提供します。
    """

    def __init__(self):
        """DocumentServiceの初期化

        SentenceTransformerとChromaDBクライアントを初期化します。
        """
        logger.info("DocumentService初期化開始")

        # 日本語に対応したEmbeddingモデルを使用
        logger.debug("Embeddingモデル読み込み開始: intfloat/multilingual-e5-large")
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        logger.info("Embeddingモデル読み込み完了")

        # ChromaDBクライアントの初期化
        logger.debug("ChromaDBクライアント初期化開始")
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")

        # コレクション（テーブルのようなもの）を取得または作成
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"description": "文書の特徴量を保存するコレクション"},
        )
        logger.info("ChromaDBクライアント初期化完了")

        logger.info("DocumentService初期化完了")

    async def add_document(self, id: str, title: str, text: str) -> Dict[str, Any]:
        """文書をベクトル化してDBに保存

        Args:
            id (str): 文書の一意ID
            title (str): 文書のタイトル
            text (str): ベクトル化するテキスト

        Returns:
            Dict[str, Any]: 保存された文書のIDとベクトル（embedding）

        Raises:
            Exception: ベクトル化またはDB保存に失敗した場合
        """
        try:
            logger.info(f"文書追加開始: ID={id}, タイトル={title}")

            # 既存データの確認
            existing = self.collection.get(ids=[id])
            if existing["ids"]:
                logger.warning(f"既存データを上書きします: ID={id}")
            else:
                logger.debug(f"新規文書データです: ID={id}")

            # テキストをベクトルに変換
            logger.debug("ベクトル化処理開始")
            embedding = self.embedding_model.encode([text])[0].tolist()
            logger.debug(f"ベクトル化完了: 次元数={len(embedding)}")

            # メタデータの準備
            doc_metadata = {
                "title": title,
                "created_at": datetime.datetime.now().isoformat(),
                "text_length": len(text),
            }

            logger.debug("ChromaDBへの保存開始")

            # upsertを使用して確実に上書き
            self.collection.upsert(
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
                ids=[id],
            )

            logger.info(f"文書保存完了: ID={id}")

            # 保存後の確認
            saved_data = self.collection.get(ids=[id])
            logger.debug(f"保存後確認完了: データ確認OK")

            return {"vector_id": id, "embedding": embedding}

        except Exception as e:
            logger.error(f"文書保存エラー: ID={id}, エラー={str(e)}", exc_info=True)
            raise Exception(f"文書の保存に失敗しました: {str(e)}")

    async def search_similar_documents(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """類似文書を検索

        Args:
            query (str): 検索クエリ
            n_results (int): 取得する結果数

        Returns:
            List[Dict[str, Any]]: 類似文書のリスト
        """
        try:
            logger.info(
                f"類似文書検索開始: クエリ={query[:50]}{'...' if len(query) > 50 else ''}"
            )
            logger.debug(f"検索結果数: {n_results}")

            # クエリをベクトル化
            logger.debug("検索クエリのベクトル化開始")
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            logger.debug("検索クエリのベクトル化完了")

            # 類似文書を検索
            logger.debug("ChromaDB検索実行")
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            # 結果を整形
            similar_docs = []
            for i in range(len(results["ids"][0])):
                similar_docs.append(
                    {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

            logger.info(
                f"類似文書検索完了: {len(similar_docs)}件の文書が見つかりました"
            )

            return similar_docs

        except Exception as e:
            logger.error(f"類似文書検索エラー: {str(e)}", exc_info=True)
            raise Exception(f"類似文書の検索に失敗しました: {str(e)}")

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """保存されている全ての文書を取得

        Returns:
            List[Dict[str, Any]]: 全文書のリスト
        """
        try:
            logger.info("全文書取得開始")

            # コレクションの全データを取得
            results = self.collection.get()

            # 結果を整形
            all_docs = []
            for i in range(len(results["ids"])):
                all_docs.append(
                    {
                        "id": results["ids"][i],
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

            logger.info(f"全文書取得完了: {len(all_docs)}件の文書を取得")

            return all_docs

        except Exception as e:
            logger.error(f"全文書取得エラー: {str(e)}", exc_info=True)
            raise Exception(f"文書の取得に失敗しました: {str(e)}")

    async def get_collection_info(self) -> Dict[str, Any]:
        """コレクションの情報を取得

        Returns:
            Dict[str, Any]: コレクションの情報
        """
        try:
            logger.debug("コレクション情報取得")

            # コレクションの基本情報
            collection_info = {
                "name": self.collection.name,
                "metadata": self.collection.metadata,
                "count": self.collection.count(),
            }

            logger.debug(f"コレクション情報: {collection_info}")
            return collection_info

        except Exception as e:
            logger.error(f"コレクション情報取得エラー: {str(e)}", exc_info=True)
            raise Exception(f"コレクション情報の取得に失敗しました: {str(e)}")

    async def delete_document(self, document_id: str) -> bool:
        """指定されたIDの文書を削除

        Args:
            document_id (str): 削除する文書のID

        Returns:
            bool: 削除成功時True
        """
        try:
            logger.info(f"文書削除開始: ID={document_id}")

            self.collection.delete(ids=[document_id])

            logger.info(f"文書削除完了: ID={document_id}")

            return True

        except Exception as e:
            logger.error(
                f"文書削除エラー: ID={document_id}, エラー={str(e)}", exc_info=True
            )
            raise Exception(f"文書の削除に失敗しました: {str(e)}")

    async def delete_all_documents(self) -> Dict[str, Any]:
        """保存されている全ての文書を削除

        Returns:
            Dict[str, Any]: 削除結果（削除数と成功フラグ）
        """
        try:
            logger.warning("全文書削除開始")

            # 削除前の文書数を取得
            count_before = self.collection.count()
            logger.info(f"削除対象文書数: {count_before}件")

            # 全文書を取得してIDのリストを作成
            all_docs = self.collection.get()
            all_ids = all_docs["ids"]

            if all_ids:
                # 全IDを指定して一括削除
                self.collection.delete(ids=all_ids)

            # 削除後の文書数を確認
            count_after = self.collection.count()
            deleted_count = count_before - count_after

            logger.warning(f"全文書削除完了: {deleted_count}件削除")

            return {"success": True, "deleted_count": deleted_count}

        except Exception as e:
            logger.error(f"全文書削除エラー: {str(e)}", exc_info=True)
            raise Exception(f"全文書の削除に失敗しました: {str(e)}")

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """指定されたIDの文書を取得

        Args:
            document_id (str): 取得する文書のID

        Returns:
            Dict[str, Any]: 文書の情報（embeddingを含む）

        Raises:
            Exception: 文書が見つからない場合
        """
        try:
            logger.debug(f"個別文書取得開始: ID={document_id}")

            # 指定されたIDの文書を取得（embeddingも含む）
            results = self.collection.get(
                ids=[document_id], include=["documents", "metadatas", "embeddings"]
            )

            if not results["ids"]:
                logger.warning(f"文書が見つかりません: ID={document_id}")
                raise Exception(f"ID {document_id} の文書が見つかりません")

            # 結果を整形
            document = {
                "id": results["ids"][0],
                "title": results["metadatas"][0].get("title", ""),
                "text": results["documents"][0],
                "metadata": results["metadatas"][0],
                "embedding": results["embeddings"][0],
            }

            logger.debug(f"個別文書取得完了: ID={document_id}")

            return document

        except Exception as e:
            logger.error(
                f"個別文書取得エラー: ID={document_id}, エラー={str(e)}", exc_info=True
            )
            raise Exception(f"文書の取得に失敗しました: {str(e)}")


def get_document_service() -> DocumentService:
    """DocumentServiceのインスタンスを取得

    Returns:
        DocumentService: DocumentServiceのインスタンス
    """
    logger.debug("DocumentServiceインスタンス作成")
    return DocumentService()


# 後方互換性のための別名
VectorService = DocumentService
get_vector_service = get_document_service
