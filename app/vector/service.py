"""
ベクトル化サービス

テキストの特徴量を算出し、ChromaDBに保存する機能を提供します。
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uuid
import datetime
from ..config import settings


class VectorService:
    """ベクトル化サービスクラス

    テキストをベクトルに変換し、ChromaDBに保存する機能を提供します。
    """

    def __init__(self):
        """VectorServiceの初期化

        SentenceTransformerとChromaDBクライアントを初期化します。
        """
        print(f"[{datetime.datetime.now()}] VectorService初期化開始")

        # 日本語に対応したEmbeddingモデルを使用
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

        # ChromaDBクライアントの初期化
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")

        # コレクション（テーブルのようなもの）を取得または作成
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"description": "文書の特徴量を保存するコレクション"},
        )

        print(f"[{datetime.datetime.now()}] VectorService初期化完了")

    async def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """文書をベクトル化してDBに保存

        Args:
            text (str): ベクトル化するテキスト
            metadata (Dict[str, Any], optional): 文書に関するメタデータ

        Returns:
            str: 保存された文書のID

        Raises:
            Exception: ベクトル化またはDB保存に失敗した場合
        """
        try:
            print(f"[{datetime.datetime.now()}] ベクトル化開始")

            # テキストをベクトルに変換
            embedding = self.embedding_model.encode([text])[0].tolist()

            print(f"[{datetime.datetime.now()}] ベクトル化完了")

            # 一意のIDを生成
            doc_id = str(uuid.uuid4())

            # メタデータの準備
            doc_metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "text_length": len(text),
            }
            if metadata:
                doc_metadata.update(metadata)

            print(f"[{datetime.datetime.now()}] ChromaDBへの保存開始")

            # ChromaDBに保存
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
                ids=[doc_id],
            )

            print(f"[{datetime.datetime.now()}] ChromaDBへの保存完了: {doc_id}")

            return doc_id

        except Exception as e:
            print(f"[{datetime.datetime.now()}] エラー: {str(e)}")
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
            print(f"[{datetime.datetime.now()}] 類似文書検索開始")

            # クエリをベクトル化
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # 類似文書を検索
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

            print(
                f"[{datetime.datetime.now()}] 類似文書検索完了: {len(similar_docs)}件"
            )

            return similar_docs

        except Exception as e:
            print(f"[{datetime.datetime.now()}] エラー: {str(e)}")
            raise Exception(f"類似文書の検索に失敗しました: {str(e)}")

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """保存されている全ての文書を取得

        Returns:
            List[Dict[str, Any]]: 全文書のリスト
        """
        try:
            print(f"[{datetime.datetime.now()}] 全文書取得開始")

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

            print(f"[{datetime.datetime.now()}] 全文書取得完了: {len(all_docs)}件")

            return all_docs

        except Exception as e:
            print(f"[{datetime.datetime.now()}] エラー: {str(e)}")
            raise Exception(f"文書の取得に失敗しました: {str(e)}")

    async def get_collection_info(self) -> Dict[str, Any]:
        """コレクションの情報を取得

        Returns:
            Dict[str, Any]: コレクションの情報
        """
        try:
            # コレクションの基本情報
            collection_info = {
                "name": self.collection.name,
                "metadata": self.collection.metadata,
                "count": self.collection.count(),
            }

            return collection_info

        except Exception as e:
            raise Exception(f"コレクション情報の取得に失敗しました: {str(e)}")

    async def delete_document(self, document_id: str) -> bool:
        """指定されたIDの文書を削除

        Args:
            document_id (str): 削除する文書のID

        Returns:
            bool: 削除成功時True
        """
        try:
            print(f"[{datetime.datetime.now()}] 文書削除開始: {document_id}")

            self.collection.delete(ids=[document_id])

            print(f"[{datetime.datetime.now()}] 文書削除完了: {document_id}")

            return True

        except Exception as e:
            print(f"[{datetime.datetime.now()}] エラー: {str(e)}")
            raise Exception(f"文書の削除に失敗しました: {str(e)}")


def get_vector_service() -> VectorService:
    """VectorServiceのインスタンスを取得

    Returns:
        VectorService: VectorServiceのインスタンス
    """
    return VectorService()
