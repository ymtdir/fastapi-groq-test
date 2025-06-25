"""
プリザンター チャットAIサービス

プリザンター業務システム専用のRAGチャットAI。
登録されている文書情報を検索し、Groq APIを使用して回答を生成します。
"""

from groq import Groq
from ..config import settings
import datetime


class ChatService:
    """チャットAIサービス

    プリザンター業務システムの文書情報を活用したRAGチャットAI。
    登録されている文書から関連情報を検索し、適切な回答を生成します。
    """

    def __init__(self):
        """ChatServiceの初期化"""
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEYが設定されていません")

        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

        # 文書検索サービス
        from .documents import DocumentService

        self.document_service = DocumentService()

        # プリザンター専用システムプロンプト
        self.system_prompt = """あなたはプリザンター業務システム専用のAIアシスタントです。
            【役割】
            - プリザンターの操作方法や機能について回答
            - 業務に関する質問をサポート
            - 登録されている文書情報を活用した正確な回答

            【回答方針】
            - 提供された参考情報のみを基に回答してください
            - 参考情報に答えがない場合は「申し訳ございませんが、その情報は見つかりませんでした。プリザンターの管理者にお問い合わせください」と回答
            - 推測や想像での回答は行わないでください
            - 回答は分かりやすく、実用的にしてください
            """

    async def process_message(self, message: str) -> str:
        """プリザンターに関する質問に回答（後方互換性用）"""
        return await self.answer_question(message)

    async def answer_question(self, question: str) -> str:
        """プリザンターに関する質問に回答

        Args:
            question (str): ユーザーからの質問

        Returns:
            str: AIからの回答

        Raises:
            ValueError: 質問が空の場合
            Exception: API エラーが発生した場合
        """
        if not question.strip():
            raise ValueError("質問が空です")

        try:
            print(f"[{datetime.datetime.now()}] プリザンター質問処理開始")

            # 1. 関連文書を検索
            related_docs = await self.document_service.search_similar_documents(
                query=question, n_results=3
            )

            # 2. コンテキストを構築
            context = self._build_context(related_docs)

            # 3. プロンプトを作成
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"""
                        参考情報:
                        {context}

                        質問: {question}
                    """,
                },
            ]

            # 4. Groq APIで回答生成
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.3,  # 正確性重視で低めに設定
                max_tokens=1024,
            )

            answer = response.choices[0].message.content
            print(f"[{datetime.datetime.now()}] プリザンター質問処理完了")

            return answer

        except Exception as e:
            print(f"[{datetime.datetime.now()}] エラー: {str(e)}")
            raise Exception(f"回答生成エラー: {str(e)}")

    def _build_context(self, documents: list) -> str:
        """文書リストからコンテキスト文字列を構築"""
        if not documents:
            return "関連する文書情報が見つかりませんでした。"

        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title", f"文書{i}")
            content = doc.get("document", "")

            if content:
                context_parts.append(f"【{title}】\n{content}")

        return (
            "\n\n".join(context_parts)
            if context_parts
            else "関連する文書情報が見つかりませんでした。"
        )


def get_chat_service() -> ChatService:
    """ChatServiceのインスタンスを取得

    Returns:
        ChatService: チャットサービスのインスタンス
    """
    return ChatService()
