"""
プリザンター チャットAIサービス

プリザンター業務システム専用のRAGチャットAI。
登録されている文書情報を検索し、Groq APIを使用して回答を生成します。
"""

import logging
from groq import Groq
from ..config import settings
import datetime

logger = logging.getLogger(__name__)


class ChatService:
    """チャットAIサービス

    プリザンター業務システムの文書情報を活用したRAGチャットAI。
    登録されている文書から関連情報を検索し、適切な回答を生成します。
    """

    def __init__(self):
        """ChatServiceの初期化"""
        logger.info("ChatService初期化開始")

        if not settings.GROQ_API_KEY:
            logger.error("GROQ_API_KEYが設定されていません")
            raise ValueError("GROQ_API_KEYが設定されていません")

        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("Groqクライアント初期化完了")

        # 文書検索サービス
        from .documents import DocumentService

        self.document_service = DocumentService()
        logger.info("DocumentService初期化完了")

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
            - 質問に対して必要最小限の情報のみ答えてください
            - 挨拶や雑談には簡潔に返し、詳細な説明は求められた場合のみ提供してください
            - 関連情報の自発的な提供は避け、聞かれたことだけに答えてください
            """

        logger.info("ChatService初期化完了")

    async def process_message(self, message: str) -> str:
        """プリザンターに関する質問に回答（後方互換性用）"""
        logger.warning(
            "process_message()は非推奨です。answer_question()を使用してください"
        )
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
            logger.warning("空の質問が送信されました")
            raise ValueError("質問が空です")

        logger.info(
            f"質問処理開始: {question[:50]}{'...' if len(question) > 50 else ''}"
        )

        # 簡単な挨拶や雑談の場合は短い返答を直接返す
        simple_greetings = self._check_simple_greeting(question)
        if simple_greetings:
            logger.info("簡単な挨拶として処理")
            return simple_greetings

        try:
            # 1. 関連文書を検索
            logger.debug("関連文書検索開始")
            related_docs = await self.document_service.search_similar_documents(
                query=question, n_results=3
            )
            logger.info(f"関連文書検索完了: {len(related_docs)}件見つかりました")

            # 2. コンテキストを構築
            context = self._build_context(related_docs)
            logger.debug(f"コンテキスト構築完了: {len(context)}文字")

            # 3. プロンプトを作成
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"""
                        参考情報:
                        {context}

                        質問: {question}
                        
                        注意: この質問に対して、必要最小限の情報のみで簡潔に答えてください。
                    """,
                },
            ]

            # 4. Groq APIで回答生成
            logger.debug("Groq API呼び出し開始")
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.1,  # より一貫した簡潔な回答のため低く設定
                max_tokens=256,  # 回答の長さを制限
            )

            answer = response.choices[0].message.content
            logger.info("Groq API呼び出し完了")
            logger.info(f"質問処理完了: 回答生成成功({len(answer)}文字)")

            return answer

        except Exception as e:
            logger.error(f"質問処理エラー: {str(e)}", exc_info=True)
            raise Exception(f"回答生成エラー: {str(e)}")

    def _check_simple_greeting(self, question: str) -> str:
        """簡単な挨拶や雑談かどうかをチェックし、該当する場合は短い返答を返す"""
        question_lower = question.lower().strip()

        # 挨拶パターン
        greetings = [
            "こんにちは",
            "こんばんは",
            "おはよう",
            "hello",
            "hi",
            "はじめまして",
        ]
        thanks = ["ありがとう", "thank", "感謝"]
        goodbye = ["さようなら", "またね", "bye", "goodbye"]

        for greeting in greetings:
            if greeting in question_lower:
                return "こんにちは！何かお手伝いできることはありますか？"

        for thank in thanks:
            if thank in question_lower:
                return "どういたしまして！"

        for bye in goodbye:
            if bye in question_lower:
                return "また何かありましたらお声かけください！"

        return None

    def _build_context(self, documents: list) -> str:
        """文書リストからコンテキスト文字列を構築"""
        if not documents:
            logger.warning("関連文書が見つかりませんでした")
            return "関連する文書情報が見つかりませんでした。"

        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title", f"文書{i}")
            content = doc.get("document", "")

            if content:
                context_parts.append(f"【{title}】\n{content}")

        result = (
            "\n\n".join(context_parts)
            if context_parts
            else "関連する文書情報が見つかりませんでした。"
        )

        logger.debug(f"コンテキスト構築: {len(context_parts)}個の文書を使用")
        return result


def get_chat_service() -> ChatService:
    """ChatServiceのインスタンスを取得

    Returns:
        ChatService: チャットサービスのインスタンス
    """
    logger.debug("ChatServiceインスタンス作成")
    return ChatService()
