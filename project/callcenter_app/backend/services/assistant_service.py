from pathlib import Path

from sqlalchemy.orm import Session

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.ai.llm import hugging_llm
from backend.ai.embedding import watson_embedding
from backend.repository.models import CallHistory
from backend.prompts.all_prompt import CALL_ASSISTANT_PROMPT
from backend.schemas.assistant_schema import AssistantResponse

VECTORDB_PATH = Path(__file__).resolve().parents[2] / "vectordb"


def answer_assistant_question(customer_id: int, question: str, db: Session):
    # 1단계: 백터 DB에서 질의

    # 벡터 DB 불러오기
    VECTORDB_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=str(VECTORDB_PATH), embedding_function=watson_embedding
    )

    # as_retriever()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # invoke() => docs => page_content
    docs = retriever.invoke(question)
    sim_context = "\n\n".join(doc.page_content for doc in docs)

    # 2단계: DB검색

    if customer_id:
        histories = (
            db.query(CallHistory)
            .filter(CallHistory.customer_id == customer_id)
            .order_by(CallHistory.created_at.desc())
            .limit(5)
            .all()
        )

        # 고객이 이전에 질의한 내역 추출
        # 문제, 해결 컬럼만 문자열로 추출
        customer_text = "\n\n".join([f"""
                                     일시: {h.created_at}
                                     문제: {h.customer_issue}
                                     해결: {h.resolution}
                                     """ for h in histories])

        # 1,2 단계 => LLM => 적절한 답변 생성
        prompt = ChatPromptTemplate.from_template(CALL_ASSISTANT_PROMPT)

        chain = prompt | hugging_llm | StrOutputParser()
        result = chain.invoke(
            {
                "sim_context": sim_context,
                "customer_text": customer_text,
                "question": question,
            }
        )

        # return {"answer": result}
        return AssistantResponse(answer=result)
