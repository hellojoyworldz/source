# LangChain + Ollama 학습 정리

LangChain으로 LLM 앱을 만드는 흐름을 **개념 위주**로 정리한 문서입니다.  
아래 순서대로 읽고, 각 단계마다 직접 코드를 써보면 됩니다.

---

## 시작하기 전에

**필요한 것**
- Python 3.10+
- 패키지: `pip install langchain langchain-ollama langchain-core langchain-community python-dotenv pydantic`
- [Ollama](https://ollama.com) 설치 후 모델 pull (`qwen3.5:4b`, `exaone3.5:2.4b`, `nomic-embed-text-v2-moe`)
- (선택) IBM watsonx API 키 → `.env`에 저장

**큰 그림**

```
[1단계] LLM에 질문하기
    ↓
[2단계] 프롬프트로 역할·형식 지정
    ↓
[3단계] 파이프라인(LCEL)으로 연결
    ↓
[4단계] 출력을 구조화(JSON, Pydantic)
    ↓
[5단계] 대화 이력 관리 (멀티턴)
    ↓
[6단계] RAG — 내 문서 기반으로 답변
    ↓
[7단계] RAG 품질 올리기
```

---

## 1단계. LLM 호출하기

생성형 AI는 **입력(프롬프트) → 출력(텍스트)** 구조입니다.  
LangChain에서는 LLM을 객체로 만들고 `invoke()`로 호출합니다.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3.5:4b")
response = llm.invoke("생성형 AI란?")
print(response.content)
```

**로컬 vs 클라우드**
- `ChatOllama` — 내 PC에서 실행 (무료, 느릴 수 있음)
- `ChatWatsonx` — IBM 클라우드 (API 키 필요, 빠름)
- `ChatOpenAI` + HuggingFace — 외부 API 경유

> **직접 해보기:** 같은 질문을 Ollama 모델 2개에 던져보고 답변 차이를 비교해보세요.

---

## 2단계. 프롬프트

LLM은 역할·형식을 알려줄수록 답변 품질이 올라갑니다.  
프롬프트 종류는 **두 가지**입니다.

### PromptTemplate — 문자열 하나

질문·지시를 **한 덩어리 텍스트**로 만들 때.  
system/human을 나눌 필요 없거나, RAG·구조화 출력처럼 context+question을 한 번에 넣을 때 씁니다. (Chat 모델과도 함께 사용 가능)

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "다음 리뷰를 분석하세요.\n리뷰: {review}\n형식: {format_instructions}"
)
```

### ChatPromptTemplate — 대화형 (챗봇용)

system / human / ai 역할을 **메시지 단위**로 나눌 때.  
`ChatOllama`, `ChatWatsonx` 같은 **Chat 모델**과 함께 쓰는 경우가 많습니다.

만드는 방법도 **두 가지**입니다.

| 방법 | 하는 일 | 언제 쓰나 |
|------|---------|-----------|
| `from_template(...)` | human 메시지 **하나**만 생성 | 질문만 변수로 넣으면 될 때 |
| `from_messages([...])` | system, human, ai 등 **여러 메시지** 조합 | 역할(system) + 질문(human)을 나눌 때 |

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from_template — human 하나
prompt = ChatPromptTemplate.from_template("{question}에 대해 설명해줘")

# from_messages — system + human
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다. 한국어로 답하세요."),
    ("human", "{question}"),
])

# from_messages + MessagesPlaceholder — 멀티턴 (대화 이력 자리)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 요리 전문가입니다."),
    MessagesPlaceholder(variable_name="history"),  # 이전 대화가 여기 들어감
    ("human", "{question}"),
])
```

**정리**
- 챗봇·역할 부여 → `ChatPromptTemplate.from_messages`
- 질문 한 줄만 → `ChatPromptTemplate.from_template` 또는 `PromptTemplate`
- 대화 이력 넣기 → `from_messages` + `MessagesPlaceholder`

`{role}`, `{question}` 같은 변수는 `invoke()`할 때 값을 넣습니다.

> **직접 해보기:** `from_template` vs `from_messages`(system 추가)로 같은 질문을 던져보고 답 차이를 비교해보세요.

---

## 3단계. LCEL 파이프라인

**LCEL** = `|` 로 단계를 이어 붙이는 문법입니다.

```
프롬프트 → LLM → 출력 파서 → 결과
```

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"role": "보안", "question": "XSS란?"})
```

| 메서드 | 언제 쓰나 |
|--------|-----------|
| `invoke()` | 하나 질문, 답 한 번에 받기 |
| `stream()` | 답이 실시간으로 흘러오게 (챗봇 UI) |
| `batch()` | 여러 입력을 한꺼번에 처리 |

**자주 쓰는 보조 도구**
- `RunnablePassthrough` — 입력값을 그대로 다음 단계로 넘김 (RAG에서 질문 전달할 때)
- `RunnableLambda` — Python 함수를 체인 중간에 끼워 넣을 때

> **직접 해보기:** `stream()`으로 출력하면서 `print(chunk, end="")` 해보기.

---

## 4단계. 출력 파서

LLM 답변은 기본적으로 **문자열**입니다.  
정해진 형식(JSON, 필드별 값)으로 받고 싶을 때 파서를 씁니다.

| 파서 | 결과 |
|------|------|
| `StrOutputParser` | 그냥 텍스트 |
| `JsonOutputParser` | dict |
| `PydanticOutputParser` | 검증된 Python 객체 |

**Pydantic이 유용한 이유:** 필드 타입·범위를 미리 정의 → LLM 출력을 자동 검증.

```python
from pydantic import BaseModel, Field
from typing import Literal

class ReviewResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0.0, le=1.0)
    keywords: list[str]

parser = PydanticOutputParser(pydantic_object=ReviewResult)

# format_instructions를 프롬프트에 미리 주입 (.partial)
prompt = PromptTemplate.from_template(
    "리뷰를 분석하세요.\n리뷰: {review}\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"review": "정말 맛있어요!"})
print(result.sentiment)  # positive
```

> **직접 해보기:** 리뷰 한 줄 넣고 sentiment, score, keywords가 객체로 나오는지 확인.

---

## 5단계. 멀티턴 대화 (메모리)

LLM은 **이전 대화를 기억하지 않습니다.**  
매번 질문할 때마다 지금까지의 대화 목록을 같이 보내야 합니다.

**수동 방식:** `messages` 리스트에 HumanMessage / AIMessage를 계속 append

**자동 방식:** `RunnableWithMessageHistory` (2단계 `MessagesPlaceholder` 프롬프트와 함께 사용)

```python
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = prompt | llm | StrOutputParser()

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

cfg = {"configurable": {"session_id": "user_1"}}
with_history.invoke({"question": "파이썬이란?"}, config=cfg)
with_history.invoke({"question": "방금 설명한 장점 3가지는?"}, config=cfg)
```

- `session_id` — 사용자(또는 대화)별로 이력 분리

대화가 길어지면 **Context Window**(한 번에 넣을 수 있는 토큰 한도)를 넘깁니다.  
→ 최근 K턴만 남기거나, 오래된 대화를 요약해서 넣는 전략을 씁니다.

> **직접 해보기:** "내 이름은 OO" → "내 이름 뭐야?" 순서로 물어보고, 이력 없을 때 vs 있을 때 답 차이 확인.

---

## 6단계. RAG — 내 문서로 답하기

**RAG (Retrieval Augmented Generation)**  
LLM 혼자 기억하지 못하는 내용을, **외부 문서에서 찾아서** 답변에 반영하는 방식입니다.  
할루시네이션(없는 내용 지어내기)을 줄이는 데 효과적입니다.

```
질문
  → 관련 문서 검색 (Retriever)
  → 검색 결과 + 질문을 LLM에 전달
  → 답변
```

**RAG 흐름**

1. **문서 로드** — PDF, CSV, 웹 등 → `Document` (본문 + metadata)
2. **청크 분할** — 문서가 길면 잘라야 함 (`chunk_size`, `chunk_overlap`)
3. **임베딩** — 텍스트를 숫자 벡터로 변환 (의미가 비슷하면 벡터도 가까움)
4. **벡터 저장소** — Chroma(디스크 저장) / FAISS(빠른 실험)
5. **Retriever** — 질문과 비슷한 청크 k개 검색
6. **RAG Chain** — 검색 결과를 context로 넣고 LLM 호출

```python
rag_chain = {
    "context": retriever | format_docs,   # 검색 → 문자열로 합치기
    "question": RunnablePassthrough(),    # 질문 그대로 전달
} | rag_prompt | llm | StrOutputParser()
```

**기억할 점**
- 임베딩 모델 ≠ LLM (역할이 다름)
- `chunk_overlap`은 보통 `chunk_size`의 10~20%
- `search_kwargs={"k": 4}` — 몇 개 청크를 가져올지 (많을수록 정확하지만 토큰 증가)

> **직접 해보기:** PDF 하나 로드 → 분할 → 임베딩 → "이 문서에서 OO에 대해 뭐라고 했어?" 질문.

---

## 7단계. RAG 품질 올리기

기본 RAG만으로 부족할 때가 있습니다.

**자주 겪는 문제**
- 비슷한 청크가 중복 검색됨
- 키워드는 맞는데 의미 검색에서 빠짐
- "2024년 이후 작성된 문서만" 같은 조건 검색 불가

**개선 방법**

| 방법 | 하는 일 |
|------|---------|
| chunk_size / overlap 조절 | 청크 크기 바꿔서 검색 품질 실험 |
| 임베딩 모델 변경 | Ollama ↔ Watsonx 등 비교 |
| **MMR** | 관련성 + 다양성 — 비슷한 청크 중복 줄임 |
| **SelfQuery** | LLM이 질문을 "검색어 + 메타데이터 필터"로 자동 변환 |

---

## 마무리 — 한 줄 요약

| 주제 | 핵심 |
|------|------|
| LLM | `invoke()`로 질문, `content`로 답 받기 |
| 프롬프트 | `PromptTemplate`(문자열) / `ChatPromptTemplate`(from_template·from_messages) |
| LCEL | `prompt \| llm \| parser` |
| 파서 | Str / Json / Pydantic — 출력 형식 고정 |
| 멀티턴 | LLM은 기억 못함 → 이력을 매번 같이 보냄 |
| RAG | 문서 검색 + LLM — 내 자료 기반 Q&A |
| RAG 개선 | 청크·임베딩 튜닝, MMR, SelfQuery |

이 순서대로 하나씩 만들어 보면, 챗봇 → 구조화 분석 → 문서 Q&A까지 혼자 따라갈 수 있습니다.
