# Python 문법 — AI · Agent 엔지니어링 학습지

LLM·RAG·Agent 코드를 **읽고 쓸 때 매일 쓰는 Python**만 골라 정리한 **0과정 학습지**입니다.  
각 장은 **「개념 정의 → 예시 → (필요할 때) AI 맥락」** 순서로 읽히도록 구성했습니다.

**읽는 방법**

1. **제1부** — 자료형·문자열·컬렉션 (데이터 다루기)
2. **제2부** — 조건·반복·함수 (로직·Tool)
3. **제3부** — 모듈·파일·환경변수 (프로젝트 구성)
4. **제4부** — 타입 힌트·Pydantic (구조화 출력·스키마)
5. **제5부** — 예외·데코레이터·async (API·Agent 실전)
6. 각 장 끝 **미니 과제**를 직접 해 보고, **오개념 정리**로 점검

> 문법 전체를 나열하는 교재가 아닙니다. **AI 앱 코드에서 반복 등장하는 패턴**에 집중합니다.  
> 이후 [watsonx/README.md](../watsonx/README.md)(1과정) → [ollama/README.md](../ollama/README.md)(2과정) 순으로 이어지면 자연스럽습니다.

---

## 목차

### 제1부 — 데이터 (1~3장)

**제1장. 변수 · 기본 자료형**

- 1.1 변수 · 할당
- 1.2 `str` · `int` · `float` · `bool` · `None`
- 1.3 형 변환 · `None` 처리

**제2장. 문자열**

- 2.1 f-string · 멀티라인
- 2.2 인덱싱 · 슬라이싱 · `strip`
- 2.3 프롬프트 문자열 패턴

**제3장. 컬렉션**

- 3.1 `list` — 메시지·배치
- 3.2 `dict` — JSON·kwargs
- 3.3 `tuple` · `set` · 중첩 구조

---

### 제2부 — 로직 (4~6장)

**제4장. 조건문 · 반복문**

- 4.1 `if` · `elif` · `else`
- 4.2 `for` · `range` · `enumerate`
- 4.3 `while` · Agent 루프 직관

**제5장. 함수**

- 5.1 `def` · `return` · 기본 인자
- 5.2 `*args` · `**kwargs`
- 5.3 docstring — Tool 설명

**제6장. 컴프리헨션 · 유틸**

- 6.1 리스트·딕셔너리 컴프리헨션
- 6.2 `zip` · `any` · `all`
- 6.3 `lambda` — 간단 변환

---

### 제3부 — 프로젝트 (7~9장)

**제7장. 모듈 · import**

- 7.1 `import` · `from`
- 7.2 `__name__ == "__main__"`
- 7.3 패키지 구조 직관

**제8장. 파일 · JSON**

- 8.1 `open` · `with` · `pathlib`
- 8.2 `json.load` · `json.dumps`
- 8.3 CSV · 텍스트 한 줄 읽기

**제9장. 환경변수**

- 9.1 `os.getenv`
- 9.2 `.env` · `python-dotenv`
- 9.3 API 키·설정 분리

---

### 제4부 — 타입 · 스키마 (10~12장)

**제10장. 타입 힌트**

- 10.1 기본 힌트 · `Optional`
- 10.2 `list[str]` · `dict[str, Any]`
- 10.3 TypedDict — State·메시지

**제11장. dataclass**

- 11.1 `@dataclass` 기본
- 11.2 dataclass vs dict

**제12장. Pydantic**

- 12.1 `BaseModel` · `Field`
- 12.2 검증 · 구조화 출력
- 12.3 LLM 출력 파싱과 연결

---

### 제5부 — 실전 (13~15장)

**제13장. 예외 · 재시도**

- 13.1 `try` · `except` · `finally`
- 13.2 API 호출 실패 처리
- 13.3 재시도 패턴 (개념)

**제14장. 데코레이터**

- 14.1 데코레이터란
- 14.2 `@tool` · `@dataclass` 맥락

**제15장. async · await**

- 15.1 동기 vs 비동기
- 15.2 `async def` · `await` · `asyncio.gather`
- 15.3 `stream` · `ainvoke` 맥락

**부록** — 핵심 정리 · 체크리스트 · 용어 사전

---

## 제1장. 변수 · 기본 자료형

> **이 장에서 배우는 것** — Python 값의 종류, `None` 처리

### 1.1 변수 · 할당

**변수**는 이름에 값을 붙여 두는 것입니다. 타입을 미리 선언하지 않습니다.

```python
model = "qwen3.5:4b"
temperature = 0.7
debug = False
```

```python
a, b = 10, 20          # 동시 할당
messages = []          # 빈 리스트로 시작 — 채팅 이력에 자주 씀
```

---

### 1.2 `str` · `int` · `float` · `bool` · `None`

| 타입    | 예시              | AI 맥락                    |
| ------- | ----------------- | -------------------------- |
| `str`   | `"안녕"`          | 프롬프트·답변·문서 본문    |
| `int`   | `512`             | `max_tokens`, `k`, 턴 수   |
| `float` | `0.0`             | `temperature`, `score`   |
| `bool`  | `True` / `False`  | 플래그·검증 결과           |
| `None`  | `None`            | “값 없음” — **조건 분기** |

```python
type("hello")   # <class 'str'>
type(42)        # <class 'int'>
type(None)      # <class 'NoneType'>
```

---

### 1.3 형 변환 · `None` 처리

```python
int("42")           # 42  — 사용자 입력을 숫자로
str(3.14)           # "3.14"
float("0.7")        # 0.7

text = None
if text is None:    # None 비교는 == 보다 is 권장
    text = ""
```

LLM 응답이 비었을 때 `if not response:` 로 검사하는 패턴이 많습니다.

---

### 1장 미니 과제

| 항목       | 내용                                      |
| ---------- | ----------------------------------------- |
| **과제 1** | `model`, `temperature`, `debug` 변수 선언 |
| **과제 2** | `None`인 변수를 빈 문자열로 바꾸기        |

> **오개념 정리**
>
> - ❌ "Python은 타입 선언 필수" → ✅ **동적 타입**; 타입 힌트는 10장(선택·문서용).
> - ❌ `if x == None` → ✅ `if x is None` (1.3).

**자주 묻는 것**

| 질문 | 답 |
| ---- | -- |
| `None`이랑 `""`(빈 문자열) 차이? | `None` = **값 없음**. `""` = **빈 텍스트**. LLM 응답이 없을 때 둘 중 뭐인지 확인. |
| `if not x`는 언제 True? | `None`, `""`, `[]`, `{}`, `0`, `False`일 때 — **비어 있거나 없음** |
| `0`이랑 `False`는 같은가? | `==`로는 같지만 **타입 다름**. Agent 설정에서 `temperature=0`은 정상 값. |
| 변수 타입 바꿔도 되나? | Python은 **재할당 시 타입 변경 가능**. `a = 1` → `a = "hi"` OK (비권장). |

---

## 제2장. 문자열

> **이 장에서 배우는 것** — f-string, 멀티라인, 프롬프트 조립

### 2.1 f-string · 멀티라인

**f-string** — 변수를 문자열 안에 넣을 때 가장 많이 씁니다.

```python
role = "요리 전문가"
question = "된장찌개 끓이는 법"
prompt = f"당신은 {role}입니다. 질문: {question}"
```

**멀티라인** — system 프롬프트·규칙 블록에 사용합니다.

```python
system = """\
1. 제공된 문서만 근거로 답하세요.
2. 모르면 모른다고 하세요.
"""
```

---

### 2.2 인덱싱 · 슬라이싱 · `strip`

```python
text = "  hello world  \n"
text.strip()        # "hello world" — 앞뒤 공백·줄바꿈 제거
text[:100]          # 앞 100자만 — 로그·미리보기
"error" in text     # 포함 여부
```

---

### 2.3 프롬프트 문자열 패턴

| 패턴 | 예 |
| ---- | -- |
| **변수 삽입** | `f"질문: {question}"` |
| **템플릿 자리** | `"질문: {question}"` → `.format()` 또는 LangChain `PromptTemplate` |
| **문서 붙이기** | `f"문서:\n{context}\n\n질문: {q}"` |

```python
context = "\n\n".join(chunks)
user_msg = f"다음 문서만 참고하세요.\n\n{context}\n\n질문: {query}"
```

2과정 [ollama/README.md](../ollama/README.md) 2장 `PromptTemplate`이 이 문자열을 **체계화**합니다.

---

### 2장 미니 과제

| 항목       | 내용                                           |
| ---------- | ---------------------------------------------- |
| **과제 1** | f-string으로 `role` + `question` 한 문장 만들기 |
| **과제 2** | 3줄 system 규칙을 `"""..."""` 로 작성          |

> **오개념 정리**
>
> - ❌ `"온도 " + temp` (숫자) → ✅ f-string 또는 `str(temp)` (1.3).
> - ❌ 프롬프트에만 집중 → ✅ **strip·슬라이싱**으로 전처리도 문자열 작업 (8장 JSON과 연결).

---

## 제3장. 컬렉션

> **이 장에서 배우는 것** — list·dict 중심, 메시지·JSON 구조

### 3.1 `list` — 메시지·배치

**순서 있는** 값의 목록입니다.

```python
messages = [
    {"role": "user", "content": "안녕"},
    {"role": "assistant", "content": "안녕하세요!"},
]
messages.append({"role": "user", "content": "이름 기억해?"})
len(messages)
messages[-1]          # 마지막 요소
```

| 연산 | 의미 |
| ---- | ---- |
| `append(x)` | 끝에 추가 |
| `extend(xs)` | 여러 개 이어 붙이기 |
| `[start:end]` | 슬라이스 — 최근 K턴만 남길 때 |

```python
recent = messages[-6:]   # 최근 3턴( user+assistant × 3 )
```

---

### 3.2 `dict` — JSON·kwargs

**키 → 값** 매핑. API 요청·응답·LCEL `invoke` 입력이 대부분 dict입니다.

```python
config = {"model": "qwen3.5:4b", "temperature": 0}
config["temperature"] = 0.7
config.get("max_tokens", 512)   # 없으면 기본값 512

invoke_input = {"question": "RAG란?", "role": "교육자"}
```

**key는 한 dict 안에서 유일**합니다. 같은 key에 다시 넣으면 **이전 value를 덮어씁니다.**

```python
{"name": "Alice", "name": "Bob"}   # → {"name": "Bob"}
```

**순회**

```python
for key, value in config.items():
    print(key, value)
```

**언패킹** — 함수·체인에 넘길 때:

```python
params = {"temperature": 0, "num_ctx": 8192}
llm = ChatOllama(model="qwen3.5:4b", **params)   # kwargs로 펼침
```

---

### 3.3 `tuple` · `set` · 중첩 구조

| 타입 | 특징 | AI 맥락 |
| ---- | ---- | ------- |
| `tuple` | 불변 `(a, b)` | 여러 값 반환, 고정 키 묶음 |
| `set` | 중복 없음 | 고유 chunk id, 키워드 집합 |

```python
def search(query):
    return docs, scores          # tuple로 두 값 반환

seen = set()
for doc in docs:
    if doc.id in seen:
        continue
    seen.add(doc.id)
```

**중첩** — JSON·메시지는 dict 안에 list, list 안에 dict가 흔합니다.

```python
response = {
    "choices": [
        {"message": {"role": "assistant", "content": "답변"}}
    ]
}
content = response["choices"][0]["message"]["content"]
```

---

### 3장 미니 과제

| 항목       | 내용                                              |
| ---------- | ------------------------------------------------- |
| **과제 1** | messages list에 user/assistant 2턴 append         |
| **과제 2** | dict `{"question": "..."}` 만들어 키 두 개 접근   |

> **오개념 정리**
>
> - ❌ "JSON ≠ dict" → ✅ Python에서 **json.load → dict·list** (8.2).
> - ❌ list를 dict처럼 `msg["role"]` → ✅ **요소가 dict일 때** 그 안의 키 (3.3).

**자주 묻는 것**

| 질문 | 답 |
| ---- | -- |
| dict key가 **겹칠** 수 있어? | **한 dict 안에서는 불가.** 같으면 **나중 값이 덮어씀**. |
| value는 겹쳐도 되나? | ✅ `"a": 1, "b": 1`처럼 **value는 같아도 됨**. |
| list를 key로 쓸 수 있어? | ❌ **hashable**해야 key 가능. `str`, `int`, `tuple` ✅ / `list`, `dict` ❌ |
| `d["key"]` vs `d.get("key")`? | `[]` — 없으면 **KeyError**. `get` — 없으면 `None` 또는 **기본값** (안전). |
| `for k, v in d.items()`는? | dict를 **키·값 쌍**으로 순회. zip과 비슷하지만 **한 dict** 안을 도는 것. |
| list와 dict 차이? | list = **순서** (`messages[0]`). dict = **이름**으로 접근 (`config["model"]`). |

---

## 제4장. 조건문 · 반복문

> **이 장에서 배우는 것** — 분기·루프, Agent 반복 직관

### 4.1 `if` · `elif` · `else`

```python
score = 0.85
if score >= 0.9:
    label = "high"
elif score >= 0.5:
    label = "medium"
else:
    label = "low"
```

```python
# truthy / falsy — 빈 문자열·빈 list·None은 False로 취급
if not retrieved_docs:
    return "관련 문서를 찾지 못했습니다."
```

---

### 4.2 `for` · `range` · `enumerate`

```python
for chunk in chunks:
    print(chunk.page_content[:80])

for i in range(3):
    print(i)   # 0, 1, 2

for i, doc in enumerate(docs):
    print(f"[{i}] {doc.metadata.get('source')}")
```

**배치 처리** — 여러 입력을 순회할 때:

```python
results = []
for article in articles:
    results.append(chain.invoke({"text": article}))
```

---

### 4.3 `while` · Agent 루프 직관

```python
max_steps = 5
step = 0
while step < max_steps:
  #   response = llm(messages, tools=tools)
  #   if no tool_call: break
  #   execute tool, append result to messages
    step += 1
```

Agent·LangGraph의 **“목표 달성까지 반복”**은 `while` + 조건 탈출 구조와 같습니다 ([ollama/README.md](../ollama/README.md) 10~14장).

---

### 4장 미니 과제

| 항목       | 내용                                    |
| ---------- | --------------------------------------- |
| **과제 1** | 빈 list이면 다른 메시지 반환하는 `if`   |
| **과제 2** | `enumerate`로 문서 3개에 번호 붙여 출력 |

> **오개념 정리**
>
> - ❌ "Agent = for문 한 번" → ✅ **조건 만족까지 반복** (4.3, 10장).
> - ❌ `if len(x) == 0`만 사용 → ✅ `if not x`도 관용적 (4.1).

**자주 묻는 것**

| 질문 | 답 |
| ---- | -- |
| `enumerate` vs `zip` 차이? | **enumerate** = `(인덱스, 항목)` — 번호 붙일 때. **zip** = **두 list를 짝** — `for a, b in zip(xs, ys)` |
| `range(3)`은? | `0, 1, 2` — **3은 포함 안 됨**. |
| `break` vs `continue`? | **break** = 반복 **완전 탈출**. **continue** = 이번만 건너뛰고 **다음 항목**. |
| Agent 루프는 `for`? `while`? | **조건 만족까지 반복**이면 `while` + `break`가 Agent·ReAct와 가깝 (4.3). |

---

## 제5장. 함수

> **이 장에서 배우는 것** — Tool 정의, 인자, docstring

### 5.1 `def` · `return` · 기본 인자

```python
def format_docs(docs):
    """검색 문서를 하나의 문자열로 합칩니다."""
    return "\n\n".join(d.page_content for d in docs)

def search(query, k=4):
    return vectorstore.similarity_search(query, k=k)
```

`k=4` — **기본 인자**. 호출 시 생략 가능.

---

### 5.2 `*args` · `**kwargs`

| 문법 | 의미 |
| ---- | ---- |
| `*args` | 위치 인자 **여러 개** → tuple |
| `**kwargs` | 키워드 인자 **여러 개** → dict |

```python
def log_request(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

log_request("rag", user_id="u1", k=4)
```

LangChain·API 호출에서 `**config` 로 옵션을 넘기는 패턴과 연결됩니다 (3.2).

---

### 5.3 docstring — Tool 설명

함수 **첫 문자열**은 docstring입니다. Agent **Tool**에서는 LLM이 **언제·어떻게** 호출할지 읽는 설명이 됩니다.

```python
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다.

    Args:
        city: 도시 이름 (예: Seoul)
    """
    return f"{city}: 맑음, 22°C"
```

docstring이 빈약하면 **잘못된 tool_call**이 늘어납니다 ([ollama/README.md](../ollama/README.md) 10.2).

---

### 5장 미니 과제

| 항목       | 내용                                           |
| ---------- | ---------------------------------------------- |
| **과제 1** | `format_docs(docs)` — list[str]을 `\n\n`로 합치기 |
| **과제 2** | docstring 있는 함수 1개 — 인자 설명 포함       |

> **오개념 정리**
>
> - ❌ "Tool = API만" → ✅ **일반 함수 + docstring**도 Tool (10장).
> - ❌ `return` 없음 → ✅ `None` 반환 — 의도했는지 확인 (5.1).

**자주 묻는 것**

| 질문 | 답 |
| ---- | -- |
| `return` 안 쓰면? | 함수는 **`None`을 반환**. Tool이 `None`이면 LLM이 혼란 — **항상 의미 있는 값** 반환. |
| `*args` vs `**kwargs`? | `*args` = **위치 인자** 여러 개(tuple). `**kwargs` = **키워드 인자** 여러 개(dict). |
| 기본 인자 `def f(x=[])` 위험? | ⚠️ list/dict **기본값은 한 번만 생성** — 공유됨. Agent Tool에서는 **불변 기본값**(`None`) + 내부 생성 권장. |
| docstring 없어도 Tool 되나? | 동작은 할 수 있지만 LLM이 **언제 호출할지** 몰라 **tool_call 품질↓** (5.3). |

---

## 제6장. 컴프리헨션 · 유틸

> **이 장에서 배우는 것** — 간결한 list/dict 생성, 배치·검사

### 6.1 리스트·딕셔너리 컴프리헨션

```python
# [표현식 for 항목 in 반복 가능]
lengths = [len(c) for c in chunks]

# {키: 값 for ...}
meta_map = {d.id: d.metadata for d in docs}

# 조건 필터
long_chunks = [c for c in chunks if len(c) > 500]
```

`batch` 입력 만들 때:

```python
inputs = [{"question": q} for q in questions]
```

---

### 6.2 `zip` · `any` · `all`

```python
for text, label in zip(texts, labels):
    print(text, label)

any(score > 0.8 for score in scores)   # 하나라도 참?
all(doc.page_content for doc in docs)  # 모두 비어 있지 않?
```

---

### 6.3 `lambda` — 간단 변환

**한 줄** 함수. LCEL `RunnableLambda`의 원형입니다.

```python
format_fn = lambda docs: "\n\n".join(d.page_content for d in docs)
```

복잡한 로직은 `def`가 낫습니다. 가독성 우선.

---

### 6장 미니 과제

| 항목       | 내용                                      |
| ---------- | ----------------------------------------- |
| **과제 1** | 컴프리헨션으로 문자열 list 각 길이 list   |
| **과제 2** | `inputs = [{"text": t} for t in ...]`   |

> **오개념 정리**
>
> - ❌ 모든 루프를 컴프리헨션 → ✅ **읽기 어려우면 for** (6.1).
> - ❌ lambda 남용 → ✅ 체인 중간 **짧은 변환**만 (6.3, ollama 3.3).

---

## 제7장. 모듈 · import

> **이 장에서 배우는 것** — 코드 분리, 재사용

### 7.1 `import` · `from`

```python
import json
from pathlib import Path
from langchain_ollama import ChatOllama

from mod1 import add          # mod1.py의 add만
from mod1 import *            # 전부 — 이름 충돌 위험, 비권장
```

| 스타일 | 예 |
| ------ | -- |
| `import m` | `m.add(1, 2)` |
| `from m import add` | `add(1, 2)` |

---

### 7.2 `__name__ == "__main__"`

파일이 **직접 실행**될 때만 아래 블록이 돕니다. import될 때는 실행되지 않습니다.

```python
def main():
    print("RAG 파이프라인 시작")

if __name__ == "__main__":
    main()
```

스크립트 vs 라이브러리 분리에 씁니다 (`basic/mod1.py` 참고).

---

### 7.3 패키지 구조 직관

```
my_project/
  .env
  rag.py          # 메인 로직
  tools.py        # @tool 함수들
  prompts.py      # 프롬프트 문자열
```

`from tools import get_weather` — **역할별 파일 분리**가 Agent 프로젝트에서도 동일합니다.

---

### 7장 미니 과제

| 항목       | 내용                                |
| ---------- | ----------------------------------- |
| **과제 1** | `mod1.add` import 해서 호출         |
| **과제 2** | `if __name__ == "__main__"` 의미 설명 |

> **오개념 정리**
>
> - ❌ 한 파일에 전부 → ✅ **tools / prompts / chain** 분리 (7.3).
> - ❌ `from x import *` 습관 → ✅ **필요한 이름만** (7.1).

---

## 제8장. 파일 · JSON

> **이 장에서 배우는 것** — 문서 로드, 설정·데이터 저장

### 8.1 `open` · `with` · `pathlib`

```python
from pathlib import Path

path = Path("./data/guide.pdf")
path.exists()

with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

with open("out.txt", "w", encoding="utf-8") as f:
    f.write("결과 저장")
```

`with` — 파일을 **자동으로 닫**습니다 (context manager).

---

### 8.2 `json.load` · `json.dumps`

```python
import json

with open("config.json", "r", encoding="utf-8") as f:
    data = json.load(f)          # 파일 → dict/list

text = json.dumps(data, ensure_ascii=False, indent=2)  # dict → 문자열
parsed = json.loads(text)        # 문자열 → dict
```

LLM이 **JSON만 출력**하라고 할 때 `json.loads(response)` 로 파싱합니다. 실무에서는 Pydantic(12장)이 더 안전합니다.

---

### 8.3 CSV · 텍스트 한 줄 읽기

```python
lines = Path("reviews.csv").read_text(encoding="utf-8").splitlines()
# CSV는 pandas·csv 모듈 또는 LangChain CSVLoader (ollama 6장)
```

RAG **원문**은 결국 파일 → 문자열 → chunk 파이프라인으로 들어갑니다.

---

### 8장 미니 과제

| 항목       | 내용                                      |
| ---------- | ----------------------------------------- |
| **과제 1** | json 파일 dict로 읽고 키 하나 출력        |
| **과제 2** | dict를 `json.dumps`로 저장                |

> **오개념 정리**
>
> - ❌ JSON에 작은따옴표 → ✅ **쌍따옴표** (`"`); Python dict와 문자열 구분 (8.2).
> - ❌ `encoding` 생략 → ✅ 한글은 **`utf-8`** (8.1).

**자주 묻는 것**

| 질문 | 답 |
| ---- | -- |
| Python **dict** vs **JSON 문자열**? | dict = 메모리 객체. JSON 문자열 = `'{"a": 1}'` **텍스트**. `json.loads()`로 변환. |
| `json.load` vs `json.loads`? | **load** = **파일**에서. **loads** = **문자열**에서 (s = string). |
| LLM 출력 JSON 파싱? | `json.loads(response)` — 실패하면 LLM이 형식 어긴 것 → Pydantic·재요청 (12장). |
| Python `{'a': 1}` 저장하면 JSON? | 파일에 쓸 땐 **`json.dumps()`** — 작은따옴표 dict 리터럴 그대로 저장 ❌ |

---

## 제9장. 환경변수

> **이 장에서 배우는 것** — API 키·설정을 코드 밖으로

### 9.1 `os.getenv`

```python
import os

api_key = os.getenv("WATSONX_API_KEY")
if not api_key:
    raise ValueError("WATSONX_API_KEY가 설정되지 않았습니다.")
```

---

### 9.2 `.env` · `python-dotenv`

프로젝트 루트 `.env` 파일 (git에 **올리지 않음**):

```
WATSONX_API_KEY=your_key_here
WATSONX_PROJECT_ID=your_project_id
```

```python
from dotenv import load_dotenv

load_dotenv()   # .env → 환경변수로 로드
api_key = os.getenv("WATSONX_API_KEY")
```

---

### 9.3 API 키·설정 분리

| 넣지 말 것 | 넣을 것 |
| ---------- | ------- |
| 코드 안 `api_key = "sk-..."` | `.env` · 환경변수 |
| Git에 `.env` 커밋 | `.gitignore`에 `.env` |

1·2과정 예제는 대부분 `load_dotenv()` 후 `os.getenv` 패턴입니다.

---

### 9장 미니 과제

| 항목       | 내용                           |
| ---------- | ------------------------------ |
| **과제 1** | getenv로 없는 키 → 기본값 사용 |
| **과제 2** | 왜 API 키를 .env에 두는지 설명 |

> **오개념 정리**
>
> - ❌ "로컬만 쓰니까 키 노출 OK" → ✅ **저장소·스크린샷** 유출 (9.3).
> - ❌ `.env`만으로 암호화 → ✅ **비밀 관리**는 별도; .env는 **개발 편의** (9.2).

---

## 제10장. 타입 힌트

> **이 장에서 배우는 것** — 함수·데이터 형태를 코드에 표시

### 10.1 기본 힌트 · `Optional`

```python
def search(query: str, k: int = 4) -> list:
    ...

from typing import Optional

def find_user(user_id: str) -> Optional[dict]:
    ...
```

힌트는 **실행에 강제되지 않습니다**. IDE·독자·Pydantic과 함께 쓸 때 유리합니다.

---

### 10.2 `list[str]` · `dict[str, Any]`

Python 3.9+ :

```python
def format_docs(docs: list[str]) -> str:
    return "\n\n".join(docs)

from typing import Any

def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    ...
```

---

### 10.3 TypedDict — State·메시지

**dict의 키·값 타입**을 고정해 둔 형태. LangGraph **State**에 씁니다.

```python
from typing import TypedDict

class RagState(TypedDict):
    query: str
    answer: str
```

[ollama/README.md](../ollama/README.md) 14.2 — `RagState`, `retrieved_docs` 등.

---

### 10장 미니 과제

| 항목       | 내용                                      |
| ---------- | ----------------------------------------- |
| **과제 1** | `(query: str, k: int) -> list` 함수 껍데기 |
| **과제 2** | TypedDict로 `query`, `answer` 필드 정의   |

> **오개념 정리**
>
> - ❌ 힌트 안 지키면 실행 오류 → ✅ **문서용**; 검증은 Pydantic (12장).
> - ❌ TypedDict = Pydantic → ✅ TypedDict는 **가벼운 dict 스키마** (10.3 vs 12장).

---

## 제11장. dataclass

> **이 장에서 배우는 것** — 데이터 묶음 클래스

### 11.1 `@dataclass` 기본

```python
from dataclasses import dataclass

@dataclass
class SearchResult:
    doc_id: str
    score: float
    snippet: str
```

`__init__`·`__repr__`을 자동 생성합니다.

---

### 11.2 dataclass vs dict

|      | dict | dataclass |
| ---- | ---- | --------- |
| 유연 | ✅   | 필드 고정 |
| IDE  | △    | ✅ 자동완성 |
| JSON | 바로 | `asdict()` 필요 |

LLM **구조화 출력**은 Pydantic(12장)이 더 많습니다. dataclass는 **내부 데이터** 묶음에 적합합니다.

---

### 11장 미니 과제

| 항목       | 내용                                |
| ---------- | ----------------------------------- |
| **과제 1** | `doc_id`, `score` dataclass 정의    |

> **오개념 정리**
>
> - ❌ 모든 구조에 dataclass → ✅ **검증·LLM 연동**은 Pydantic (12장).

---

## 제12장. Pydantic

> **이 장에서 배우는 것** — 스키마 검증, 구조화 출력

### 12.1 `BaseModel` · `Field`

```python
from pydantic import BaseModel, Field
from typing import Literal

class ReviewResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0.0, le=1.0)
    keywords: list[str]
```

| Field | 의미 |
| ----- | ---- |
| `ge=0, le=1` | 0~1 범위 |
| `description` | 필드 설명 (스키마 export) |

---

### 12.2 검증 · 구조화 출력

```python
ReviewResult(sentiment="positive", score=0.9, keywords=["맛", "배송"])
# 잘못된 타입·범위 → ValidationError

data = {"sentiment": "positive", "score": 0.9, "keywords": ["a"]}
result = ReviewResult.model_validate(data)
```

LLM이 JSON을 **지키지 않을 때** 파서 단계에서 걸러집니다 ([ollama/README.md](../ollama/README.md) 4장).

---

### 12.3 LLM 출력 파싱과 연결

```python
parser = PydanticOutputParser(pydantic_object=ReviewResult)
format_instructions = parser.get_format_instructions()
# 프롬프트에 format_instructions 삽입 → chain | parser
```

**흐름:** 스키마 정의 → 프롬프트에 형식 주입 → LLM 출력 → `model_validate` / 파서.

---

### 12장 미니 과제

| 항목       | 내용                                           |
| ---------- | ---------------------------------------------- |
| **과제 1** | `title: str`, `score: float` 모델 + 검증 실패 케이스 |
| **과제 2** | dict를 `model_validate`로 객체 변환            |

> **오개념 정리**
>
> - ❌ Pydantic = DB ORM → ✅ **데이터 검증·스키마** (12.2).
> - ❌ JSON 파싱만으로 충분 → ✅ **필드·범위** 검증은 Pydantic (ollama 4.4).

**자주 묻는 것**

| 질문 | 답 |
| ---- | -- |
| dict vs `BaseModel`? | dict = **검증 없음**. BaseModel = **필드·타입·범위** 자동 검증. |
| `ReviewResult(...)` vs `model_validate(dict)`? | 생성자 = **키워드**로 직접. `model_validate` = **dict/JSON**에서 변환 — LLM 출력에 주로 사용. |
| 검증 실패하면? | **ValidationError** — try/except 또는 파서가 재시도 (13장). |
| `result.score` vs `result["score"]`? | BaseModel은 **속성** `.score`로 접근. dict처럼 `["score"]`도 가능하지만 **`.`가 일반적**. |

---

## 제13장. 예외 · 재시도

> **이 장에서 배우는 것** — API 실패 대응

### 13.1 `try` · `except` · `finally`

```python
try:
    response = llm.invoke("질문")
except TimeoutError:
    response = "요청 시간 초과"
except Exception as e:
    response = f"오류: {e}"
finally:
    pass   # 정리 작업 (연결 close 등)
```

**구체적 예외**를 먼저, `Exception`은 마지막에 잡는 편이 좋습니다.

---

### 13.2 API 호출 실패 처리

| 실패 | 대응 |
| ---- | ---- |
| timeout | 재시도·타임아웃 늘리기 |
| rate limit | sleep 후 재시도 |
| 잘못된 JSON | 파서 예외 → 프롬프트 수정 |

```python
try:
    result = parser.invoke({"review": text})
except ValidationError:
    result = None   # fallback 또는 재요청
```

---

### 13.3 재시도 패턴 (개념)

```
for attempt in range(3):
    try:
        return api_call()
    except TransientError:
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)   # 지수 백오프
```

[watsonx/README.md](../watsonx/README.md) 6.7 — rate limit·재시도와 연결.

---

### 13장 미니 과제

| 항목       | 내용                              |
| ---------- | --------------------------------- |
| **과제 1** | 0으로 나누기 `try/except`         |
| **과제 2** | 실패 시 기본 문자열 반환 패턴 작성 |

> **오개념 정리**
>
> - ❌ bare `except:` → ✅ **예외 타입** 지정 (13.1).
> - ❌ 모든 오류 무시 → ✅ **로그·재시도 상한** (13.3, ollama 11.6).

---

## 제14장. 데코레이터

> **이 장에서 배우는 것** — `@` 로 함수 감싸기, Tool·dataclass

### 14.1 데코레이터란

**함수를 인자로 받아 새 함수를 반환**하는 패턴. `@이름` 은 그걸 짧게 쓴 문법입니다.

```python
def my_decorator(fn):
    def wrapper(*args, **kwargs):
        print("before")
        return fn(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    return f"Hi, {name}"
```

---

### 14.2 `@tool` · `@dataclass` 맥락

| 데코레이터 | 하는 일 |
| ---------- | ------- |
| `@dataclass` | 데이터 클래스 필드·생성자 자동 (11장) |
| `@tool` | 함수를 Agent **Tool**로 등록 + 스키마 생성 (ollama 10.2) |

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b
```

**본질**은 여전히 **함수 + docstring + 타입 힌트**입니다. 데코레이터가 메타데이터를 붙입니다.

---

### 14장 미니 과제

| 항목       | 내용                                    |
| ---------- | --------------------------------------- |
| **과제 1** | `@tool` 함수 하나 — docstring 포함      |

> **오개념 정리**
>
> - ❌ 데코레이터 = 새 언어 → ✅ **함수를 감싸는 함수** (14.1).
> - ❌ Tool은 클래스만 → ✅ **@tool 함수**가 일반적 (14.2).

---

## 제15장. async · await

> **이 장에서 배우는 것** — 비동기 호출, stream·병렬

### 15.1 동기 vs 비동기

|      | 동기 `invoke` | 비동기 `ainvoke` |
| ---- | ------------- | ---------------- |
| 대기 | 호출이 끝날 때까지 **블로킹** | 다른 작업 **양보** 가능 |
| 적합 | 단순 스크립트 | **여러 API** 동시·서버 |

---

### 15.2 `async def` · `await` · `asyncio.gather`

```python
import asyncio

async def fetch_one(text: str):
    return await chain.ainvoke({"text": text})

async def fetch_all(texts: list[str]):
    tasks = [fetch_one(t) for t in texts]
    return await asyncio.gather(*tasks)

# asyncio.run(fetch_all(["a", "b", "c"]))
```

`await` — **비동기 작업이 끝날 때까지** 기다리되, 이벤트 루프에 제어를 넘깁니다.

---

### 15.3 `stream` · `ainvoke` 맥락

```python
# 동기 스트리밍 (ollama 3.2)
for chunk in chain.stream({"question": "..."}):
    print(chunk, end="")
```

스트리밍은 **토큰이 도착할 때마다** yield. async 서버에서는 `astream` 변형을 씁니다.  
처음에는 **동기 `invoke` / `stream`** 만 익혀도 충분합니다.

---

### 15장 미니 과제

| 항목       | 내용                                      |
| ---------- | ----------------------------------------- |
| **과제 1** | 동기 vs 비동기 차이 한 줄로 설명          |
| **과제 2** | `gather`가 하는 일 설명                   |

> **오개념 정리**
>
> - ❌ async가 항상 더 빠름 → ✅ **I/O 대기 병렬**에 유리; CPU 연산은 별개 (15.1).
> - ❌ 모든 코드를 async로 → ✅ **필요할 때** (다건 API, 웹 서버) (ollama 11.5).

---

## 부록

### 핵심 정리

| 주제 | AI·Agent에서 쓰는 이유 |
| ---- | ---------------------- |
| f-string · dict · list | 프롬프트·messages·invoke 입력 |
| 함수 · docstring | Tool 정의 |
| json · Path | 설정·문서·RAG 데이터 |
| os.getenv · dotenv | API 키 |
| 타입 힌트 · TypedDict | State·메시지 구조 |
| Pydantic | 구조화 출력·검증 |
| try/except | API·파싱 실패 |
| @tool | Agent Tool 등록 |
| async | 다건 처리·서버 |

---

### 학습 체크리스트

- [ ] **1~3장** — 자료형, f-string, list·dict·중첩
- [ ] **4~6장** — if/for/while, def·kwargs, 컴프리헨션
- [ ] **7~9장** — import, json, .env
- [ ] **10~12장** — 타입 힌트, TypedDict, Pydantic
- [ ] **13~15장** — 예외, @tool, async 개념

**다음 단계:** [watsonx/README.md](../watsonx/README.md) (LLM 개념) → [ollama/README.md](../ollama/README.md) (RAG·Agent 실습)

---

### 용어 사전

| 용어 | 한 줄 정의 | 해당 장 |
| ---- | ---------- | ------- |
| **f-string** | `f"{변수}"` 형태 문자열 포맷 | 2.1 |
| **kwargs** | `**dict`로 키워드 인자 펼침 | 3.2, 5.2 |
| **docstring** | 함수 설명 문자열 — Tool에서 LLM이 읽음 | 5.3 |
| **컴프리헨션** | `[... for x in ...]` 간결한 list 생성 | 6.1 |
| **TypedDict** | 키·타입이 정해진 dict 스키마 | 10.3 |
| **BaseModel** | Pydantic 검증 모델 | 12.1 |
| **데코레이터** | `@`로 함수를 감싸 동작 확장 | 14.1 |
| **await** | 비동기 작업 완료 대기 | 15.2 |
