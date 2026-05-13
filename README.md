# LangGraph 기반 운동 추천 멀티 에이전트

> 사용자의 상태를 한 번에 답변하지 않고, **증상 추출 → 운동 후보 추천 → 최종 답변 생성** 단계로 나누어 처리하는 LangGraph 멀티 에이전트 실습 프로젝트입니다.

이 프로젝트는 단순히 LLM에 질문을 던지는 방식이 아니라, 각 단계별 역할을 가진 Agent를 만들고 `StateGraph`로 연결해 하나의 워크플로우처럼 동작하도록 구성했습니다.  
포트폴리오에서는 **LangGraph를 활용해 Agent 간 상태 전달과 순차 실행 흐름을 설계할 수 있음**을 보여주는 용도로 정리했습니다.

---

## 1. 프로젝트 개요

사용자 질문 예시:

```text
체력이 안좋고, 살이 계속 찌는데 어떤 운동을 할까?
```

위 질문을 바로 LLM에게 넘겨 답변하는 것이 아니라, 아래처럼 3단계로 분리했습니다.

1. 사용자의 문제나 증상을 먼저 추출
2. 해당 문제에 맞는 운동 후보 리스트 생성
3. 증상과 운동 후보를 바탕으로 최종 운동 추천 답변 생성

이렇게 나눈 이유는 답변 생성 과정을 명확하게 확인할 수 있고, 각 Agent의 역할을 분리해 유지보수하기 쉽기 때문입니다.

---

## 2. 기술 스택

| 구분 | 사용 기술 |
|---|---|
| LLM 실행 | Ollama |
| LLM 모델 | exaone3.5:2.4b |
| Prompt 구성 | LangChain PromptTemplate |
| Agent Workflow | LangGraph StateGraph |
| 상태 관리 | Python TypedDict |
| 실행 환경 | Jupyter Notebook |
| 시각화 | LangGraph Mermaid Graph |

> 참고: 기존 코드에서는 `langchain_community.llms.Ollama`를 사용했지만, 최신 LangChain에서는 `langchain_ollama` 패키지의 `OllamaLLM` 사용을 권장합니다.

```python
# 기존 코드
from langchain_community.llms import Ollama
llm = Ollama(model="exaone3.5:2.4b")

# 최신 권장 방식
# pip install -U langchain-ollama
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="exaone3.5:2.4b")
```

---

## 3. 멀티 에이전트 구조

이 프로젝트에서는 하나의 LLM을 여러 역할로 나누어 사용했습니다.

| Agent | 역할 | 출력 |
|---|---|---|
| Extractor Agent | 사용자 질문에서 핵심 증상/문제 추출 | `체력 부족, 체중 증가` |
| Matcher Agent | 추출된 문제에 맞는 운동 후보 생성 | `산책, 조깅, 수영, ...` |
| Answer Agent | 최종 사용자 답변 생성 | 운동 추천 설명 |

각 Agent는 이전 단계의 결과를 `AgentState`에 저장하고, 다음 Agent가 해당 값을 이어받아 처리하도록 구성했습니다.

```python
class AgentState(TypedDict):
    query: str
    symptoms: str
    exercise_candidates: str
    result: str
```

---

## 4. LangGraph 실행 흐름

전체 그래프 흐름은 아래와 같습니다.

```text
START
  ↓
extractor
  ↓
matcher
  ↓
answer
  ↓
END
```

<p align="center">
  <img src="images/langgraph_structure.png" width="500">
</p>

그래프는 `StateGraph`를 이용해 직접 구성했습니다.

```python
graph = StateGraph(AgentState)

graph.add_node("extractor", extractor_agent)
graph.add_node("matcher", matcher_agent)
graph.add_node("answer", answer_agent)

graph.set_entry_point("extractor")

graph.add_edge("extractor", "matcher")
graph.add_edge("matcher", "answer")
graph.add_edge("answer", END)

app = graph.compile()
```

---

## 5. 실행 결과

### 입력 질문

```text
체력이 안좋고, 살이 계속 찌는데 어떤 운동을 할까?
```

### 1단계: 증상 추출 결과

```text
체력 부족, 체중 증가
```

### 2단계: 추천 운동 후보

```text
산책, 조깅, 수영, 자전거 타기, 플랭크, 스쿼트 변형, 런지, 팔굽혀펴기, 레그 레이즈
```

### 3단계: 최종 답변 예시

```text
사용자의 주요 문제는 체력 부족과 체중 증가입니다.

추천 운동은 산책, 조깅, 수영, 자전거 타기와 같은 유산소 운동과
플랭크, 스쿼트, 런지, 팔굽혀펴기와 같은 근력 운동으로 구성했습니다.

초보자는 산책처럼 부담이 적은 운동부터 시작하고,
운동 시간과 강도는 점진적으로 늘리는 것이 좋습니다.
```

실제 실행 화면은 아래 이미지처럼 정리했습니다.

<p align="center">
  <img src="images/result_example.png" width="750">
</p>

---

## 6. 구현하면서 신경 쓴 부분

### 1. 역할 분리

처음에는 하나의 프롬프트로 바로 답변을 만들 수도 있었지만, LangGraph를 연습하기 위해 기능을 3개의 Agent로 분리했습니다.  
이렇게 구성하니 어떤 단계에서 어떤 결과가 만들어지는지 확인하기 쉬웠습니다.

### 2. State 기반 데이터 전달

`AgentState`를 정의해서 각 Agent가 만든 결과를 다음 Agent가 사용할 수 있도록 했습니다.  
단순 함수 호출이 아니라, 그래프의 상태가 단계별로 업데이트되는 구조를 직접 구현했습니다.

### 3. Prompt 조건 명확화

각 Agent의 출력 형식을 제한했습니다.

- Extractor Agent: 증상만 쉼표로 출력
- Matcher Agent: 운동 이름만 쉼표로 출력
- Answer Agent: 사용자에게 보여줄 최종 답변 작성

출력 조건을 명확하게 나누면서 LLM 응답을 구조화하는 연습을 할 수 있었습니다.

---

## 7. 핵심 코드

### Extractor Agent

```python
def extractor_agent(state: AgentState):
    chain = extractor_prompt | llm
    symptoms = chain.invoke({"query": state["query"]})

    return {
        **state,
        "symptoms": symptoms.strip()
    }
```

### Matcher Agent

```python
def matcher_agent(state: AgentState):
    chain = matcher_prompt | llm
    exercises = chain.invoke({"symptoms": state["symptoms"]})

    return {
        **state,
        "exercise_candidates": exercises.strip()
    }
```

### Answer Agent

```python
def answer_agent(state: AgentState):
    chain = answer_prompt | llm
    answer = chain.invoke({
        "symptoms": state["symptoms"],
        "exercise_candidates": state["exercise_candidates"]
    })

    return {
        **state,
        "result": answer.strip()
    }
```

---

## 8. 실행 방법

### 1. Ollama 설치 및 모델 다운로드

```bash
ollama pull exaone3.5:2.4b
```

### 2. 필요한 패키지 설치

```bash
pip install langchain langgraph langchain-ollama
```

기존 방식으로 실행하려면 아래 패키지도 필요할 수 있습니다.

```bash
pip install langchain-community
```

### 3. 노트북 실행

```bash
jupyter notebook lang_graph_prac.ipynb
```

---

## 9. 프로젝트를 통해 보여줄 수 있는 역량

이 실습을 통해 아래 내용을 구현했습니다.

- LangGraph의 `StateGraph` 기반 워크플로우 구성
- Agent별 역할 분리
- Agent 간 상태 전달 구조 설계
- LangChain `PromptTemplate`을 이용한 프롬프트 체이닝
- Ollama 로컬 LLM 연동
- 그래프 구조 시각화
- 멀티 에이전트 기반 응답 생성 흐름 구현

특히 단순 LLM 호출이 아니라, **입력 → 중간 결과 → 최종 응답**으로 이어지는 구조를 직접 설계했다는 점을 포트폴리오에서 강조할 수 있습니다.

---

## 10. 개선 방향

현재는 순차 실행 구조로 구성했지만, 이후에는 아래 방향으로 확장할 수 있습니다.

- 조건부 분기 추가  
  예: 통증이 있는 경우 병원 상담 안내 Agent로 분기

- RAG 연결  
  운동 관련 문서나 가이드라인을 검색한 뒤 답변에 반영

- 사용자 프로필 반영  
  나이, 체중, 운동 경험, 부상 여부 등을 상태값에 추가

- 웹 서비스화  
  Streamlit 또는 FastAPI를 이용해 간단한 운동 추천 챗봇으로 확장

---

## 한 줄 정리

> LangGraph를 이용해 증상 추출, 운동 후보 추천, 최종 답변 생성을 분리한 멀티 에이전트 워크플로우를 구현한 프로젝트입니다.
