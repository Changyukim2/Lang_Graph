# LangGraph 기반 멀티 에이전트 운동 추천 챗봇

> 사용자의 건강/운동 관련 질문을 입력받아 **증상 추출 → 운동 후보 추천 → 최종 답변 생성**까지 단계적으로 처리하는 LangGraph 기반 멀티 에이전트 시스템입니다.  
> 단일 프롬프트로 한 번에 답변하는 방식이 아니라, 역할이 분리된 여러 Agent를 그래프 구조로 연결하여 더 구조적이고 확장 가능한 LLM 워크플로우를 구현했습니다.

---

## 1. 프로젝트 개요

이 프로젝트는 LangGraph를 활용하여 **멀티 에이전트 기반 운동 추천 챗봇**을 구현한 실습 프로젝트입니다.

사용자가 다음과 같은 질문을 입력하면,

```text
체력이 안좋고, 살이 계속 찌는데 어떤 운동을 할까?
```

시스템은 아래 순서로 답변을 생성합니다.

1. **Extractor Agent**: 사용자 질문에서 증상, 문제점, 신체 상태를 추출합니다.
2. **Matcher Agent**: 추출된 증상과 문제를 바탕으로 적절한 운동 후보를 추천합니다.
3. **Answer Agent**: 증상과 운동 후보를 종합하여 사용자에게 이해하기 쉬운 최종 답변을 생성합니다.

---

## 2. 핵심 목표

- LangGraph의 `StateGraph`를 활용한 멀티 에이전트 흐름 구현
- Agent별 역할 분리와 상태 기반 데이터 전달 구조 이해
- Ollama 로컬 LLM과 LangChain PromptTemplate 연동
- LLM 응답을 단일 출력이 아닌 단계별 추론 파이프라인으로 구성
- 그래프 기반 워크플로우 설계 역량 어필

---

## 3. 기술 스택

| 구분 | 사용 기술 | 설명 |
|---|---|---|
| Language | Python | 전체 구현 언어 |
| LLM Runtime | Ollama | 로컬 환경에서 LLM 실행 |
| LLM Model | exaone3.5:2.4b | Ollama에서 호출한 한국어 LLM |
| Framework | LangChain | PromptTemplate과 LLM 체인 구성 |
| Graph Framework | LangGraph | Agent 간 실행 흐름을 그래프 구조로 제어 |
| State Management | TypedDict | Agent 간 공유되는 상태 스키마 정의 |
| Notebook | Jupyter Notebook | 실험 및 결과 확인 환경 |

---

## 4. 멀티 에이전트 구조

본 프로젝트는 3개의 Agent를 순차적으로 연결한 구조입니다.

```text
START
  ↓
Extractor Agent
  ↓
Matcher Agent
  ↓
Answer Agent
  ↓
END
```

### Agent별 역할

| Agent | 역할 | 입력 | 출력 |
|---|---|---|---|
| Extractor Agent | 사용자 질문에서 핵심 증상/문제 추출 | query | symptoms |
| Matcher Agent | 증상에 맞는 운동 후보 추천 | symptoms | exercise_candidates |
| Answer Agent | 최종 운동 추천 답변 생성 | symptoms, exercise_candidates | result |

---

## 5. LangGraph 상태 설계

LangGraph에서는 각 Agent가 독립적으로 동작하지만, 공통 상태 객체를 통해 데이터를 전달합니다.

```python
class AgentState(TypedDict):
    query: str
    symptoms: str
    exercise_candidates: str
    result: str
```

이 구조를 통해 각 Agent는 필요한 입력만 사용하고, 자신의 처리 결과를 상태에 추가하여 다음 Agent로 전달합니다.

---

## 6. 그래프 구조

아래 이미지는 LangGraph로 구성한 Agent 실행 흐름입니다.

<p align="center">
  <img src="images/langgraph_structure.png" width="500">
</p>

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

이처럼 `StateGraph`를 사용해 Agent를 노드로 등록하고, `add_edge()`를 통해 실행 순서를 명확하게 제어했습니다.

---

## 7. 실행 결과 예시

### 입력 질문

```text
체력이 안좋고, 살이 계속 찌는데 어떤 운동을 할까?
```

### 1단계: 증상 추출 결과

```text
체력 부족, 체중 증가
```

### 2단계: 운동 후보 추천 결과

```text
산책, 조깅, 수영, 자전거 타기, 플랭크, 스쿼트 변형, 런지, 팔굽혀펴기, 레그 레이즈
```

### 3단계: 최종 답변 생성

최종 Agent는 사용자의 문제를 먼저 정리한 뒤, 각 운동이 왜 도움이 되는지와 초보자가 시작할 수 있는 방법을 함께 제시합니다.

<p align="center">
  <img src="images/result_example.png" width="750">
</p>

---

## 8. 주요 코드 설명

### 8.1 Extractor Agent

사용자의 자연어 질문에서 운동 추천에 필요한 핵심 문제만 추출합니다.

```python
extractor_prompt = PromptTemplate.from_template("""
사용자의 질문에서 운동 추천에 필요한 증상, 문제점, 신체 상태를 추출하세요.

조건:
1. 사용자가 말한 핵심 문제만 추출하세요.
2. 결과는 쉼표로 구분된 문자열로 출력하세요.
3. 반드시 한국어로만 출력하세요.
4. 설명 문장은 쓰지 말고 추출 결과만 출력하세요.

질문: {query}
""")
```

### 8.2 Matcher Agent

추출된 증상 및 문제를 바탕으로 운동 후보 리스트를 생성합니다.

```python
matcher_prompt = PromptTemplate.from_template("""
다음 증상 또는 문제를 개선하는 데 도움이 될 수 있는 운동 리스트를 추천하세요.

증상 및 문제:
{symptoms}

조건:
1. 체력이 부족한 초보자도 할 수 있는 운동을 우선 추천하세요.
2. 체중 증가 문제를 고려해 유산소 운동과 근력 운동을 함께 포함하세요.
3. 운동 이름만 쉼표로 구분해서 출력하세요.
4. 반드시 한국어로만 출력하세요.
5. 설명 문장은 쓰지 마세요.
""")
```

### 8.3 Answer Agent

증상과 운동 후보를 종합하여 최종 답변을 생성합니다.

```python
answer_prompt = PromptTemplate.from_template("""
사용자의 증상 및 문제:
{symptoms}

추천 운동 리스트:
{exercise_candidates}

위 내용을 바탕으로 사용자에게 운동 추천 답변을 작성하세요.

출력 조건:
1. 전체 답변은 반드시 한국어로 작성하세요.
2. 개조식으로 작성하세요.
3. 사용자의 문제를 먼저 정리하세요.
4. 추천 운동을 항목별로 설명하세요.
5. 각 운동이 왜 도움이 되는지 간단히 설명하세요.
6. 초보자가 시작할 수 있는 운동 방법도 함께 제시하세요.
7. 무리하지 말고 점진적으로 운동해야 한다는 주의사항을 포함하세요.
""")
```

---

## 9. 프로젝트에서 어필할 수 있는 역량

- LangGraph의 `StateGraph` 기반 워크플로우 설계
- 단일 LLM 호출이 아닌 Agent 역할 분리 구조 구현
- PromptTemplate을 활용한 Agent별 프롬프트 설계
- TypedDict 기반 상태 관리
- Ollama 로컬 LLM 연동 경험
- 멀티 에이전트 파이프라인 구성 및 실행 결과 검증
- 그래프 구조 시각화 및 결과 출력

---

## 10. 개선 방향

현재는 순차 실행 구조의 멀티 에이전트 시스템입니다. 향후에는 다음과 같이 확장할 수 있습니다.

- 조건부 분기 추가: 통증이 심한 경우 운동 추천 대신 병원 방문 안내 Agent로 이동
- RAG 연동: 운동 가이드 문서나 건강 관리 매뉴얼을 검색하여 근거 기반 답변 생성
- Memory 추가: 사용자의 이전 운동 기록과 선호도를 저장하여 개인화 추천 제공
- Agent 확장: 식단 추천 Agent, 주의사항 검토 Agent, 운동 루틴 생성 Agent 추가
- Streamlit 웹앱 배포: 사용자가 직접 질문을 입력하고 추천 결과를 확인할 수 있는 챗봇 서비스 구현

---

## 11. 실행 방법

### 1. Ollama 모델 다운로드

```bash
ollama pull exaone3.5:2.4b
```

### 2. 필요 패키지 설치

```bash
pip install langchain langchain-community langgraph
```

### 3. 노트북 실행

```bash
jupyter notebook lang_graph_prac.ipynb
```

---

## 12. 프로젝트 한 줄 요약

> LangGraph를 활용해 사용자 질문을 증상 추출, 운동 후보 추천, 최종 답변 생성 단계로 분리한 멀티 에이전트 기반 운동 추천 LLM 워크플로우입니다.
