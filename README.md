# LangGraph 기반 운동 추천 멀티 에이전트

> 사용자의 상태를 한 번에 답변하지 않고, **증상 추출 → 운동 후보 추천 → 최종 답변 생성** 단계로 나누어 처리하는
> LangGraph 멀티 에이전트 실습 프로젝트입니다.

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
<img width="108" height="432" alt="output" src="https://github.com/user-attachments/assets/76c30794-aadf-406f-bebc-e114ee0f2f10" />


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


<img width="586" height="505" alt="1" src="https://github.com/user-attachments/assets/4c721cd4-cc21-4d81-b764-42cfaef367a7" />
<img width="629" height="529" alt="2" src="https://github.com/user-attachments/assets/77c68f99-82e5-45d5-8435-f3ff3f04ef2f" />
<img width="911" height="97" alt="3" src="https://github.com/user-attachments/assets/96097186-80f3-4af9-8cdb-3dc43dcd8f76" />


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


> LangGraph를 이용해 증상 추출, 운동 후보 추천, 최종 답변 생성을 분리한 멀티 에이전트 워크플로우를 구현한 프로젝트입니다.
