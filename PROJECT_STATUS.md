# GraphRAG DRIFT 프로젝트 현황 및 개발 계획

## 1. 목표

이 저장소의 목표는 Microsoft GraphRAG의 검색 전략인 **Local Search**, **Global Search**, **DRIFT Search**를
Neo4j 기반 Knowledge Graph와 LangGraph orchestration으로 재구성하는 것입니다.

최종 목표 아키텍처는 다음과 같습니다.

```text
User Query
   |
LangGraph Router
   |-------------------------------|
   |               |               |
 Local           Global           DRIFT
   |               |               |
   |          Community Reports    |
   |          Map -> Rank ->       |
   |             Reduce            |
   |                               |
   |                         Primer / Follow-up
   |                               |
   |                         Iterative Local
   |                               |
   |-------------------------------|
                   |
            neo4j-graphrag
                   |
                 Neo4j
```

## 2. 지금까지 조사한 결론

### Microsoft GraphRAG

Microsoft의 공식 GraphRAG는 Neo4j를 기본 backend로 사용하지 않습니다.
기본적으로 indexing 결과를 테이블 형태로 저장하고 vector store를 함께 사용하며,
query layer에서 Local / Global / DRIFT를 각각 별도 검색 엔진으로 구현합니다.

### Neo4j GraphRAG

Neo4j의 `neo4j-graphrag-python`은 다음과 같은 retrieval primitive를 제공합니다.

- Vector retrieval
- Hybrid retrieval
- Vector + Cypher graph traversal
- Text-to-Cypher
- Knowledge Graph pipeline

따라서 Local Search의 저수준 retrieval을 처음부터 다시 구현할 필요는 없습니다.

### LangGraph의 역할

LangGraph는 retrieval engine 자체보다는 다음 orchestration에 사용합니다.

- Local / Global / DRIFT routing
- Global map/reduce
- DRIFT iterative follow-up loop
- search state 관리
- 종료 조건 관리

## 3. 현재 구현 완료 범위

현재 MVP에서는 Neo4j나 외부 LLM 없이도 검색 전략 자체를 검증할 수 있도록
retrieval과 reasoning을 인터페이스로 분리했습니다.

### GraphRAGEngine

`src/graphrag_drift/core.py`

구현 기능:

- `local()`
- `global_search()`
- `drift_primer()`
- `drift_expand()`
- `drift_reduce()`
- `drift()`

### Retriever interface

`src/graphrag_drift/retrieval.py`

```python
class Retriever(Protocol):
    def local_search(...)
    def community_reports(...)
```

현재 테스트에서는 `InMemoryRetriever`를 사용합니다.

향후 이 인터페이스를 다음 Neo4j 구현으로 교체할 예정입니다.

```text
local_search()
    -> neo4j-graphrag
       VectorCypherRetriever
       or HybridCypherRetriever

community_reports()
    -> Neo4j :CommunityReport query
```

### Reasoner interface

`src/graphrag_drift/llm.py`

현재는 deterministic 테스트를 위한 `DeterministicReasoner`를 사용합니다.

향후 실제 LLM adapter를 추가할 수 있습니다.

예:

- OpenAI
- Amazon Bedrock
- Gemini
- OpenAI-compatible local model

### LangGraph workflow

`src/graphrag_drift/workflow.py`

현재 workflow:

```text
START
  |
 router
  |
  |------------------------|
  |            |           |
 local       global       drift_primer
  |            |           |
 END          END      drift_expand
                           |
                         loop
                           |
                      drift_reduce
                           |
                          END
```

DRIFT는 `max_depth`와 follow-up 존재 여부를 종료 조건으로 사용합니다.

## 4. 현재 검색 방식

### Local Search

현재 MVP:

```text
Query
  |
Retriever.local_search()
  |
Evidence
  |
Reasoner
  |
Answer
```

향후 Neo4j 버전:

```text
Query
  |
Vector / Hybrid Search
  |
Entity
  |
Cypher Traversal
  |-------------------------|
Relationship  Neighbor   TextUnit/Chunk
  |
Context
  |
LLM
```

### Global Search

현재 MVP:

```text
Query
  |
Community Reports
  |
Map each report
  |
score / rank
  |
Reduce
  |
Answer
```

현재 core 구현은 deterministic 검증을 위해 map 단계를 순차 실행합니다.

향후 LangGraph 구현에서는 community report별 map을 병렬화하고,
Microsoft GraphRAG와 유사하게 score filtering 및 token budget을 적용할 예정입니다.

### DRIFT Search

현재 MVP:

```text
Query
  |
Primer Local Search
  |
Generate Follow-up
  |
Local Search
  |
Add Evidence
  |
Generate Follow-up
  |
repeat until max_depth / no follow-up
  |
Reduce
```

즉 DRIFT를 별도 retriever가 아니라
**global/context planning + iterative local exploration 전략**으로 구현합니다.

## 5. 테스트 상태

현재 두 계층의 테스트가 있습니다.

### Core algorithm tests

`tests/test_core.py`

검증 항목:

- Local retrieval + answer
- Global report ranking + reduce
- DRIFT iterative loop / max depth

### LangGraph workflow tests

`tests/test_workflow.py`

검증 항목:

- Local route
- Global route
- DRIFT conditional loop

GitHub Actions 환경에서 실제 LangGraph를 설치한 후 실행한 결과:

```text
6 passed in 0.31s
```

Python 3.11 및 실제 `langgraph` dependency를 사용했습니다.

## 6. 다음 개발 단계

### Phase 1 - Neo4j adapter

가장 먼저 구현할 작업입니다.

추가 예정:

```text
src/graphrag_drift/neo4j_retriever.py
```

역할:

- `VectorCypherRetriever` 또는 `HybridCypherRetriever` 래핑
- Entity similarity search
- Relationship traversal
- 관련 Chunk / TextUnit 조회
- Community / CommunityReport 조회

목표는 기존 `Retriever` protocol을 유지하여
workflow 코드를 수정하지 않고 backend만 교체하는 것입니다.

### Phase 2 - Neo4j schema

예상 schema:

```text
(:Entity)-[:RELATED_TO]->(:Entity)

(:Chunk)-[:MENTIONS]->(:Entity)

(:Entity)-[:IN_COMMUNITY]->(:Community)

(:Community)-[:PARENT]->(:Community)

(:Community)-[:HAS_REPORT]->(:CommunityReport)
```

Entity에는 embedding/vector index를 구성합니다.

### Phase 3 - Indexing pipeline

문서에서 다음 정보를 생성합니다.

```text
Document
  |
Chunk
  |
Entity / Relationship extraction
  |
Neo4j
  |
Entity embedding
  |
Leiden Community Detection
  |
Community Report generation
```

가능하면 `neo4j-graphrag`의 KG pipeline을 사용하고,
community detection은 Neo4j GDS Leiden을 사용합니다.

### Phase 4 - Community Report

Microsoft GraphRAG의 Global Search를 위해 community-level summary를 생성합니다.

참고 구현:

- Microsoft GraphRAG
- neo4j-contrib/ms-graphrag-neo4j

저장 형태:

```text
(:Community)-[:HAS_REPORT]->(:CommunityReport)
```

### Phase 5 - Global Search 고도화

추가 기능:

- community report relevance selection
- 병렬 map
- structured key point output
- score filtering
- score descending ranking
- token budget
- final reduce

LangGraph의 parallel Send 패턴 사용을 검토합니다.

### Phase 6 - DRIFT 고도화

Microsoft GraphRAG의 DRIFT 구조를 참고하여 다음을 추가합니다.

- primer 단계
- action state
- follow-up priority/ranking
- 여러 follow-up 병렬 탐색
- evidence deduplication
- depth / budget 기반 종료
- final reduce

예상 State:

```python
class DriftState(TypedDict):
    query: str
    pending_actions: list
    evidence: list
    followups: list[str]
    depth: int
    max_depth: int
    answer: str
```

### Phase 7 - 실제 LLM adapter

`Reasoner` protocol 구현체를 추가합니다.

예:

```text
OpenAIReasoner
BedrockReasoner
GeminiReasoner
```

LLM provider와 retrieval logic을 분리합니다.

### Phase 8 - Integration / E2E tests

Docker 기반 테스트 환경:

```text
pytest
  |
Neo4j Docker
  |
sample documents
  |
KG indexing
  |
community detection
  |
community report
  |
Local / Global / DRIFT queries
```

CI에서는 외부 LLM 없이 실행 가능한 integration fixture를 우선 구성하고,
LLM-dependent E2E test는 별도 optional test로 분리합니다.

## 7. 개발 원칙

1. Microsoft GraphRAG를 그대로 dependency로 감싸기보다 검색 전략을 재현합니다.
2. Neo4j가 잘 제공하는 retrieval 기능은 재구현하지 않습니다.
3. retrieval과 orchestration을 분리합니다.
4. LangGraph는 search strategy/state orchestration에 집중합니다.
5. Neo4j는 graph/vector/community storage 및 traversal에 집중합니다.
6. 외부 LLM/DB 없이 core algorithm을 항상 deterministic test할 수 있게 유지합니다.

## 8. 최종 목표

최종 구조:

```text
                  LangGraph
                      |
                Search Router
        |-------------|-------------|
        |             |             |
      Local         Global         DRIFT
        |             |             |
        |       Map / Reduce     Iterative
        |                       Exploration
        |-------------|-------------|
                      |
              neo4j-graphrag
                      |
                   Neo4j
        |-------------|-------------|
        |             |             |
      Entity     Relationship    Community
        |                           |
      Chunk                  CommunityReport
```

이 구조를 통해 Microsoft GraphRAG의 검색 전략을 유지하면서
Neo4j의 실제 property graph, vector index, Cypher traversal을 직접 활용하는 것이 목표입니다.
