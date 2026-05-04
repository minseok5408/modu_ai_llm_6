# ============================================================
#  p3_weekend2_langgraph_agent_260502_김민석.py
#  프로젝트 3 - Weekend 2: LangGraph 기반 법률 에이전트 시스템
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("✅ 환경 설정 완료")

# ============================================================
# 실습 데이터: 법률 조문 샘플 (민법/형법/상법 발췌)
# ============================================================
SAMPLE_LAW_ARTICLES = [
    # --- 민법 ---
    {"law": "민법", "article": "제2조", "title": "신의성실의 원칙",
     "content": "권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다. 권리는 남용하지 못한다.",
     "topic": "기본원칙"},
    {"law": "민법", "article": "제103조", "title": "반사회질서의 법률행위",
     "content": "선량한 풍속 기타 사회질서에 위반한 사항을 내용으로 하는 법률행위는 무효로 한다.",
     "topic": "법률행위"},
    {"law": "민법", "article": "제109조", "title": "착오로 인한 의사표시",
     "content": "의사표시는 법률행위의 내용의 중요부분에 착오가 있는 때에는 취소할 수 있다. 그러나 그 착오가 표의자의 중대한 과실로 인한 때에는 취소하지 못한다.",
     "topic": "의사표시"},
    {"law": "민법", "article": "제390조", "title": "채무불이행과 손해배상",
     "content": "채무자가 채무의 내용에 좇은 이행을 하지 아니한 때에는 채권자는 손해배상을 청구할 수 있다. 그러나 채무자의 고의나 과실없이 이행할 수 없게 된 때에는 그러하지 아니하다.",
     "topic": "채권"},
    {"law": "민법", "article": "제750조", "title": "불법행위의 내용",
     "content": "고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
     "topic": "불법행위"},
    {"law": "민법", "article": "제840조", "title": "재판상 이혼원인",
     "content": "부부의 일방은 다음 각호의 사유가 있는 경우에는 가정법원에 이혼을 청구할 수 있다. 1. 배우자에 부정한 행위가 있었을 때 2. 배우자가 악의로 다른 일방을 유기한 때 3. 배우자 또는 그 직계존속으로부터 심히 부당한 대우를 받았을 때 4. 자기의 직계존속이 배우자로부터 심히 부당한 대우를 받았을 때 5. 배우자의 생사가 3년이상 분명하지 아니한 때 6. 기타 혼인을 계속하기 어려운 중대한 사유가 있을 때",
     "topic": "가족법"},
    # --- 형법 ---
    {"law": "형법", "article": "제250조", "title": "살인, 존속살해",
     "content": "사람을 살해한 자는 사형, 무기 또는 5년 이상의 징역에 처한다. 자기 또는 배우자의 직계존속을 살해한 자는 사형, 무기 또는 7년 이상의 징역에 처한다.",
     "topic": "생명"},
    {"law": "형법", "article": "제260조", "title": "폭행, 존속폭행",
     "content": "사람의 신체에 대하여 폭행을 가한 자는 2년 이하의 징역, 500만원 이하의 벌금, 구류 또는 과료에 처한다. 자기 또는 배우자의 직계존속에 대하여 폭행을 가한 자는 5년 이하의 징역 또는 700만원 이하의 벌금에 처한다.",
     "topic": "신체"},
    {"law": "형법", "article": "제329조", "title": "절도",
     "content": "타인의 재물을 절취한 자는 6년 이하의 징역 또는 1천만원 이하의 벌금에 처한다.",
     "topic": "재산"},
    {"law": "형법", "article": "제347조", "title": "사기",
     "content": "사람을 기망하여 재물의 교부를 받거나 재산상의 이익을 취득한 자는 10년 이하의 징역 또는 2천만원 이하의 벌금에 처한다. 전항의 방법으로 제삼자로 하여금 재물의 교부를 받게 하거나 재산상의 이익을 취득하게 한 때에도 전항의 형과 같다.",
     "topic": "재산"},
    {"law": "형법", "article": "제366조", "title": "재물손괴등",
     "content": "타인의 재물, 문서 또는 전자기록등 특수매체기록을 손괴 또는 은닉 기타 방법으로 기효용을 해한 자는 3년 이하의 징역 또는 700만원 이하의 벌금에 처한다.",
     "topic": "재산"},
    {"law": "형법", "article": "제307조", "title": "명예훼손",
     "content": "공연히 사실을 적시하여 사람의 명예를 훼손한 자는 2년 이하의 징역이나 금고 또는 500만원 이하의 벌금에 처한다. 공연히 허위의 사실을 적시하여 사람의 명예를 훼손한 자는 5년 이하의 징역, 10년 이하의 자격정지 또는 1천만원 이하의 벌금에 처한다.",
     "topic": "명예"},
    # --- 상법 ---
    {"law": "상법", "article": "제46조", "title": "기본적 상행위",
     "content": "영업으로 하는 다음의 행위를 상행위라 한다. 다만, 오로지 임금을 받을 목적으로 물건을 제조하거나 노무에 종사하는 자의 행위는 그러하지 아니하다. 1. 동산, 부동산, 유가증권 기타의 재산의 매매 2. 동산, 부동산, 유가증권 기타의 재산의 임대차 ... (22호)",
     "topic": "상행위"},
    {"law": "상법", "article": "제169조", "title": "회사의 의의",
     "content": "이 법에서 회사란 상행위나 그 밖의 영리를 목적으로 하여 설립한 법인을 말한다.",
     "topic": "회사법"},
    {"law": "상법", "article": "제329조", "title": "자본금의 구성",
     "content": "주식회사의 자본금은 이 법에서 달리 규정한 경우 외에는 발행주식의 액면총액으로 한다. 회사는 정관으로 정한 경우에는 주식의 전부를 무액면주식으로 발행할 수 있다. 다만, 무액면주식을 발행하는 경우에는 액면주식을 발행할 수 없다.",
     "topic": "회사법"},
    {"law": "상법", "article": "제382조", "title": "이사의 선임, 임기",
     "content": "이사는 주주총회에서 선임한다. 이사의 임기는 3년을 초과하지 못한다.",
     "topic": "회사법"},
    {"law": "상법", "article": "제399조", "title": "이사의 회사에 대한 책임",
     "content": "이사가 고의 또는 과실로 법령 또는 정관에 위반한 행위를 하거나 그 임무를 게을리한 때에는 그 이사는 회사에 대하여 연대하여 손해를 배상할 책임이 있다.",
     "topic": "회사법"},
    {"law": "상법", "article": "제732조", "title": "보험계약",
     "content": "보험계약은 당사자 일방이 약정한 보험료를 지급하고 상대방이 일정한 사유가 생길 경우에 일정한 보험금액 기타의 급여를 지급할 것을 약정함으로써 효력이 생긴다.",
     "topic": "보험"},
]

print(f"📚 법률 조문 로드 완료: 총 {len(SAMPLE_LAW_ARTICLES)}개")
laws = set(a["law"] for a in SAMPLE_LAW_ARTICLES)
for law in sorted(laws):
    cnt = sum(1 for a in SAMPLE_LAW_ARTICLES if a["law"] == law)
    print(f"  - {law}: {cnt}개")

# ============================================================
# ✅ 문제 1: Weekend 1 자산 복원 (법률 문서 + 벡터스토어 + search_law 도구)
# ============================================================
print("\n" + "=" * 60)
print("문제 1: Weekend 1 자산 복원")
print("=" * 60)

law_docs = [
    Document(
        page_content=f"{a['law']} {a['article']} ({a['title']}): {a['content']}",
        metadata={"law": a["law"], "article": a["article"], "title": a["title"]},
    )
    for a in SAMPLE_LAW_ARTICLES
]

vector_store = FAISS.from_documents(law_docs, embeddings)


@tool
def search_law(query: str, top_k: int = 3) -> str:
    """법률 조문을 의미 기반으로 검색합니다."""
    results = vector_store.similarity_search(query, k=top_k)
    out = [{"law": r.metadata["law"], "article": r.metadata["article"],
            "title": r.metadata["title"], "snippet": r.page_content[:120]}
           for r in results]
    return json.dumps(out, ensure_ascii=False, indent=2)


def verify_weekend1_assets():
    checks = {
        "SAMPLE_LAW_ARTICLES": len(SAMPLE_LAW_ARTICLES) >= 5,
        "law_docs": law_docs is not None and len(law_docs) >= 5,
        "vector_store": vector_store is not None,
        "search_law tool": hasattr(search_law, 'name') and search_law.name == "search_law",
    }
    for name, ok in checks.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    return all(checks.values())


verify_weekend1_assets()

# ============================================================
# ✅ 문제 2: LegalAgentState 정의
# ============================================================
print("\n" + "=" * 60)
print("문제 2: LegalAgentState 정의")
print("=" * 60)


class LegalAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    question_type: Literal["article_search", "case_search", "term_explain", "general"]
    search_results: list
    answer: str


print("✅ 필드:", list(LegalAgentState.__annotations__.keys()))

# ============================================================
# ✅ 문제 3: classify_node — 질문 유형 분류 노드
# ============================================================
print("\n" + "=" * 60)
print("문제 3: classify_node — 질문 유형 분류")
print("=" * 60)

VALID_TYPES = {"article_search", "case_search", "term_explain", "general"}


def classify_node(state):
    """질문을 4가지 유형 중 하나로 분류합니다."""
    query = state["query"]
    prompt = f"""다음 질문을 아래 4개 중 하나로 분류하세요. 답변은 영문 소문자 한 단어만.

article_search: 법률 조문 검색이 필요한 질문 (예: "이혼 사유는?", "명예훼손 처벌")
case_search: 판례 검색이 필요한 질문 (예: "~ 판례 알려줘", "~ 관련 사례")
term_explain: 법률 용어 정의를 묻는 질문 (예: "미필적 고의가 뭐야?")
general: 법률과 무관하거나 인사 (예: "안녕하세요")

질문: {query}
분류:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    label = response.content.strip().lower().split()[0].strip(".,")
    if label not in VALID_TYPES:
        label = "general"
    return {"question_type": label}


for q in ["이혼 사유는 뭐가 있어?", "명예훼손 관련 판례 알려줘", "미필적 고의가 뭐야?", "안녕하세요"]:
    state = LegalAgentState(messages=[], query=q, question_type="general", search_results=[], answer="")
    print(f"  '{q}' → {classify_node(state)['question_type']}")

# ============================================================
# ✅ 문제 4: search_node + analyze_node
# ============================================================
print("\n" + "=" * 60)
print("문제 4: search_node + analyze_node")
print("=" * 60)


def search_node(state):
    """벡터 검색으로 관련 조문을 찾습니다."""
    query = state["query"]
    docs = vector_store.similarity_search(query, k=3)
    results = [{
        "law": d.metadata.get("law"),
        "article": d.metadata.get("article"),
        "title": d.metadata.get("title"),
        "content": d.page_content,
    } for d in docs]
    return {"search_results": results}


def analyze_node(state):
    """검색 결과를 바탕으로 LLM이 답변을 생성합니다."""
    query = state["query"]
    results = state["search_results"]
    if not results:
        answer = "관련 조문을 찾지 못했습니다."
    else:
        context = "\n\n".join([f"- {r['law']} {r['article']}: {r['content']}" for r in results])
        prompt = f"""다음 법률 조문을 참고하여 질문에 답하세요.

질문: {query}

참고 조문:
{context}

답변 (3-5문장):"""
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
    return {"messages": [AIMessage(content=answer)], "answer": answer}


test_state = LegalAgentState(messages=[], query="이혼 사유", question_type="article_search",
                             search_results=[], answer="")
sr = search_node(test_state)
print(f"🔍 검색 결과 {len(sr['search_results'])}개")
for r in sr["search_results"][:2]:
    print(f"  - {r['law']} {r['article']}")

# ============================================================
# ✅ 문제 5: StateGraph 기본 구축 (classify → search → analyze)
# ============================================================
print("\n" + "=" * 60)
print("문제 5: StateGraph 기본 구축")
print("=" * 60)


def build_basic_graph():
    """classify → search → analyze 순서 그래프."""
    builder = StateGraph(LegalAgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("search", search_node)
    builder.add_node("analyze", analyze_node)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "search")
    builder.add_edge("search", "analyze")
    builder.add_edge("analyze", END)
    return builder.compile()


graph = build_basic_graph()
result = graph.invoke({
    "messages": [HumanMessage(content="명예훼손 처벌은?")],
    "query": "명예훼손 처벌은?",
    "question_type": "", "search_results": [], "answer": "",
})
print(f"🤖 유형: {result['question_type']}, 검색: {len(result['search_results'])}개")
print(f"💬 {result['answer'][:200]}")

# ============================================================
# ✅ 문제 6: 조건부 라우팅 — 질문 유형별 분기
# ============================================================
print("\n" + "=" * 60)
print("문제 6: 조건부 라우팅 그래프")
print("=" * 60)


def term_node(state):
    """용어 설명 전용 노드."""
    prompt = f"법률 용어 '{state['query']}'를 2-3문장으로 쉽게 설명하세요."
    answer = llm.invoke([HumanMessage(content=prompt)]).content
    return {"messages": [AIMessage(content=answer)], "answer": answer}


def general_node(state):
    """일반 질문 노드."""
    prompt = f"다음 질문에 간단히 답하세요: {state['query']}"
    answer = llm.invoke([HumanMessage(content=prompt)]).content
    return {"messages": [AIMessage(content=answer)], "answer": answer}


def route_by_type(state):
    """질문 유형에 따라 다음 노드를 결정합니다."""
    t = state["question_type"]
    if t in ("article_search", "case_search"):
        return "search"
    if t == "term_explain":
        return "term"
    return "general"


def build_routed_graph():
    """질문 유형별 조건부 라우팅 그래프."""
    builder = StateGraph(LegalAgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("search", search_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("term", term_node)
    builder.add_node("general", general_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", route_by_type,
        {"search": "search", "term": "term", "general": "general"}
    )
    builder.add_edge("search", "analyze")
    builder.add_edge("analyze", END)
    builder.add_edge("term", END)
    builder.add_edge("general", END)
    return builder.compile()


graph = build_routed_graph()
for q in ["이혼 사유는?", "미필적 고의가 뭐야?", "안녕하세요"]:
    r = graph.invoke({
        "messages": [HumanMessage(content=q)], "query": q,
        "question_type": "", "search_results": [], "answer": "",
    })
    print(f"  '{q}' [{r['question_type']}] → {r['answer'][:80]}...")

# ============================================================
# ✅ 문제 7: ReAct 패턴 — 도구 호출 루프
# ============================================================
print("\n" + "=" * 60)
print("문제 7: ReAct 패턴 — 도구 호출 루프")
print("=" * 60)

_react_tool_map = {}
_react_llm_with_tools = None


def agent_node(state):
    """LLM이 응답 또는 도구 호출을 결정합니다."""
    response = _react_llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state):
    """마지막 AIMessage의 tool_calls를 실행합니다."""
    last = state["messages"][-1]
    new_messages = []
    for tc in last.tool_calls:
        tool_fn = _react_tool_map.get(tc["name"])
        try:
            result = tool_fn.invoke(tc["args"]) if tool_fn else f"도구 '{tc['name']}' 없음"
        except Exception as e:
            result = f"도구 오류: {e}"
        new_messages.append(ToolMessage(content=str(result)[:3000], tool_call_id=tc["id"]))
    return {"messages": new_messages}


def should_continue(state):
    """도구 호출 여부로 계속 실행 또는 종료를 결정합니다."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_react_graph(tools):
    """ReAct 패턴 그래프를 빌드합니다."""
    global _react_tool_map, _react_llm_with_tools
    _react_tool_map = {t.name: t for t in tools}
    _react_llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(LegalAgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


graph = build_react_graph([search_law])
result = graph.invoke({
    "messages": [HumanMessage(content="사기죄 형량을 알려줘")],
    "query": "사기죄 형량을 알려줘",
    "question_type": "", "search_results": [], "answer": "",
})
print(f"📨 최종 메시지 수: {len(result['messages'])}")
print(f"🤖 {result['messages'][-1].content[:200]}")

# ============================================================
# ✅ 문제 8: MemorySaver — 대화 이력 영속화
# ============================================================
print("\n" + "=" * 60)
print("문제 8: MemorySaver — 대화 이력 영속화")
print("=" * 60)


def build_memory_graph(tools):
    """MemorySaver를 체크포인터로 사용하는 ReAct 그래프."""
    global _react_tool_map, _react_llm_with_tools
    _react_tool_map = {t.name: t for t in tools}
    _react_llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(LegalAgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=MemorySaver())


graph = build_memory_graph([search_law])
config = {"configurable": {"thread_id": "session-demo"}}

print("👤 Q1: 명예훼손 처벌은?")
r1 = graph.invoke({
    "messages": [HumanMessage(content="명예훼손 처벌은?")],
    "query": "명예훼손 처벌은?",
    "question_type": "", "search_results": [], "answer": "",
}, config=config)
print(f"🤖 {r1['messages'][-1].content[:150]}")

print("\n👤 Q2: 허위 사실이면 어떻게 달라져?")
r2 = graph.invoke({
    "messages": [HumanMessage(content="허위 사실이면 어떻게 달라져?")],
    "query": "허위 사실이면 어떻게 달라져?",
    "question_type": "", "search_results": [], "answer": "",
}, config=config)
print(f"🤖 {r2['messages'][-1].content[:150]}")

state = graph.get_state(config)
print(f"\n📝 누적 메시지 수: {len(state.values['messages'])}")

# ============================================================
# ✅ 문제 9: Human-in-the-Loop — 도구 실행 전 사용자 확인
# ============================================================
print("\n" + "=" * 60)
print("문제 9: Human-in-the-Loop (interrupt_before)")
print("=" * 60)


def build_hitl_graph(tools):
    """도구 실행 전 interrupt하는 Human-in-the-Loop 그래프."""
    global _react_tool_map, _react_llm_with_tools
    _react_tool_map = {t.name: t for t in tools}
    _react_llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(LegalAgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )


graph = build_hitl_graph([search_law])
config = {"configurable": {"thread_id": "hitl-demo"}}

print("=== 1단계: 도구 실행 전 정지 ===")
r1 = graph.invoke({
    "messages": [HumanMessage(content="사기죄 형량 알려줘")],
    "query": "사기죄 형량 알려줘",
    "question_type": "", "search_results": [], "answer": "",
}, config=config)
state = graph.get_state(config)
print(f"다음 실행 노드: {state.next}")
if state.values["messages"]:
    last_msg = state.values["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print(f"대기 중인 tool_calls: {[tc['name'] for tc in last_msg.tool_calls]}")

print("\n=== 2단계: 재개 (도구 실행 승인) ===")
r2 = graph.invoke(None, config=config)
print(f"🤖 최종: {r2['messages'][-1].content[:200]}")

# ============================================================
# ✅ 문제 10: 미니 프로젝트 — LegalGraphAgent 통합
# ============================================================
print("\n" + "=" * 60)
print("문제 10: LegalGraphAgent 통합")
print("=" * 60)


class LegalGraphAgent:
    """Weekend 2 최종 결과물 — LangGraph 기반 법률 에이전트."""

    DISCLAIMER = "\n\n⚠️ 본 답변은 일반 정보이며 법률 자문이 아닙니다."

    def __init__(self, tools):
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """ReAct + MemorySaver 그래프를 구축합니다."""
        global _react_tool_map, _react_llm_with_tools
        _react_tool_map = self.tool_map
        _react_llm_with_tools = self.llm_with_tools

        builder = StateGraph(LegalAgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=self.memory)

    def ask(self, query, thread_id="default"):
        """질문에 답하며 thread_id별 세션 이력을 유지합니다."""
        start = time.time()
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query, "question_type": "",
            "search_results": [], "answer": "",
        }, config=config)
        elapsed = round(time.time() - start, 2)
        answer = result["messages"][-1].content + self.DISCLAIMER
        return {
            "answer": answer, "thread_id": thread_id,
            "elapsed": elapsed, "messages_count": len(result["messages"]),
        }

    def visualize(self):
        """그래프 구조를 Mermaid 형식으로 반환합니다."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"시각화 실패: {e}"

    def get_history(self, thread_id="default"):
        """해당 세션의 메시지 이력을 반환합니다."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        return list(state.values.get("messages", []))


agent = LegalGraphAgent([search_law])

print("=== 그래프 구조 ===")
print(agent.visualize()[:300])

print("\n=== Thread A ===")
result_a = agent.ask("명예훼손 처벌은?", thread_id="user-A")
print(f"🤖 {result_a['answer'][:200]}")
print(f"⏱️ {result_a['elapsed']}초, 메시지 {result_a['messages_count']}개")

print("\n=== Thread B (다른 세션) ===")
result_b = agent.ask("사기죄는?", thread_id="user-B")
print(f"🤖 {result_b['answer'][:200]}")
print(f"⏱️ {result_b['elapsed']}초, 메시지 {result_b['messages_count']}개")

print(f"\n📝 A 이력: {len(agent.get_history('user-A'))}개, B 이력: {len(agent.get_history('user-B'))}개")

# ============================================================
# ✅ Weekend 2 완료 요약
# ============================================================
print("\n" + "=" * 60)
print("Weekend 2 완료!")
print("=" * 60)
summary = [
    ("문제 1",  "Weekend 1 자산 복원 — law_docs, vector_store, @tool search_law"),
    ("문제 2",  "LegalAgentState — TypedDict + Annotated[list, add_messages]"),
    ("문제 3",  "classify_node — LLM 기반 질문 유형 분류 (4가지 + fallback)"),
    ("문제 4",  "search_node + analyze_node — 벡터 검색 + LLM 답변 생성"),
    ("문제 5",  "build_basic_graph() — START → classify → search → analyze → END"),
    ("문제 6",  "build_routed_graph() — 조건부 라우팅 (article/case/term/general)"),
    ("문제 7",  "build_react_graph() — ReAct 패턴 (agent ⇄ tools 반복 루프)"),
    ("문제 8",  "build_memory_graph() — MemorySaver 체크포인터로 이력 영속화"),
    ("문제 9",  "build_hitl_graph() — interrupt_before=['tools'] Human-in-the-Loop"),
    ("문제 10", "LegalGraphAgent — ask / visualize / get_history 통합 클래스"),
]
for lab, desc in summary:
    print(f"  {lab}: {desc}")
