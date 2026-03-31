# ============================================================
#  p1_weekend3_memory_and_deploy_260328_김민석.py
#  프로젝트 1 - Weekend 3: 메모리와 프롬프트 엔지니어링
#  핵심 기술: Few-shot, CoT, Memory, Gradio
# ============================================================

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS

import gradio as gr

client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================
# 📦 주택청약 FAQ 샘플 데이터 (실습용)
# ============================================================
SAMPLE_FAQ_DATA = [
    {"id": "FAQ001", "category": "청약통장",
     "question": "주택청약종합저축이란 무엇인가요?",
     "answer": "주택청약종합저축은 국민주택과 민영주택 모두에 청약할 수 있는 만능 통장입니다.\n"
               "1) 매월 2만원~50만원 자유 납입\n"
               "2) 가입 후 일정 기간 경과 시 청약 자격 부여\n"
               "3) 2009년 5월 이후 모든 청약통장이 통합됨",
     "keywords": ["청약종합저축", "만능통장", "납입", "가입", "청약통장"], "difficulty": "easy"},

    {"id": "FAQ004", "category": "청약통장",
     "question": "청약통장 1순위 조건은 무엇인가요?",
     "answer": "1순위 조건은 주택 유형에 따라 다릅니다.\n"
               "1) 민영주택: 수도권 12개월, 비수도권 6개월 + 예치금\n"
               "2) 국민주택: 수도권 12개월(24회), 비수도권 6개월(12회)\n"
               "3) 투기과열지구: 2년, 24회 납입",
     "keywords": ["1순위", "가입기간", "예치금", "투기과열지구", "순위"], "difficulty": "medium"},

    {"id": "FAQ005", "category": "청약자격",
     "question": "주택 청약 신청 자격 조건은 무엇인가요?",
     "answer": "1) 만 19세 이상 (기혼자는 연령 제한 없음)\n"
               "2) 청약통장 가입 필수\n"
               "3) 국민주택: 무주택 세대구성원\n"
               "4) 민영주택: 세대주 또는 세대원 가능\n"
               "※ 투기과열지구는 세대주만 청약 가능",
     "keywords": ["청약자격", "만19세", "무주택", "세대주", "자격", "조건"], "difficulty": "easy"},

    {"id": "FAQ006", "category": "청약자격",
     "question": "무주택자 기준은 무엇인가요?",
     "answer": "본인과 세대원 모두 주택 미소유 시 무주택자입니다.\n"
               "예외: 60세 이상 직계존속 소유 주택, 20㎡ 이하 소형주택, 상속 후 3개월 내 처분 주택\n"
               "※ 분양권/입주권도 주택 수에 포함",
     "keywords": ["무주택", "세대원", "소형주택", "분양권", "무주택자"], "difficulty": "medium"},

    {"id": "FAQ009", "category": "특별공급",
     "question": "특별공급의 종류에는 어떤 것이 있나요?",
     "answer": "1) 기관추천 (국가유공자, 장애인 등)\n"
               "2) 다자녀가구 (3명 이상)\n"
               "3) 신혼부부 (혼인 7년 이내)\n"
               "4) 생애최초 (최초 주택 구입)\n"
               "5) 노부모부양 (만 65세 이상 부모)\n"
               "※ 2021년부터 신혼/생애최초 물량 확대",
     "keywords": ["특별공급", "기관추천", "다자녀", "신혼부부", "생애최초", "특공"], "difficulty": "medium"},

    {"id": "FAQ010", "category": "특별공급",
     "question": "신혼부부 특별공급 조건은 무엇인가요?",
     "answer": "1) 혼인기간 7년 이내 무주택 세대주\n"
               "2) 소득: 도시근로자 월평균소득 100~140%\n"
               "3) 전용면적 85㎡ 이하\n"
               "4) 혼인기간 짧을수록 + 자녀 많을수록 가점 높음\n"
               "5) 예비 신혼부부도 신청 가능",
     "keywords": ["신혼부부", "혼인기간", "소득기준", "가점", "특공", "특별공급"], "difficulty": "medium"},

    {"id": "FAQ013", "category": "일반공급",
     "question": "가점제와 추첨제의 차이는 무엇인가요?",
     "answer": "가점제: 무주택기간+부양가족+가입기간으로 점수화 (84점 만점)\n"
               "추첨제: 무작위 추첨\n"
               "1) 투기과열지구: 가점제 100%\n"
               "2) 청약과열지역: 가점 75% + 추첨 25%\n"
               "3) 기타: 가점 40% + 추첨 60%",
     "keywords": ["가점제", "추첨제", "84점", "투기과열지구", "가점", "추첨"], "difficulty": "medium"},

    {"id": "FAQ017", "category": "당첨/계약",
     "question": "당첨자 발표는 어떻게 확인하나요?",
     "answer": "1) 청약홈(www.applyhome.co.kr) 접속\n"
               "2) 당첨자 조회 메뉴 클릭\n"
               "3) 문자 알림 서비스 신청 가능\n"
               "※ 당첨 후 서류 제출 기간과 계약 일정 반드시 확인",
     "keywords": ["당첨자발표", "청약홈", "SMS알림", "서류제출", "당첨", "발표"], "difficulty": "easy"},

    {"id": "FAQ020", "category": "당첨/계약",
     "question": "재당첨 제한이란 무엇인가요?",
     "answer": "당첨 후 일정 기간 다른 주택 청약 불가:\n"
               "1) 투기과열지구: 10년\n"
               "2) 청약과열지역: 7년\n"
               "3) 수도권 공공주택: 5년\n"
               "※ 세대원 전원 적용 (배우자 당첨 시 본인도 제한)",
     "keywords": ["재당첨제한", "10년", "7년", "세대원", "재당첨"], "difficulty": "medium"},

    {"id": "FAQ023", "category": "기타",
     "question": "청약홈 사이트는 어떻게 이용하나요?",
     "answer": "청약홈(www.applyhome.co.kr) - 한국부동산원 운영\n"
               "1) 회원가입 후 공인인증서/간편인증 로그인\n"
               "2) 청약 신청, 당첨 확인, 가점 계산 가능\n"
               "3) 모바일 앱(청약홈)도 동일 서비스 제공",
     "keywords": ["청약홈", "공인인증서", "간편인증", "가점계산", "applyhome"], "difficulty": "easy"},
]

SAMPLE_TEST_QUERIES = [
    {"query": "청약통장 가입하려면 어떻게 해요?",    "expected_category": "청약통장", "expected_faq_id": "FAQ001"},
    {"query": "1순위 되려면 뭐가 필요해요?",         "expected_category": "청약통장", "expected_faq_id": "FAQ004"},
    {"query": "신혼부부 특공 자격이 궁금해요",        "expected_category": "특별공급", "expected_faq_id": "FAQ010"},
    {"query": "가점이 높으면 유리한가요?",            "expected_category": "일반공급", "expected_faq_id": "FAQ013"},
    {"query": "당첨되면 어떻게 확인해요?",            "expected_category": "당첨/계약", "expected_faq_id": "FAQ017"},
]

print(f"📦 FAQ 데이터 로드 완료: {len(SAMPLE_FAQ_DATA)}개 QA, {len(SAMPLE_TEST_QUERIES)}개 테스트 질의")

# ============================================================
# 🔧 Weekend 2 복원: 벡터 스토어 + RAG 파이프라인
# ============================================================

documents = [
    Document(
        page_content=f"질문: {f['question']}\n답변: {f['answer']}",
        metadata={"id": f["id"], "category": f["category"], "difficulty": f["difficulty"]}
    )
    for f in SAMPLE_FAQ_DATA
]

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n---\n".join([f"[{d.metadata.get('category','')}] {d.page_content}" for d in docs])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "주택청약 전문 상담원입니다. 참고 FAQ:\n{context}\n\nFAQ 기반으로 친절하게 답변하세요."),
    ("user", "{question}")
])

baseline_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

print(f"✅ RAG 파이프라인 복원 완료 ({len(documents)}개 문서)")

# ============================================================
# 사이클 1: Weekend 2 복원 + 기준선 측정
# ============================================================
# Weekend 2의 RAG 체인을 복원하고, `SAMPLE_TEST_QUERIES` 5개로 응답 시간과 답변 길이의 기준선을 측정하세요.

def cycle1_baseline_measurement():
    print("\n" + "=" * 60)
    print("사이클 1: Weekend 2 복원 + 기준선 측정")
    print("=" * 60)

    results = []
    for tq in SAMPLE_TEST_QUERIES:
        start = time.time()
        answer = baseline_chain.invoke(tq["query"])
        elapsed = round(time.time() - start, 2)
        results.append({"query": tq["query"], "time": elapsed, "length": len(answer)})
        print(f"❓ {tq['query']} → {elapsed}초, {len(answer)}자")

    avg_time = sum(r["time"] for r in results) / len(results)
    avg_len  = sum(r["length"] for r in results) / len(results)
    print(f"\n📊 기준선: 평균 {avg_time:.1f}초, 평균 {avg_len:.0f}자")

    return results

# ============================================================
# 사이클 2: Few-shot 프롬프트
# ============================================================
# FAQ 답변 예시 3개를 포함한 few-shot 프롬프트를 만들고, 기본 프롬프트 대비 답변 형식이 개선되는지 비교하세요.

fewshot_prompt = ChatPromptTemplate.from_messages([
    ("system", "주택청약 전문 상담원입니다. 참고 FAQ:\n{context}\n\n아래 예시처럼 답변하세요."),
    ("user",       "청약통장이 뭔가요?"),
    ("assistant",  "## 📋 청약통장 안내\n\n**청약종합저축**은 국민주택과 민영주택 모두에 청약할 수 있는 만능 통장입니다.\n\n### 핵심 정보\n1. 매월 2만~50만원 자유 납입\n2. 가입 후 일정 기간 경과 시 청약 자격 부여\n\n📞 추가 문의: 청약홈(1644-7445)"),
    ("user",       "특별공급 종류 알려주세요"),
    ("assistant",  "## 📋 특별공급 종류 안내\n\n특별공급은 **5가지 유형**이 있습니다.\n\n### 종류\n1. 기관추천 (국가유공자, 장애인)\n2. 다자녀가구 (3명 이상)\n3. 신혼부부 (혼인 7년 이내)\n4. 생애최초 (최초 주택 구입)\n5. 노부모부양 (만 65세 이상)\n\n📞 추가 문의: 청약홈(1644-7445)"),
    ("user",       "당첨 확인은 어떻게?"),
    ("assistant",  "## 📋 당첨 확인 방법\n\n### 확인 절차\n1. 청약홈(www.applyhome.co.kr) 접속\n2. 당첨자 조회 메뉴 클릭\n3. 문자 알림 서비스 신청 가능\n\n### ⚠️ 주의\n- 서류 제출 기간과 계약 일정 반드시 확인\n\n📞 추가 문의: 청약홈(1644-7445)"),
    ("user",       "{question}")
])

fewshot_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | fewshot_prompt | llm | StrOutputParser()
)

def cycle2_fewshot_prompt():
    print("\n" + "=" * 60)
    print("사이클 2: Few-shot 프롬프트")
    print("=" * 60)

    q = "가점제가 뭐예요?"
    print("📋 기본 프롬프트:")
    print(baseline_chain.invoke(q)[:200])
    print("\n📋 Few-shot 프롬프트:")
    print(fewshot_chain.invoke(q)[:200])

# ============================================================
# 사이클 3: Chain-of-Thought 프롬프트
# ============================================================
# "1단계-문제 파악, 2단계-원인 분석, 3단계-해결 방법" 사고 과정을 명시하는 CoT 프롬프트를 만들고, 복잡한 질문 3개로 테스트하세요.

cot_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "주택청약 전문 상담원입니다. 참고 FAQ:\n{context}\n\n"
     "복잡한 질문에는 다음 사고 과정을 따르세요:\n"
     "1단계 - 문제 파악: 사용자의 질문 핵심 파악\n"
     "2단계 - 관련 정보 정리: FAQ에서 관련 내용 추출\n"
     "3단계 - 해결 방법 제시: 단계별 안내\n\n"
     "각 단계를 명시적으로 표시하여 답변하세요."),
    ("user", "{question}")
])

cot_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | cot_prompt | llm | StrOutputParser()
)

def cycle3_cot_prompt():
    print("\n" + "=" * 60)
    print("사이클 3: Chain-of-Thought 프롬프트")
    print("=" * 60)

    complex_qs = [
        "청약통장 1순위인데 특별공급도 신청할 수 있나요?",
        "무주택인데 부모님이 집이 있으면 청약 가능한가요?",
        "가점이 낮은데 당첨 가능성을 높이려면 어떻게 해야 하나요?"
    ]
    for q in complex_qs:
        print(f"\n{'=' * 50}")
        print(f"❓ {q}")
        print(cot_chain.invoke(q)[:300])

# ============================================================
# 사이클 4: ConversationBufferMemory
# ============================================================
# `ConversationBufferMemory`와 `MessagesPlaceholder`를 사용해 이전 대화를 기억하는 챗봇을 만드세요. 3턴 이상의 연속 대화를 테스트하세요.
# `ConversationBufferMemory`가 deprecated된 `Modern Langchain`을 활용해서도 구현해보세요

memory_store: dict = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "주택청약 전문 상담원입니다. 참고 FAQ:\n{context}\n\n이전 대화를 참고하여 답변하세요."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])

memory_base_chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
    | memory_prompt | llm | StrOutputParser()
)

chain_with_memory = RunnableWithMessageHistory(
    memory_base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

def cycle4_conversation_memory():
    print("\n" + "=" * 60)
    print("사이클 4: 대화 메모리 (RunnableWithMessageHistory)")
    print("=" * 60)

    config = {"configurable": {"session_id": "cycle4_test"}}
    turns = ["청약통장이 뭔가요?", "1순위 요건은 어떻게 돼요?", "수도권은 몇 개월이면 돼요?"]

    for q in turns:
        answer = chain_with_memory.invoke({"question": q}, config=config)
        print(f"\n사용자: {q}")
        print(f"챗봇: {answer[:150]}...")

    msg_count = len(memory_store.get("cycle4_test", InMemoryChatMessageHistory()).messages)
    print(f"\n메모리: {msg_count}개 메시지 (자동 저장됨)")

# ============================================================
# 사이클 5: ConversationBufferWindowMemory
# ============================================================
# `ConversationBufferWindowMemory(k=3)`으로 최근 3턴만 기억하는 메모리를 만들고, 5턴 대화 후 초기 대화가 잊혀지는지 확인하세요.
# `ConversationBufferWindowMemory`가 deprecated된 `Modern Langchain`을 활용해서도 구현해보세요

WINDOW_K = 3
window_store: dict = {}

def get_window_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in window_store:
        window_store[session_id] = InMemoryChatMessageHistory()
    history = window_store[session_id]
    # 최근 K턴(= 2K개 메시지)만 유지
    while len(history.messages) > WINDOW_K * 2:
        history.messages.pop(0)
    return history

window_chain = RunnableWithMessageHistory(
    memory_base_chain,
    get_window_history,
    input_messages_key="question",
    history_messages_key="history",
)

def cycle5_window_memory():
    print("\n" + "=" * 60)
    print(f"사이클 5: 윈도우 메모리 (최근 {WINDOW_K}턴 유지)")
    print("=" * 60)

    config = {"configurable": {"session_id": "cycle5_test"}}
    turns = ["청약통장 가입 방법", "1순위 조건은?", "특별공급 종류는?", "가점제란?", "첫 번째 질문이 뭐였죠?"]

    for q in turns:
        answer = window_chain.invoke({"question": q}, config=config)
        h = window_store.get("cycle5_test", InMemoryChatMessageHistory()).messages
        print(f"\n사용자: {q}")
        print(f"챗봇: {answer[:120]}...")
        print(f"메모리: {len(h)}개 메시지 (최근 {len(h) // 2}턴)")

# ============================================================
# 사이클 6: RAG + Memory 통합 챗봇
# ============================================================
# RAG 검색 + 대화 메모리를 결합한 `FAQChatbotV3` 클래스를 만드세요. `ask(question)` 메서드와 `reset()` 메서드를 구현하고, 5턴 멀티턴 대화를 테스트하세요.

class FAQChatbotV3:
    def __init__(self, vs, model):
        self.retriever = vs.as_retriever(search_kwargs={"k": 3})
        self._store: dict = {}
        self._session = "default"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "주택청약 전문 상담원입니다.\n참고 FAQ:\n{context}\n\n이전 대화를 참고하여 답변하세요."),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{question}")
        ])

        base = (
            RunnablePassthrough.assign(context=lambda x: format_docs(self.retriever.invoke(x["question"])))
            | prompt | model | StrOutputParser()
        )

        self._chain = RunnableWithMessageHistory(
            base,
            self._get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def ask(self, question: str) -> dict:
        start = time.time()
        config = {"configurable": {"session_id": self._session}}
        answer = self._chain.invoke({"question": question}, config=config)
        sources = self.retriever.invoke(question)
        return {
            "answer":  answer,
            "time":    round(time.time() - start, 2),
            "sources": [{"id": d.metadata.get("id"), "category": d.metadata.get("category")} for d in sources],
        }

    def reset(self):
        self._store[self._session] = InMemoryChatMessageHistory()
        print("💬 대화 이력 초기화")


chatbot_v3 = FAQChatbotV3(vectorstore, llm)

def cycle6_chatbot_v3():
    print("\n" + "=" * 60)
    print("사이클 6: FAQChatbotV3 (RAG + Memory)")
    print("=" * 60)

    chatbot_v3.reset()
    turns = [
        "청약통장이 뭔지 알고 싶어요",
        "그럼 1순위 조건도 알려주세요",
        "투기과열지구는 조건이 달라요?",
        "특별공급에 대해서도 설명해주세요",
        "신혼부부 특공 자격이 어떻게 되나요?",
    ]
    for q in turns:
        r = chatbot_v3.ask(q)
        print(f"\n사용자: {q}")
        print(f"챗봇: {r['answer'][:150]}... ({r['time']}초)")

# ============================================================
# 사이클 7: 의도 분류기
# ============================================================
# 사용자 메시지를 greeting/question/complaint/chitchat으로 분류하는 체인을 만들고, 의도별로 다른 응답 전략을 적용하세요. 5가지 메시지로 테스트하세요.

intent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "사용자 메시지의 의도를 분류하세요.\n"
     "카테고리: greeting(인사), question(청약질문), complaint(불만), chitchat(잡담)\n"
     'JSON만 출력: {{"intent": "카테고리"}}'),
    ("user", "{message}")
])
intent_chain = intent_prompt | llm | StrOutputParser()

def handle_by_intent(message: str) -> dict:
    raw = intent_chain.invoke({"message": message})
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
    except Exception:
        result = {"intent": "question"}
    intent = result.get("intent", "question")

    if intent == "greeting":
        return {"intent": intent, "answer": "안녕하세요! 주택청약 상담입니다. 무엇을 도와드릴까요?"}
    elif intent == "chitchat":
        return {"intent": intent, "answer": "저는 주택청약 관련 질문에 답변드리는 챗봇입니다. 청약 관련 질문을 해주세요!"}
    elif intent == "complaint":
        r = chatbot_v3.ask(message)
        return {"intent": intent, "answer": f"불편을 드려 죄송합니다. 도와드리겠습니다.\n\n{r['answer']}"}
    else:
        r = chatbot_v3.ask(message)
        return {"intent": intent, "answer": r["answer"]}

def cycle7_intent_classifier():
    print("\n" + "=" * 60)
    print("사이클 7: 의도 분류기")
    print("=" * 60)

    test_msgs = [
        "안녕하세요~",
        "청약 1순위가 되려면 어떻게 해야 해요?",
        "청약 제도가 너무 복잡해서 짜증나네요",
        "요즘 날씨 어때요?",
        "특공 신청이 계속 실패해요!",
    ]
    for msg in test_msgs:
        r = handle_by_intent(msg)
        print(f"💬 '{msg}' → [{r['intent']}] {r['answer'][:80]}...")

# ============================================================
# 사이클 8: 답변 불가 처리
# ============================================================
# `similarity_search_with_score`의 거리 값으로 FAQ 범위 밖 질문을 감지하세요. 답변 가능/불가를 판별하는 `smart_answer(question)` 함수를 만들고, FAQ 관련/무관 질문 각 3개로 테스트하세요.

def smart_answer(question: str, threshold: float = 10.0) -> dict:
    results = vectorstore.similarity_search_with_score(question, k=1)
    if not results:
        return {"answer": "관련 FAQ를 찾지 못했습니다.", "answerable": False, "score": 0.0}

    doc, score = results[0]
    if score > threshold:
        return {
            "answer":    "해당 질문은 주택청약 FAQ 범위를 벗어납니다.\n📞 문의: 청약홈(1644-7445)",
            "answerable": False,
            "score":     score,
        }
    r = chatbot_v3.ask(question)
    return {"answer": r["answer"], "answerable": True, "score": score}

def cycle8_out_of_domain():
    print("\n" + "=" * 60)
    print("사이클 8: 답변 불가 처리 (유사도 기반)")
    print("=" * 60)

    test_qs = [
        "청약통장 1순위 조건",
        "신혼부부 특별공급",
        "당첨 확인 방법",
        "서울 맛집 추천해줘",
        "오늘 주식 시장 어때?",
        "영화 추천해줘",
    ]
    for q in test_qs:
        r = smart_answer(q)
        icon = "✅" if r["answerable"] else "❌"
        print(f"{icon} '{q}' (score:{r['score']:.3f}) → {r['answer'][:80]}...")

# ============================================================
# 사이클 9: Gradio 최종 UI
# ============================================================
# 의도 분류 + RAG + 메모리를 통합한 최종 `gr.ChatInterface`를 만드세요.

def cycle9_launch_gradio():
    print("\n" + "=" * 60)
    print("사이클 9: Gradio 최종 UI 실행")
    print("=" * 60)

    chatbot_ui = FAQChatbotV3(vectorstore, llm)

    def final_chat(message: str, history: list) -> str:
        if not message or not message.strip():
            return "질문을 입력해주세요!"
        # FAQ 범위 밖 질문 사전 차단
        results = vectorstore.similarity_search_with_score(message, k=1)
        if results and results[0][1] > 10.0:
            return "해당 질문은 주택청약 FAQ 범위를 벗어납니다.\n📞 문의: 청약홈(1644-7445)"
        r = handle_by_intent(message)
        return r["answer"]

    demo = gr.ChatInterface(
        fn          = final_chat,
        title       = "🏠 주택청약 FAQ 챗봇 v3 (최종)",
        description = "의도 분류 + RAG + 대화 메모리 통합 챗봇",
        examples    = [
            "안녕하세요!",
            "청약통장이 뭔가요?",
            "1순위 조건 알려주세요",
            "신혼부부 특별공급 자격은?",
            "당첨자 발표 확인 방법",
        ],
    )
    demo.launch(share=False, inbrowser=True)

# ============================================================
# 사이클 10: 최종 데모 & 프로젝트 회고
# ============================================================
# 5턴 멀티턴 대화 데모를 실행하고, 10개 질문으로 최종 벤치마크를 돌리세요. 3주간 구현한 기능 목록과 향후 개선점 3가지를 정리하세요.

def cycle10_final_demo_and_review():
    print("\n" + "=" * 60)
    print("사이클 10: 최종 데모 & 프로젝트 회고")
    print("=" * 60)

    # 멀티턴 대화 데모
    chatbot_v3.reset()
    print("💬 멀티턴 대화 데모 (5턴):")
    demo_turns = [
        "반갑습니다!",
        "청약통장에 대해 알려주세요",
        "1순위 조건이 궁금해요",
        "고마워요. 그럼 특별공급도 알려주세요",
        "신혼부부 특공 조건은요?",
    ]
    for q in demo_turns:
        r = handle_by_intent(q)
        print(f"\n사용자: {q}")
        print(f"챗봇 [{r['intent']}]: {r['answer'][:150]}...")

    # 벤치마크
    print("\n\n📊 최종 벤치마크 (10개 질문)")
    print("=" * 60)
    chatbot_v3.reset()
    bench_qs = [
        "청약통장 어떻게 가입해요",
        "1순위 되는 조건",
        "무주택 판단 기준",
        "신혼부부 특공 내용",
        "가점제란?",
        "청약 당첨 확인하기",
        "재당첨 제한 기간",
        "청약홈 이용 방법",
        "특별공급 유형",
        "생애최초 특공 조건",
    ]
    total_time = 0
    for i, q in enumerate(bench_qs, 1):
        r = chatbot_v3.ask(q)
        total_time += r["time"]
        cat = r["sources"][0]["category"] if r["sources"] else "N/A"
        print(f"[{i:2d}] {q} → {cat} ({r['time']}초)")

    print(f"\n📊 요약: 평균 {total_time / len(bench_qs):.1f}초, 총 {total_time:.1f}초")
    print("=" * 60)

    # 회고
    print("\n📋 3주 프로젝트 회고")
    print("  W1: OpenAI API + LCEL 체인 + 키워드 검색 + Gradio UI")
    print("  W2: Embeddings + FAISS 벡터 스토어 + RAG 체인 + 소스 표시")
    print("  W3: Few-shot + CoT + 대화 메모리 + 의도 분류 + 최종 UI")
    print("\n🚀 향후 개선점:")
    print("  1. 실제 청약 DB 연동 (대용량 FAQ)")
    print("  2. 사용자 피드백 기반 답변 개선")
    print("  3. 음성 인터페이스 추가")


if __name__ == "__main__":
    print("🏠 주택청약 FAQ 챗봇 v3 - 메모리 & 프롬프트 엔지니어링 실행")
    print(f"FAQ 데이터: {len(SAMPLE_FAQ_DATA)}개 | 테스트 질의: {len(SAMPLE_TEST_QUERIES)}개")

    cycle1_baseline_measurement()    # 사이클 1: 기준선 측정
    cycle2_fewshot_prompt()          # 사이클 2: Few-shot 프롬프트
    cycle3_cot_prompt()              # 사이클 3: CoT 프롬프트
    cycle4_conversation_memory()     # 사이클 4: 대화 메모리
    cycle5_window_memory()           # 사이클 5: 윈도우 메모리
    cycle6_chatbot_v3()              # 사이클 6: FAQChatbotV3
    cycle7_intent_classifier()       # 사이클 7: 의도 분류기
    cycle8_out_of_domain()           # 사이클 8: 답변 불가 처리
    cycle10_final_demo_and_review()  # 사이클 10: 최종 데모 & 회고

    cycle9_launch_gradio()           # 사이클 9: Gradio UI (마지막에 실행)