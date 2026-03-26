# ============================================================
#  p1_weekend1_api_and_chain_0314_김민석.py
#  프로젝트 1 - Weekend 1: API와 체인 기반 FAQ 시스템 (Easy)
#  핵심 기술: Python, OpenAI API, LangChain LCEL, Gradio
# ============================================================

import os
import time
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import gradio as gr

client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
# 사이클 1: 첫 OpenAI API 호출
# ============================================================
# OpenAI API로 주택청약 관련 질문을 보내고 답변을 받아보세요. `system` 역할에 "주택청약 전문 상담원"을 설정하세요.

def cycle1_first_api_call():
    print("\n" + "=" * 60)
    print("사이클 1: 첫 OpenAI API 호출")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 주택청약 전문 상담원입니다. 친절하고 정확하게 답변해주세요."},
            {"role": "user",   "content": "주택청약종합저축이 뭔가요?"}
        ]
    )

    answer = response.choices[0].message.content
    print(f"답변:\n{answer}")
    return answer

# ============================================================
# 사이클 2: FAQ 데이터 탐색
# ============================================================
# `SAMPLE_FAQ_DATA`에서 카테고리별 FAQ 개수를 세고, `difficulty`가 `"easy"`인 항목만 필터링해서 출력하세요.

def cycle2_explore_faq():
    print("\n" + "=" * 60)
    print("사이클 2: FAQ 데이터 탐색")
    print("=" * 60)

    category_counts = Counter(item["category"] for item in SAMPLE_FAQ_DATA)
    print(f"전체 FAQ 수: {len(SAMPLE_FAQ_DATA)}개")
    print("\n카테고리별 FAQ 개수:")
    for cat, cnt in category_counts.items():
        print(f"  {cat}: {cnt}개")

    # difficulty == "easy"
    easy_faqs = [item for item in SAMPLE_FAQ_DATA if item["difficulty"] == "easy"]
    print(f"\ndifficulty='easy' 항목: {len(easy_faqs)}개")
    for item in easy_faqs:
        print(f"  [{item['id']}] [{item['category']}] {item['question']}")

    return category_counts, easy_faqs

# ============================================================
# 사이클 3: FAQ 검색 함수
# ============================================================
# 질문 문자열을 받아 키워드 매칭으로 관련 FAQ를 찾는 `search_faq(query, faq_data, top_k=3)` 함수를 만들고, `SAMPLE_TEST_QUERIES`로 테스트하세요.

def search_faq(query: str, faq_data: list, top_k: int = 3) -> list:
    results = []

    for item in faq_data:
        score = 0

        for keyword in item["keywords"]:
            if keyword in query:
                score += 2

        for word in query.split():
            if word in item["question"] and len(word) > 1:
                score += 1

        if score > 0:
            results.append({**item, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def cycle3_test_search():
    print("\n" + "=" * 60)
    print("사이클 3: FAQ 검색 함수 테스트")
    print("=" * 60)

    for test in SAMPLE_TEST_QUERIES:
        found = search_faq(test["query"], SAMPLE_FAQ_DATA, top_k=3)
        print(f"\n질문: {test['query']}")
        print(f"  기대 FAQ ID: {test['expected_faq_id']}")
        if found:
            top_match = found[0]
            hit = "✅" if top_match["id"] == test["expected_faq_id"] else "⚠️"
            print(f"  검색 결과: [{top_match['id']}] {top_match['question']} (score={top_match['score']}) {hit}")
        else:
            print("  검색 결과: 없음 ❌")

# ============================================================
# 사이클 4: 검색 결과 + LLM 답변 생성
# ============================================================
# 검색된 FAQ를 system prompt에 넣어 답변을 생성하는 `ask_faq(question, faq_data, client)` 함수를 만드세요. 답변과 함께 참고한 FAQ 목록도 반환하세요.

def ask_faq(question: str, faq_data: list, client: OpenAI):

    relevant_faqs = search_faq(question, faq_data, top_k=3)

    if not relevant_faqs:
        return (
            "죄송합니다. 관련 정보를 찾지 못했습니다. "
            "청약홈(www.applyhome.co.kr)에서 확인하거나 전문 상담원에게 문의해주세요.",
            []
        )

    context = "\n\n".join([
        f"Q: {faq['question']}\nA: {faq['answer']}"
        for faq in relevant_faqs
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 "당신은 주택청약 전문 상담원입니다. 아래 FAQ를 참고하여 친절하게 답변해주세요.\n"
                 "FAQ에 없는 내용은 '청약홈 또는 전문 상담원에게 문의'하도록 안내하세요.\n\n"
                 f"[참고 FAQ]\n{context}"
             )},
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    referenced_ids = [faq["id"] for faq in relevant_faqs]

    return answer, referenced_ids

def cycle4_test_ask_faq():
    print("\n" + "=" * 60)
    print("사이클 4: 검색 결과 + LLM 답변 생성 테스트")
    print("=" * 60)

    test_questions = ["청약통장 가입하려면 어떻게 해요?", "신혼부부 특공 자격이 궁금해요"]
    for q in test_questions:
        answer, refs = ask_faq(q, SAMPLE_FAQ_DATA, client)
        print(f"\n질문: {q}")
        print(f"참고 FAQ: {refs}")
        print(f"답변: {answer[:150]}...")

# ============================================================
# 사이클 5: ChatPromptTemplate 구성
# ============================================================
# `ChatPromptTemplate`으로 `{context}`와 `{question}` 변수를 사용하는 FAQ 답변용 프롬프트를 만들고, 카테고리 분류용 프롬프트도 하나 더 만들어서 각각 테스트하세요.

faq_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 주택청약 전문 상담원입니다.\n"
     "아래 FAQ 데이터를 참고하여 친절하고 정확하게 답변해주세요.\n"
     "FAQ 데이터에 없는 내용은 '청약홈(www.applyhome.co.kr) 또는 전문 상담원에게 문의해주세요'라고 안내하세요.\n\n"
     "[FAQ 데이터]\n{context}"),
    ("human", "{question}")
])

category_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "주택청약 관련 질문을 아래 카테고리 중 하나로 분류하세요. 카테고리명만 출력하세요.\n"
     "카테고리: 청약통장, 청약자격, 특별공급, 일반공급, 당첨/계약, 기타"),
    ("human", "{question}")
])

def cycle5_test_prompts():
    print("\n" + "=" * 60)
    print("사이클 5: PromptTemplate 테스트")
    print("=" * 60)

    parser = StrOutputParser()

    # FAQ 답변 프롬프트 테스트
    faq_chain = faq_prompt | llm | parser
    sample_context = "Q: 청약통장이란?\nA: 국민·민영 모두 청약 가능한 만능 통장"
    result = faq_chain.invoke({"question": "청약통장 가입 방법은?", "context": sample_context})
    print(f"[FAQ 프롬프트 결과]\n{result[:200]}")

    # 카테고리 분류 프롬프트 테스트
    category_chain = category_prompt | llm | parser
    test_qs = ["1순위 조건이 뭔가요?", "신혼부부 특공 자격은?", "당첨 확인 어떻게 해요?"]
    print("\n[카테고리 분류 결과]")
    for q in test_qs:
        cat = category_chain.invoke({"question": q})
        print(f"  '{q}' → {cat}")

# ============================================================
# 사이클 6: LCEL 체인
# ============================================================
# `prompt | llm | StrOutputParser()` 패턴으로 FAQ 답변 체인(`faq_chain`)을 만들고, 질문 2개로 테스트하세요. `.stream()`으로 스트리밍 출력도 해보세요.

parser = StrOutputParser()
faq_chain = faq_prompt | llm | parser

def build_context(question: str) -> str:
    relevant_faqs = search_faq(question, SAMPLE_FAQ_DATA, top_k=3)
    if not relevant_faqs:
        return "관련 FAQ 없음"
    return "\n\n".join([
        f"Q: {faq['question']}\nA: {faq['answer']}"
        for faq in relevant_faqs
    ])

def cycle6_chain_and_stream():
    print("\n" + "=" * 60)
    print("사이클 6: LCEL 체인 + 스트리밍")
    print("=" * 60)

    test_qs = ["청약통장 가입하려면 어떻게 해요?", "신혼부부 특공 자격이 궁금해요"]
    for q in test_qs:
        context = build_context(q)
        result = faq_chain.invoke({"question": q, "context": context})
        print(f"\nQ: {q}\nA: {result[:150]}...")

    stream_q = "1순위 되려면 뭐가 필요해요?"
    stream_context = build_context(stream_q)
    print(f"\n[스트리밍 출력] Q: {stream_q}")
    print("A: ", end="")
    for chunk in faq_chain.stream({"question": stream_q, "context": stream_context}):
        print(chunk, end="", flush=True)
    print()

# ============================================================
# 사이클 7: 검색
# ============================================================
# 질문을 넣으면 자동으로 FAQ 검색 → 답변 생성하는 `rag_chain`을 만드세요. `SAMPLE_TEST_QUERIES` 5개로 테스트하세요.

def retrieve_and_build_context(inputs: dict) -> dict:
    question = inputs["question"]
    context = build_context(question)  # 검색 + context 문자열 변환
    return {"question": question, "context": context}

rag_chain = RunnableLambda(retrieve_and_build_context) | faq_prompt | llm | parser

def cycle7_test_rag_chain():
    print("\n" + "=" * 60)
    print("사이클 7: rag_chain 테스트 (SAMPLE_TEST_QUERIES 5개)")
    print("=" * 60)

    for test in SAMPLE_TEST_QUERIES:
        result = rag_chain.invoke({"question": test["query"]})
        print(f"\nQ: {test['query']}")
        print(f"A: {result[:150]}...")

# ============================================================
# 사이클 8: 에러 처리
# ============================================================
# 빈 입력, 500자 초과, 숫자만 입력 등을 검증하고 `try/except`로 API 오류를 처리하는 `safe_ask(question, rag_chain)` 함수를 만드세요. 정상/에러 케이스 6가지 이상 테스트하세요.

def safe_ask(question: str, chain) -> str:

    # 1. 빈 입력 또는 공백만 있는 경우
    if not question or not question.strip():
        return "❌ 오류: 질문을 입력해주세요."

    # 2. 공백 제거 후 실제 내용 기준으로 검증
    question = question.strip()

    # 3. 너무 짧은 입력 (1글자): 의미 있는 질문이 될 수 없음
    if len(question) < 2:
        return "❌ 오류: 질문이 너무 짧습니다. 좀 더 구체적으로 입력해주세요."

    # 4. 너무 긴 입력: API 비용 및 프롬프트 토큰 한계 방지
    if len(question) > 500:
        return f"❌ 오류: 질문이 너무 깁니다. (현재 {len(question)}자, 최대 500자)"

    # 5. 숫자만 입력한 경우: 의미 있는 질문이 아님
    if question.isdigit():
        return "❌ 오류: 숫자만으로는 질문할 수 없습니다. 문장으로 질문해주세요."

    # 6. 특수문자로만 이루어진 경우 (알파벳/한글/숫자가 전혀 없음)
    if not any(c.isalnum() for c in question):
        return "❌ 오류: 유효한 질문을 입력해주세요."

    # API 오류 처리
    try:
        result = chain.invoke({"question": question})
        return result
    except Exception as e:
        # 네트워크 오류, API 키 만료, Rate Limit 등 모든 예외를 잡아 사용자에게 안내
        return f"❌ API 오류가 발생했습니다. 잠시 후 다시 시도해주세요. (상세: {str(e)})"

def cycle8_test_error_handling():
    print("\n" + "=" * 60)
    print("사이클 8: 에러 처리 테스트 (6가지 케이스)")
    print("=" * 60)

    test_cases = [
        ("",            "빈 입력"),
        ("   ",         "공백만 입력"),
        ("a",           "1글자 입력"),
        ("1234567890",  "숫자만 입력"),
        ("!@#$%^",      "특수문자만 입력"),
        ("x" * 501,     "501자 초과 입력"),
        # 정상 케이스
        ("청약통장이 뭔가요?", "정상 질문"),
        ("1순위 조건은?",      "정상 질문"),
    ]

    for question, case_name in test_cases:
        result = safe_ask(question, rag_chain)
        # 에러 메시지는 짧으므로 전체 출력, 정상 답변은 앞 60자만
        display = result if result.startswith("❌") else result[:60] + "..."
        print(f"  [{case_name}] → {display}")

# ============================================================
# 사이클 9: Gradio 채팅 UI
# ============================================================
# `gr.ChatInterface`로 지금까지 만든 RAG 체인을 웹 채팅 UI로 만드세요. 제목, 설명, 예시 질문 5개를 설정하세요.

def chat_response_for_gradio(message: str, history: list) -> str:
    return safe_ask(message, rag_chain)

def cycle9_launch_gradio():
    print("\n" + "=" * 60)
    print("사이클 9: Gradio 챗봇 UI 실행")
    print("=" * 60)

    demo = gr.ChatInterface(
        fn          = chat_response_for_gradio,
        title       = "🏠 주택청약 FAQ 챗봇",
        description = "주택청약에 관한 궁금한 점을 무엇이든 질문해보세요!",
        # examples: 사용자가 클릭 한 번으로 질문을 넣을 수 있는 예시 버튼
        examples    = [
            "청약통장 1순위 조건이 뭔가요?",
            "신혼부부 특별공급 자격은 어떻게 되나요?",
            "가점제와 추첨제 차이가 뭔가요?",
            "당첨자 발표는 어떻게 확인하나요?",
            "무주택자 기준을 알려주세요",
        ],
    )

    # inbrowser=True: 실행 즉시 기본 브라우저에서 자동으로 열림
    demo.launch(share=False, inbrowser=True)


# ============================================================
# 사이클 10: 최종 통합 테스트
# ============================================================
# 전체 파이프라인(입력 검증 → 검색 → 답변 생성)을 하나의 함수로 정리하고, 10개 질문으로 테스트하세요. 각 질문의 응답 시간, 참고 FAQ 수를 포함한 결과표를 출력하세요.

def cycle10_final_integration_test():
    print("\n" + "=" * 60)
    print("사이클 10: 최종 통합 테스트 (10개 질문)")
    print("=" * 60)

    # 10개 테스트 질문
    final_questions = [
        "청약통장이 뭔가요?",
        "1순위 조건이 뭔가요?",
        "청약 신청 자격이 뭔가요?",
        "무주택자 기준을 알려주세요",
        "특별공급 종류는 뭐가 있나요?",
        "신혼부부 특공 자격이 궁금해요",
        "가점제와 추첨제 차이가 뭔가요?",
        "당첨되면 어떻게 확인해요?",
        "재당첨 제한이 뭔가요?",
        "청약홈 사이트 이용방법 알려주세요",
    ]

    results = []

    for q in final_questions:

        start_time = time.time()
        relevant_faqs = search_faq(q, SAMPLE_FAQ_DATA, top_k=3)
        answer = safe_ask(q, rag_chain)
        elapsed = round(time.time() - start_time, 2)
        results.append({
            "질문":         q,
            "참고FAQ수":    len(relevant_faqs),
            "응답시간(초)": elapsed,
            "답변미리보기": answer[:40] + "..." if len(answer) > 40 else answer,
        })

    header = f"{'질문':<32} {'참고FAQ':>6} {'응답(초)':>8}  답변 미리보기"
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r['질문']:<32} "
            f"{r['참고FAQ수']:>6} "
            f"{r['응답시간(초)']:>8}  "
            f"{r['답변미리보기']}"
        )

    avg_time = round(sum(r["응답시간(초)"] for r in results) / len(results), 2)
    print(f"\n평균 응답시간: {avg_time}초")
    print("=" * 90)

    return results

# ============================================================
# 최종 Gradio UI - 통합 버전
# ============================================================
# Gradio UI도 최종 버전으로 만드세요.

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def build_context_with_ids(question: str):

    relevant_faqs = search_faq(question, SAMPLE_FAQ_DATA, top_k=3)
    if not relevant_faqs:
        return "관련 FAQ 없음", []
    context = "\n\n".join([
        f"Q: {faq['question']}\nA: {faq['answer']}"
        for faq in relevant_faqs
    ])
    return context, [faq["id"] for faq in relevant_faqs]


def final_chat_fn(message: str, history: list) -> str:

    if not message or not message.strip():
        return "❌ 질문을 입력해주세요."
    message = message.strip()
    if len(message) > 500:
        return f"❌ 질문이 너무 깁니다. ({len(message)}자, 최대 500자)"

    context, ref_ids = build_context_with_ids(message)

    messages = [
        SystemMessage(content=(
            "당신은 주택청약 전문 상담원입니다.\n"
            "아래 FAQ 데이터를 참고하여 친절하고 정확하게 답변해주세요.\n"
            "FAQ에 없는 내용은 청약홈(www.applyhome.co.kr) 문의를 안내하세요.\n\n"
            f"[FAQ 데이터]\n{context}"
        ))
    ]

    for entry in history:
        if isinstance(entry, dict):
            role    = entry.get("role", "")
            content = entry.get("content", "") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            human_msg, ai_msg = entry
            if human_msg:
                messages.append(HumanMessage(content=str(human_msg)))
            if ai_msg:
                messages.append(AIMessage(content=str(ai_msg)))

    messages.append(HumanMessage(content=message))

    try:
        response = llm.invoke(messages)
        answer = response.content

        if ref_ids:
            answer += f"\n\n📎 참고 FAQ: {', '.join(ref_ids)}"

        return answer

    except Exception as e:
        return f"❌ 오류가 발생했습니다: {str(e)}"


def launch_final_demo():
    demo = gr.ChatInterface(
        fn          = final_chat_fn,
        title       = "🏠 주택청약 FAQ 챗봇",
        description = (
            "주택청약에 관한 궁금한 점을 무엇이든 질문해보세요!\n"
            "청약통장, 청약자격, 특별공급, 일반공급, 당첨/계약 등 다양한 주제를 다룹니다."
        ),
        examples    = [
            "청약통장 1순위 조건이 뭔가요?",
            "신혼부부 특별공급 자격은 어떻게 되나요?",
            "가점제와 추첨제 차이가 뭔가요?",
            "당첨자 발표는 어떻게 확인하나요?",
            "무주택자 기준을 알려주세요",
        ],
    )
    demo.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    print("🏠 주택청약 FAQ 챗봇 - 전체 파이프라인 실행")
    print(f"FAQ 데이터: {len(SAMPLE_FAQ_DATA)}개 | 테스트 질의: {len(SAMPLE_TEST_QUERIES)}개")

    cycle1_first_api_call()           # 사이클 1: 첫 API 호출
    cycle2_explore_faq()              # 사이클 2: FAQ 데이터 탐색
    cycle3_test_search()              # 사이클 3: FAQ 검색 함수
    cycle4_test_ask_faq()             # 사이클 4: 검색결과 + LLM 답변 생성
    cycle5_test_prompts()             # 사이클 5: PromptTemplate
    cycle6_chain_and_stream()         # 사이클 6: LCEL 체인
    cycle7_test_rag_chain()           # 사이클 7: 검색
    cycle8_test_error_handling()      # 사이클 8: 에러 처리
    cycle10_final_integration_test()  # 사이클 10: 최종 통합 테스트

    launch_final_demo()
