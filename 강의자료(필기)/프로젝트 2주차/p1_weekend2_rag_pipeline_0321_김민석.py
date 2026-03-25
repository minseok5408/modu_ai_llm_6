# ============================================================
#  p1_weekend2_rag_pipeline_0321_김민석.py
#  프로젝트 1 - Weekend 2: RAG 파이프라인
#  핵심 기술: Python, OpenAI API, LangChain LCEL, Gradio
# ============================================================

import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
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
# 사이클 1: Weekend 1 복원 + Document 객체
# ============================================================
# FAQ 데이터를 LangChain `Document` 객체 리스트로 변환하세요. `page_content`에 질문+답변, `metadata`에 id/category를 넣으세요.

def cycle1_build_documents():
    print("\n" + "=" * 60)
    print("사이클 1: Document 객체 변환")
    print("=" * 60)

    docs = []
    for faq in SAMPLE_FAQ_DATA:
        doc = Document(
            page_content=f"질문: {faq['question']}\n답변: {faq['answer']}",
            metadata={"id": faq["id"], "category": faq["category"], "difficulty": faq["difficulty"]}
        )
        docs.append(doc)

    print(f"📄 Document 객체: {len(docs)}개")
    print(f"\n샘플:")
    print(f"  page_content: {docs[0].page_content[:80]}...")
    print(f"  metadata: {docs[0].metadata}")

    return docs

# ============================================================
# 사이클 2: OpenAI Embeddings
# ============================================================
# `OpenAIEmbeddings`로 FAQ 텍스트들을 임베딩하고, 질문 간 코사인 유사도를 계산하세요. 의미적으로 유사한 질문이 높은 유사도를 보이는지 확인하세요.

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def cycle2_embeddings_test():
    print("\n" + "=" * 60)
    print("사이클 2: OpenAI Embeddings + 코사인 유사도")
    print("=" * 60)

    faq_texts = [f"질문: {f['question']}\n답변: {f['answer']}" for f in SAMPLE_FAQ_DATA[:5]]
    doc_vecs = embeddings.embed_documents(faq_texts)
    print(f"임베딩 차원: {len(doc_vecs[0])}")

    test_queries = ["청약통장 가입 방법", "신혼부부 특별공급", "오늘 날씨 어때?"]
    for query in test_queries:
        qv = embeddings.embed_query(query)
        sims = [(i, cosine_sim(qv, dv)) for i, dv in enumerate(doc_vecs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🔍 '{query}':")
        for idx, sim in sims[:3]:
            print(f"  [{sim:.4f}] {SAMPLE_FAQ_DATA[idx]['question'][:50]}")

# ============================================================
# 사이클 3: FAISS 벡터 스토어
# ============================================================
# `FAISS.from_documents()`로 벡터 스토어를 만들고, `similarity_search()`와 `similarity_search_with_score()`로 검색하세요. 저장/로드도 테스트하세요.

def cycle3_faiss_vectorstore(documents):
    print("\n" + "=" * 60)
    print("사이클 3: FAISS 벡터 스토어")
    print("=" * 60)

    vs = FAISS.from_documents(documents, embeddings)
    print(f"✅ FAISS 생성: {len(documents)}개 문서")

    # similarity_search
    results = vs.similarity_search("청약통장 가입", k=3)
    print("\n🔍 similarity_search:")
    for doc in results:
        print(f"  [{doc.metadata['category']}] {doc.page_content[:60]}...")

    # similarity_search_with_score
    results_s = vs.similarity_search_with_score("신혼부부 특공", k=3)
    print("\n🔍 with_score:")
    for doc, score in results_s:
        print(f"  [{score:.4f}] [{doc.metadata['category']}] {doc.page_content[:50]}...")

    # 저장/로드
    vs.save_local("/tmp/faiss_faq")
    loaded = FAISS.load_local("/tmp/faiss_faq", embeddings, allow_dangerous_deserialization=True)
    verify = loaded.similarity_search("당첨 확인", k=1)
    print(f"\n💾 저장/로드 확인: {verify[0].metadata['id']}")

    return vs

# ============================================================
# 사이클 4: Retriever 구성
# ============================================================
# 벡터 스토어에서 `as_retriever()`로 retriever를 만들고, `similarity` vs `mmr` 검색 타입과 `k=1,3,5` 결과를 비교하세요.

def cycle4_retriever(vectorstore):
    print("\n" + "=" * 60)
    print("사이클 4: Retriever 구성")
    print("=" * 60)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke("청약통장 1순위 조건")
    print("similarity (k=3):")
    for d in docs:
        print(f"  [{d.metadata['category']}] {d.page_content[:50]}...")

    mmr_ret = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
    mmr_docs = mmr_ret.invoke("청약통장 1순위 조건")
    print("\nMMR (k=3):")
    for d in mmr_docs:
        print(f"  [{d.metadata['category']}] {d.page_content[:50]}...")

    print("\nk값 비교:")
    for k in [1, 3, 5]:
        r = vectorstore.as_retriever(search_kwargs={"k": k})
        d = r.invoke("특별공급 종류")
        cats = [x.metadata["category"] for x in d]
        print(f"  k={k}: {cats}")

    return retriever

# ============================================================
# 사이클 5: RAG 체인
# ============================================================
# `retriever | format_docs`를 context로 사용하는 RAG 체인을 LCEL로 만들고, 질문 5개로 테스트하세요. Weekend 1의 키워드 검색 대비 답변 품질 차이를 확인하세요.

def format_docs(docs):
    return "\n---\n".join([f"[{d.metadata.get('category','')}] {d.page_content}" for d in docs])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "주택청약 전문 상담원입니다. 참고 FAQ:\n{context}\n\nFAQ 기반으로 친절하게 단계별 답변하세요."),
    ("user", "{question}")
])

def build_rag_chain(retriever):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    )

def cycle5_rag_chain(retriever):
    print("\n" + "=" * 60)
    print("사이클 5: RAG 체인")
    print("=" * 60)

    rag_chain = build_rag_chain(retriever)

    for tq in SAMPLE_TEST_QUERIES:
        answer = rag_chain.invoke(tq["query"])
        print(f"❓ {tq['query']}")
        print(f"💬 {answer[:120]}...")
        print("-" * 50)

    return rag_chain

# ============================================================
# 사이클 6: 검색 결과 검증
# ============================================================
# `SAMPLE_TEST_QUERIES` 5개로 RAG 체인이 올바른 FAQ를 찾는지 검증하세요. `similarity_search_with_score()`로 유사도 점수도 함께 확인하고, 기대한 FAQ ID가 검색 결과에 포함되는지 O/X로 판정하세요. 결과를 표로 정리하세요.

def cycle6_validate_retrieval(vectorstore):
    print("\n" + "=" * 60)
    print("사이클 6: 검색 결과 검증")
    print("=" * 60)

    print(f"{'질문':<30} {'기대':>8} {'검색 결과':<25} {'점수':>8} {'판정':>4}")
    print("-" * 80)

    for tq in SAMPLE_TEST_QUERIES:
        results = vectorstore.similarity_search_with_score(tq["query"], k=3)
        found_ids = [doc.metadata["id"] for doc, score in results]
        top_score = results[0][1]

        expected = tq["expected_faq_id"]
        hit = "O" if expected in found_ids else "X"

        print(f"{tq['query']:<30} {expected:>8} {str(found_ids):<25} {top_score:>8.4f} {hit:>4}")

    print(f"\n💡 점수는 L2 거리(낮을수록 유사). 점수가 높은 질문은 검색이 불안정할 수 있습니다.")

# ============================================================
# 사이클 7: 소스 문서 표시
# ============================================================
# RAG 답변에 참고한 FAQ 출처(ID, 카테고리)를 함께 반환하는 체인을 만드세요. `RunnableParallel`로 answer와 sources를 동시에 가져오세요.

def cycle7_rag_with_sources(rag_chain, retriever):
    print("\n" + "=" * 60)
    print("사이클 7: 소스 문서 표시")
    print("=" * 60)

    rag_with_sources = RunnableParallel(answer=rag_chain, sources=retriever)

    for q in ["청약통장 1순위 조건", "신혼부부 특별공급", "당첨 확인 방법"]:
        r = rag_with_sources.invoke(q)
        src = "\n".join([f"  - [{d.metadata['id']}] {d.metadata['category']}" for d in r["sources"]])
        print(f"❓ {q}")
        print(f"💬 {r['answer'][:120]}...")
        print(f"📎 참고 FAQ:\n{src}")
        print("-" * 50)

# ============================================================
# 사이클 8: FAQChatbotV2 클래스
# ============================================================
# vectorstore, retriever, rag_chain을 하나로 묶는 `FAQChatbotV2` 클래스를 만드세요. `ask(question)` 메서드가 answer, sources, time을 반환하도록 하세요.

class FAQChatbotV2:
    def __init__(self, vectorstore, llm):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt | llm | StrOutputParser()
        )

    def ask(self, question):
        start = time.time()
        answer = self.chain.invoke(question)
        sources = self.retriever.invoke(question)
        return {
            "answer": answer,
            "sources": [{"id": d.metadata.get("id"), "category": d.metadata.get("category")} for d in sources],
            "time": round(time.time() - start, 2)
        }

def cycle8_test_chatbot_v2(vectorstore):
    print("\n" + "=" * 60)
    print("사이클 8: FAQChatbotV2 클래스 테스트")
    print("=" * 60)

    chatbot = FAQChatbotV2(vectorstore, llm)
    for tq in SAMPLE_TEST_QUERIES:
        r = chatbot.ask(tq["query"])
        print(f"✅ {tq['query']} → {r['sources'][0]['category']} ({r['time']}초)")

    return chatbot

# ============================================================
# 사이클 9: Gradio UI v2
# ============================================================
# `gr.ChatInterface`로 RAG 기반 FAQ 챗봇 UI를 만드세요. 답변에 참고 FAQ 출처도 포함해서 표시하세요.

def cycle9_launch_gradio(chatbot):
    print("\n" + "=" * 60)
    print("사이클 9: Gradio UI v2 실행")
    print("=" * 60)

    def rag_chat(message, history):
        if not message or not message.strip():
            return "질문을 입력해주세요."
        if len(message.strip()) > 500:
            return f"❌ 질문이 너무 깁니다. ({len(message.strip())}자, 최대 500자)"
        try:
            r = chatbot.ask(message)
            src = "\n".join([f"  - [{s['id']}] {s['category']}" for s in r["sources"]])
            return f"{r['answer']}\n\n---\n📎 참고 FAQ:\n{src}\n⏱️ {r['time']}초"
        except Exception as e:
            return f"❌ 오류가 발생했습니다: {e}"

    demo = gr.ChatInterface(
        fn          = rag_chat,
        title       = "🏠 주택청약 FAQ 챗봇 v2 (RAG)",
        description = (
            "벡터 검색 기반 FAQ 챗봇입니다.\n"
            "주택청약에 관한 궁금한 점을 무엇이든 질문해보세요!"
        ),
        examples    = [
            "청약통장이 뭔가요?",
            "1순위 조건",
            "신혼부부 특공",
            "가점제란?",
            "당첨 확인",
        ],
    )
    demo.launch(share=False, inbrowser=True)

# ============================================================
# 사이클 10: 통합 테스트
# ============================================================
# 전체 RAG 파이프라인을 10개 질문으로 테스트하세요. 각 질문의 응답 시간, 검색된 카테고리, 답변 길이를 포함한 결과표를 출력하고, 전체 정확도와 평균 응답 시간을 요약하세요.

def cycle10_final_integration_test(chatbot):
    print("\n" + "=" * 60)
    print("사이클 10: 통합 테스트 (10개 질문)")
    print("=" * 60)

    test_questions = [
        "청약통장 가입하려면?",
        "1순위 조건이 뭐예요?",
        "무주택자 기준",
        "신혼부부 특별공급 자격",
        "가점제와 추첨제 차이",
        "당첨 확인 방법",
        "재당첨 제한 기간",
        "청약홈 사용법",
        "특별공급 종류",
        "생애최초 특별공급 조건",
    ]

    print("📊 Weekend 2 최종 테스트")
    print("=" * 60)
    total_time = 0
    for i, q in enumerate(test_questions, 1):
        r = chatbot.ask(q)
        total_time += r["time"]
        cat = r["sources"][0]["category"] if r["sources"] else "N/A"
        print(f"[{i:2d}] ❓ {q}")
        print(f"     💬 {r['answer'][:80]}...")
        print(f"     📂 {cat} | ⏱️ {r['time']}초 | 📎 {len(r['sources'])}개 FAQ")

    avg_time = total_time / len(test_questions)
    print(f"\n📊 요약: 평균 {avg_time:.1f}초, 총 {total_time:.1f}초")
    print("=" * 60)


if __name__ == "__main__":
    print("🏠 주택청약 FAQ 챗봇 v2 - RAG 파이프라인 실행")
    print(f"FAQ 데이터: {len(SAMPLE_FAQ_DATA)}개 | 테스트 질의: {len(SAMPLE_TEST_QUERIES)}개")


    documents = cycle1_build_documents()               # 사이클 1: Document 객체 변환
    cycle2_embeddings_test()                           # 사이클 2: OpenAI Embeddings + 코사인 유사도
    vectorstore = cycle3_faiss_vectorstore(documents)  # 사이클 3: FAISS 벡터 스토어 생성
    retriever = cycle4_retriever(vectorstore)          # 사이클 4: Retriever 구성
    rag_chain = cycle5_rag_chain(retriever)            # 사이클 5: RAG 체인
    cycle6_validate_retrieval(vectorstore)             # 사이클 6: 검색 결과 검증
    cycle7_rag_with_sources(rag_chain, retriever)      # 사이클 7: 소스 문서 표시
    chatbot = cycle8_test_chatbot_v2(vectorstore)      # 사이클 8: FAQChatbotV2 클래스
    cycle10_final_integration_test(chatbot)            # 사이클 10: 통합 테스트
    cycle9_launch_gradio(chatbot)                      # 사이클 9: Gradio UI v2 실행
