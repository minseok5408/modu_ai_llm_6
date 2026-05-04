# ============================================================
#  p3_weekend1_tool_calling_0425_김민석.py
#  프로젝트 3 - Weekend 1: 도구 호출 기반 법률 검색 기초
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
from dotenv import load_dotenv

load_dotenv()

# ⚠️ LangChain 패턴만 사용 — 원시 openai SDK 직접 import 금지
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

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
# ✅ 문제 1: 법률 데이터셋 구축 — Document 객체 변환
# ============================================================
print("\n" + "=" * 60)
print("문제 1: 법률 데이터셋 구축")
print("=" * 60)


def build_law_documents(articles):
    """법률 조문 dict 리스트를 Document 리스트로 변환합니다."""
    documents = []
    for article in articles:
        page_content = f"{article['law']} {article['article']} ({article['title']}): {article['content']}"
        metadata = {
            "law": article["law"],
            "article": article["article"],
            "title": article["title"],
            "topic": article["topic"],
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


law_docs = build_law_documents(SAMPLE_LAW_ARTICLES)
print(f"✅ Document {len(law_docs)}개 생성")
print(f"첫 문서 page_content: {law_docs[0].page_content[:80]}...")
print(f"첫 문서 metadata: {law_docs[0].metadata}")

# ============================================================
# ✅ 문제 2: FAISS 벡터스토어 구축
# ============================================================
print("\n" + "=" * 60)
print("문제 2: FAISS 벡터스토어 구축")
print("=" * 60)


def build_vector_store(docs):
    """Document 리스트로 FAISS 벡터스토어를 생성합니다."""
    return FAISS.from_documents(docs, embeddings)


vector_store = build_vector_store(law_docs)

query = "이혼 사유"
results = vector_store.similarity_search(query, k=3)
print(f"🔍 '{query}' 검색 결과 ({len(results)}개):")
for i, doc in enumerate(results, 1):
    meta = doc.metadata
    print(f"  [{i}] {meta['law']} {meta['article']} - {meta['title']}")

# ============================================================
# ✅ 문제 3: @tool search_law — 법률 검색 도구
# ============================================================
print("\n" + "=" * 60)
print("문제 3: @tool search_law")
print("=" * 60)


@tool
def search_law(query: str, top_k: int = 3, law_filter: str = "") -> str:
    """법률 조문을 의미 기반으로 검색합니다.
    질문과 관련된 법률 조문을 찾아 반환합니다.

    Args:
        query: 검색 질의 (예: "이혼 사유", "명예훼손 처벌")
        top_k: 반환할 조문 수 (기본 3)
        law_filter: 특정 법령으로 필터링 ("민법", "형법", "상법" 중 하나, 빈 문자열이면 전체)
    """
    raw_results = vector_store.similarity_search(query, k=top_k * 3)

    if law_filter:
        filtered = [d for d in raw_results if d.metadata.get("law") == law_filter]
    else:
        filtered = raw_results

    top = filtered[:top_k]
    output = []
    for d in top:
        m = d.metadata
        output.append({
            "law": m.get("law"),
            "article": m.get("article"),
            "title": m.get("title"),
            "snippet": d.page_content[:150],
        })
    return json.dumps(output, ensure_ascii=False, indent=2)


print("도구 이름:", search_law.name)
print("도구 설명:", search_law.description[:80], "...")
result = search_law.invoke({"query": "재산 범죄", "top_k": 3, "law_filter": "형법"})
print(f"\n📡 search_law('재산 범죄', law_filter='형법'):\n{result[:300]}")

# ============================================================
# ✅ 문제 4: @tool interpret_law — 조문 해석 도구
# ============================================================
print("\n" + "=" * 60)
print("문제 4: @tool interpret_law")
print("=" * 60)


@tool
def interpret_law(article_id: str) -> str:
    """법률 조문을 일반인이 이해할 수 있게 쉬운 말로 해석합니다.

    Args:
        article_id: 조문 식별자 (예: "민법 제750조", "형법 제329조")
    """
    parts = article_id.strip().split()
    if len(parts) < 2:
        return f"'{article_id}' 형식이 잘못되었습니다. 예: '민법 제750조'"

    law, art = parts[0], parts[1]
    match = next(
        (a for a in SAMPLE_LAW_ARTICLES if a["law"] == law and a["article"] == art),
        None,
    )
    if match is None:
        return f"'{article_id}'를 찾을 수 없습니다."

    prompt = (
        f"다음 법률 조문을 일반인이 이해할 수 있는 3-4문장으로 쉽게 설명해주세요.\n\n"
        f"{match['law']} {match['article']} ({match['title']})\n"
        f"{match['content']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


print(interpret_law.invoke({"article_id": "민법 제750조"})[:300])
print()
print(interpret_law.invoke({"article_id": "없는법 제1조"}))

# ============================================================
# ✅ 문제 5: @tool compare_articles — 조문 비교 도구
# ============================================================
print("\n" + "=" * 60)
print("문제 5: @tool compare_articles")
print("=" * 60)


@tool
def compare_articles(article_id1: str, article_id2: str) -> str:
    """두 법률 조문의 공통점과 차이점을 분석합니다.

    Args:
        article_id1: 첫 번째 조문 식별자 (예: "형법 제347조")
        article_id2: 두 번째 조문 식별자 (예: "형법 제329조")
    """
    def _find(aid):
        parts = aid.strip().split()
        if len(parts) < 2:
            return None
        law, art = parts[0], parts[1]
        return next((a for a in SAMPLE_LAW_ARTICLES if a["law"] == law and a["article"] == art), None)

    a1 = _find(article_id1)
    a2 = _find(article_id2)

    if not a1:
        return f"'{article_id1}'를 찾을 수 없습니다."
    if not a2:
        return f"'{article_id2}'를 찾을 수 없습니다."

    prompt = f"""다음 두 법률 조문을 비교 분석해주세요.

[조문 A]
{a1['law']} {a1['article']} ({a1['title']})
{a1['content']}

[조문 B]
{a2['law']} {a2['article']} ({a2['title']})
{a2['content']}

다음 형식으로 답변해주세요:
공통점: (2-3개)
차이점: (2-3개)
주의사항: (실무상 혼동하기 쉬운 지점)
"""
    return llm.invoke([HumanMessage(content=prompt)]).content


result = compare_articles.invoke({
    "article_id1": "형법 제329조",
    "article_id2": "형법 제347조"
})
print(result[:400])

# ============================================================
# ✅ 문제 6: 에이전트 루프 구현
# ============================================================
print("\n" + "=" * 60)
print("문제 6: 에이전트 루프")
print("=" * 60)


def run_legal_agent(query, tools, max_turns=3):
    """LLM이 도구를 자동 선택하며 실행하는 에이전트 루프."""
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content="당신은 법률 도우미입니다. 필요한 경우 도구를 사용하여 정확한 정보를 제공하세요."),
        HumanMessage(content=query)
    ]

    for turn in range(max_turns):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn is None:
                result = f"도구 '{tc['name']}'를 찾을 수 없습니다."
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"도구 오류: {e}"
            messages.append(ToolMessage(
                content=str(result)[:3000],
                tool_call_id=tc["id"]
            ))

    final = llm_with_tools.invoke(messages)
    return final.content


tools_list = [search_law, interpret_law, compare_articles]
answer = run_legal_agent("이혼 사유에는 어떤 것들이 있어?", tools_list)
print("🤖 답변:", answer[:300])

# ============================================================
# ✅ 문제 7: LegalChatbot 클래스 — 대화 이력 관리
# ============================================================
print("\n" + "=" * 60)
print("문제 7: LegalChatbot 클래스")
print("=" * 60)


class LegalChatbot:
    """법률 도메인 대화형 챗봇"""

    def __init__(self, tools):
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools)
        self.system_msg = SystemMessage(content=(
            "당신은 법률 도우미입니다. 법률 조문을 검색하고 해석하는 도구를 활용하여 "
            "정확하고 이해하기 쉬운 답변을 제공하세요."
        ))
        self.reset()

    def reset(self):
        """대화 이력 초기화 (SystemMessage만 유지)."""
        self.messages = [self.system_msg]

    def chat(self, user_message, max_turns=3):
        """사용자 메시지에 답하며 대화 이력을 유지."""
        self.messages.append(HumanMessage(content=user_message))

        for turn in range(max_turns):
            response = self.llm_with_tools.invoke(self.messages)
            self.messages.append(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                tool_fn = self.tool_map.get(tc["name"])
                try:
                    result = tool_fn.invoke(tc["args"]) if tool_fn else f"도구 '{tc['name']}' 없음"
                except Exception as e:
                    result = f"도구 오류: {e}"
                self.messages.append(ToolMessage(
                    content=str(result)[:3000], tool_call_id=tc["id"]
                ))

        final = self.llm_with_tools.invoke(self.messages)
        self.messages.append(final)
        return final.content


bot = LegalChatbot([search_law, interpret_law, compare_articles])
print("👤 Q1:", "명예훼손 처벌이 어떻게 돼?")
print("🤖:", bot.chat("명예훼손 처벌이 어떻게 돼?")[:200])
print()
print("👤 Q2:", "그럼 허위 사실이면 더 무거워?")
print("🤖:", bot.chat("그럼 허위 사실이면 더 무거워?")[:200])
print(f"\n📝 대화 이력 길이: {len(bot.messages)}")

# ============================================================
# ✅ 문제 8: 추가 도구 — search_cases + explain_term
# ============================================================
print("\n" + "=" * 60)
print("문제 8: 추가 도구 search_cases / explain_term")
print("=" * 60)

SAMPLE_CASES = [
    {"case_no": "2018다301350", "court": "대법원", "date": "2020-03-26",
     "keyword": ["이혼", "위자료"],
     "summary": "유책배우자의 이혼청구는 원칙적으로 허용되지 않으나 예외 사유에 해당하는 경우 허용될 수 있다는 판례."},
    {"case_no": "2019도14526", "court": "대법원", "date": "2020-07-16",
     "keyword": ["명예훼손", "온라인"],
     "summary": "SNS 게시글이 특정인을 지칭하지 않아도 맥락상 식별 가능하면 명예훼손이 성립한다는 판례."},
    {"case_no": "2017다208748", "court": "대법원", "date": "2019-11-14",
     "keyword": ["사기", "기망"],
     "summary": "부동산 매매에서 중요 사실을 고지하지 않은 경우 부작위에 의한 기망으로 사기죄 성립 가능."},
]

LEGAL_TERMS = {
    "선의의 제3자": "법률관계에서 어떤 사실을 알지 못한 상태에서 새로운 법률관계에 들어온 제3자. 특정 상황에서 보호받는다.",
    "채무불이행": "채무자가 자신의 의무를 이행하지 않거나 불완전하게 이행하는 것.",
    "불법행위": "고의 또는 과실로 타인에게 위법하게 손해를 가하는 행위.",
    "미필적 고의": "결과 발생을 확실히 의도한 것은 아니지만, 발생해도 무방하다고 용인하는 심리 상태.",
    "공소시효": "범죄 후 일정 기간이 지나면 공소를 제기할 수 없게 되는 제도.",
}


@tool
def search_cases(keyword: str) -> str:
    """판례를 키워드로 검색합니다. 예: 이혼, 명예훼손, 사기"""
    results = [c for c in SAMPLE_CASES if keyword in c["keyword"] or keyword in c["summary"]]
    if not results:
        return f"'{keyword}' 관련 판례 없음"
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def explain_term(term: str) -> str:
    """법률 용어의 정의를 설명합니다. 예: 선의의 제3자, 채무불이행"""
    if term in LEGAL_TERMS:
        return f"{term}: {LEGAL_TERMS[term]}"
    partial = [k for k in LEGAL_TERMS if term in k or k in term]
    if partial:
        return f"유사 용어: {', '.join(partial)}"
    return f"'{term}'에 대한 정의가 사전에 없습니다."


print("판례 검색:", search_cases.invoke({"keyword": "이혼"})[:200])
print()
print("용어 설명:", explain_term.invoke({"term": "미필적 고의"}))

# ============================================================
# ✅ 문제 9: 에러 처리 + 면책 조항
# ============================================================
print("\n" + "=" * 60)
print("문제 9: 에러 처리 + 면책 조항")
print("=" * 60)

LEGAL_DISCLAIMER = (
    "⚠️ 본 답변은 일반적인 정보 제공 목적이며, 구체적인 법률 자문이 아닙니다. "
    "정확한 판단은 전문 변호사와 상담하세요."
)


def safe_invoke(tool_fn, args):
    """도구 실행을 안전하게 감쌉니다. 에러 시 메시지 반환."""
    try:
        return tool_fn.invoke(args)
    except Exception as e:
        return f"⚠️ 도구 오류: {type(e).__name__}: {e}"


def add_disclaimer(answer):
    """답변에 면책 조항을 추가합니다."""
    return f"{answer}\n\n---\n{LEGAL_DISCLAIMER}"


print("정상 호출:", safe_invoke(search_law, {"query": "살인", "top_k": 1})[:100])
print()
print("에러 호출:", safe_invoke(search_law, {"wrong_key": "test"})[:100])
print()
print(add_disclaimer("이혼은 민법 제840조의 사유가 있을 때 재판상 청구할 수 있습니다."))

# ============================================================
# ✅ 문제 10: 미니 프로젝트 — LegalSearchAgent 통합
# ============================================================
print("\n" + "=" * 60)
print("문제 10: LegalSearchAgent 통합")
print("=" * 60)


class LegalSearchAgent:
    """모든 법률 도구를 통합한 에이전트 (Weekend 1 최종 결과물)."""

    def __init__(self):
        self.tools = [search_law, interpret_law, compare_articles, search_cases, explain_term]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.session_log = []
        self.system_msg = SystemMessage(content=(
            "당신은 법률 도우미입니다. 주어진 도구를 적극 활용하여 정확한 답변을 제공하세요."
        ))

    def ask(self, query, max_turns=3):
        """질문에 답하며 세션 로그를 기록합니다."""
        start = time.time()
        tools_used = []
        messages = [self.system_msg, HumanMessage(content=query)]

        for turn in range(max_turns):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                elapsed = round(time.time() - start, 2)
                final_answer = add_disclaimer(response.content)
                record = {
                    "query": query, "answer": final_answer,
                    "tools_used": tools_used,
                    "elapsed": elapsed, "turns": turn + 1,
                }
                self.session_log.append(record)
                return record

            for tc in response.tool_calls:
                tools_used.append(tc["name"])
                tool_fn = self.tool_map.get(tc["name"])
                result = safe_invoke(tool_fn, tc["args"]) if tool_fn else f"도구 '{tc['name']}' 없음"
                messages.append(ToolMessage(
                    content=str(result)[:3000], tool_call_id=tc["id"]
                ))

        final = self.llm_with_tools.invoke(messages)
        elapsed = round(time.time() - start, 2)
        final_answer = add_disclaimer(final.content)
        record = {
            "query": query, "answer": final_answer,
            "tools_used": tools_used,
            "elapsed": elapsed, "turns": max_turns,
        }
        self.session_log.append(record)
        return record

    def export_session(self):
        """세션 전체 요약을 반환합니다."""
        tool_freq = {}
        total_time = 0.0
        for log in self.session_log:
            total_time += log["elapsed"]
            for t in log["tools_used"]:
                tool_freq[t] = tool_freq.get(t, 0) + 1
        return {
            "total_queries": len(self.session_log),
            "tool_frequency": tool_freq,
            "total_time": round(total_time, 2),
        }


agent = LegalSearchAgent()
for q in [
    "이혼 사유에는 어떤 것들이 있어?",
    "명예훼손과 허위사실 명예훼손은 어떻게 달라?",
    "미필적 고의가 뭐야?",
]:
    print(f"👤 {q}")
    result = agent.ask(q)
    print(f"🔧 사용 도구: {result['tools_used']}")
    print(f"⏱️ {result['elapsed']}초")
    print(f"🤖 {result['answer'][:200]}...\n")

print("📊 세션 요약:", agent.export_session())

# ============================================================
# ✅ Weekend 1 완료 요약
# ============================================================
print("\n" + "=" * 60)
print("Weekend 1 완료!")
print("=" * 60)
summary = [
    ("문제 1", "법률 조문 18개 → Document 변환 (page_content + metadata)"),
    ("문제 2", "FAISS 벡터스토어 구축 + 유사도 검색"),
    ("문제 3", "@tool search_law — 의미 기반 법률 검색 + 법령 필터"),
    ("문제 4", "@tool interpret_law — LLM 기반 조문 해석"),
    ("문제 5", "@tool compare_articles — 두 조문 공통점/차이점 비교"),
    ("문제 6", "에이전트 루프 — bind_tools + ToolMessage 자동 실행"),
    ("문제 7", "LegalChatbot — 대화 이력 누적 + reset()"),
    ("문제 8", "@tool search_cases / explain_term — 판례 + 용어 사전"),
    ("문제 9", "safe_invoke + add_disclaimer — 에러 처리 + 면책 조항"),
    ("문제 10", "LegalSearchAgent — 5개 도구 통합 + 세션 로그"),
]
for lab, desc in summary:
    print(f"  {lab}: {desc}")
