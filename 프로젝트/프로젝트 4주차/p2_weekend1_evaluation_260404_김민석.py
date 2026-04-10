# ============================================================
#  p2_weekend1_evaluation_260404_김민석.py
#  프로젝트 2 - Weekend 1: RAG 평가 프레임워크와 하이브리드 검색
#  핵심 기술: Hit Rate, MRR, NDCG, BM25, Hybrid Search, Query Expansion
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever  # noqa

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

load_dotenv()

client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("✅ 환경 설정 완료")

# ============================================================
# 데이터 준비
# ============================================================
SAMPLE_ETF_DATA = [
    {"ticker": "KODEX200", "name": "KODEX 200", "category": "국내주식",
     "description": "KOSPI 200 지수를 추종하는 국내 대표 ETF. 삼성전자, SK하이닉스 등 대형주 중심으로 구성되어 있으며, 국내 주식시장 전체의 흐름을 반영합니다.",
     "expense_ratio": 0.15, "aum_billion": 58000, "risk_level": "중간",
     "returns": {"1m": 2.1, "3m": 5.4, "1y": 12.3, "3y": 28.5},
     "keywords": ["코스피", "대형주", "인덱스", "패시브"]},
    {"ticker": "TIGER미국S&P500", "name": "TIGER 미국 S&P500", "category": "해외주식",
     "description": "미국 S&P500 지수를 추종. 애플, 마이크로소프트, 아마존 등 미국 대형 기술주 포함. 환헤지 미적용으로 원/달러 환율 변동에 노출됩니다.",
     "expense_ratio": 0.07, "aum_billion": 42000, "risk_level": "중간",
     "returns": {"1m": 3.2, "3m": 8.1, "1y": 18.7, "3y": 45.2},
     "keywords": ["미국", "S&P500", "대형주", "기술주"]},
    {"ticker": "KODEX미국나스닥100", "name": "KODEX 미국나스닥100", "category": "해외주식",
     "description": "나스닥100 지수 추종. 기술 성장주 중심으로 애플, 엔비디아, 메타 등 포함. 고성장/고변동성 특성으로 공격적 투자자에게 적합합니다.",
     "expense_ratio": 0.09, "aum_billion": 35000, "risk_level": "높음",
     "returns": {"1m": 4.5, "3m": 12.3, "1y": 25.1, "3y": 62.8},
     "keywords": ["나스닥", "기술주", "성장주", "AI"]},
    {"ticker": "KODEX국고채10년", "name": "KODEX 국고채 10년", "category": "채권",
     "description": "대한민국 10년 만기 국고채를 추종하는 채권 ETF. 금리 인하 시 가격 상승, 안전자산 선호 시 수요 증가. 변동성이 낮아 안정적 투자에 적합합니다.",
     "expense_ratio": 0.05, "aum_billion": 12000, "risk_level": "낮음",
     "returns": {"1m": 0.8, "3m": 1.5, "1y": 4.2, "3y": 8.1},
     "keywords": ["국채", "채권", "안전자산", "금리"]},
    {"ticker": "TIGER금은혼합", "name": "TIGER 금은혼합", "category": "원자재",
     "description": "금과 은에 분산 투자하는 원자재 ETF. 인플레이션 헤지 수단으로 활용되며, 지정학적 리스크 시 안전자산으로 수요 증가합니다.",
     "expense_ratio": 0.39, "aum_billion": 3500, "risk_level": "중간",
     "returns": {"1m": 1.2, "3m": 3.8, "1y": 15.6, "3y": 35.2},
     "keywords": ["금", "은", "원자재", "인플레이션"]},
    {"ticker": "KODEX리츠", "name": "KODEX 한국부동산리츠인프라", "category": "부동산",
     "description": "국내 상장 리츠 및 인프라 펀드에 투자. 임대료 수입 기반의 안정적 배당 수익을 제공하며, 부동산 간접투자 수단으로 활용됩니다.",
     "expense_ratio": 0.09, "aum_billion": 4800, "risk_level": "중간",
     "returns": {"1m": -0.5, "3m": 2.1, "1y": 7.8, "3y": 15.3},
     "keywords": ["리츠", "부동산", "배당", "임대"]},
    {"ticker": "KODEX2차전지", "name": "KODEX 2차전지산업", "category": "테마",
     "description": "2차전지(배터리) 관련 기업에 집중 투자. LG에너지솔루션, 삼성SDI, 포스코퓨처엠 등 포함. 전기차 시장 성장에 따른 수혜가 기대됩니다.",
     "expense_ratio": 0.45, "aum_billion": 18000, "risk_level": "높음",
     "returns": {"1m": -2.3, "3m": -5.1, "1y": -12.4, "3y": 8.7},
     "keywords": ["2차전지", "배터리", "전기차", "테마"]},
    {"ticker": "TIGERBBD", "name": "TIGER 미국배당다우존스", "category": "배당",
     "description": "미국 고배당 우량주에 투자하는 ETF. 안정적인 배당 수익과 자본 이득을 동시에 추구합니다. 월배당 지급으로 현금흐름 관리에 유리합니다.",
     "expense_ratio": 0.01, "aum_billion": 52000, "risk_level": "낮음",
     "returns": {"1m": 1.8, "3m": 4.2, "1y": 10.5, "3y": 32.1},
     "keywords": ["배당", "미국", "월배당", "인컴"]},
    {"ticker": "KODEXKSM", "name": "KODEX 코스닥150", "category": "국내주식",
     "description": "코스닥 150 지수를 추종. 중소형 성장주 중심으로 바이오, IT, 게임 등 혁신 기업 포함. 코스피 대비 높은 변동성과 성장 잠재력을 가집니다.",
     "expense_ratio": 0.20, "aum_billion": 8500, "risk_level": "높음",
     "returns": {"1m": -1.2, "3m": 3.5, "1y": 8.9, "3y": 18.7},
     "keywords": ["코스닥", "중소형", "성장주", "바이오"]},
    {"ticker": "KOSEF단기자금", "name": "KOSEF 단기자금", "category": "머니마켓",
     "description": "초단기 채권 및 예금에 투자하는 MMF형 ETF. 원금 손실 위험이 극히 낮으며, 여유 자금 파킹 용도로 활용됩니다. 하루 단위 이자 발생.",
     "expense_ratio": 0.03, "aum_billion": 25000, "risk_level": "매우낮음",
     "returns": {"1m": 0.3, "3m": 0.9, "1y": 3.5, "3y": 10.2},
     "keywords": ["단기", "파킹", "안전", "예금"]},
]

SAMPLE_USER_PROFILES = [
    {"user_id": "U001", "name": "김초보", "level": "초보",
     "risk_tolerance": "낮음", "investment_goal": "안정적 수익",
     "monthly_budget": 500000, "preferred_categories": ["채권", "머니마켓"],
     "sample_queries": ["안전한 투자 상품 추천해주세요", "원금 손실 없는 ETF가 뭐가 있나요?"]},
    {"user_id": "U002", "name": "이중급", "level": "중급",
     "risk_tolerance": "중간", "investment_goal": "자산 증식",
     "monthly_budget": 2000000, "preferred_categories": ["국내주식", "해외주식"],
     "sample_queries": ["S&P500 추종 ETF 비교해주세요", "배당과 성장 균형 잡힌 포트폴리오 추천"]},
    {"user_id": "U003", "name": "박전문", "level": "전문",
     "risk_tolerance": "높음", "investment_goal": "공격적 수익",
     "monthly_budget": 10000000, "preferred_categories": ["테마", "해외주식"],
     "sample_queries": ["AI 관련 ETF 섹터 분석해줘", "나스닥100 vs 코스닥150 변동성 비교"]},
]

SAMPLE_EVAL_QUERIES = [
    {"query": "초보자인데 안전한 투자 추천해주세요", "expected_tickers": ["KODEX국고채10년", "KOSEF단기자금", "TIGERBBD"]},
    {"query": "미국 기술주에 투자하고 싶어요", "expected_tickers": ["TIGER미국S&P500", "KODEX미국나스닥100"]},
    {"query": "월배당 받을 수 있는 ETF 있나요?", "expected_tickers": ["TIGERBBD", "KODEX리츠"]},
    {"query": "전기차 관련 투자 상품 알려주세요", "expected_tickers": ["KODEX2차전지"]},
    {"query": "인플레이션 헤지용 상품 추천", "expected_tickers": ["TIGER금은혼합", "KODEX리츠"]},
    {"query": "분산투자 포트폴리오 짜주세요", "expected_tickers": ["KODEX200", "TIGER미국S&P500", "KODEX국고채10년"]},
]

print(f"📦 ETF 데이터 로드 완료: {len(SAMPLE_ETF_DATA)}개 상품, {len(SAMPLE_USER_PROFILES)}개 프로필, {len(SAMPLE_EVAL_QUERIES)}개 평가 질의")
for cat in sorted(set(e["category"] for e in SAMPLE_ETF_DATA)):
    cnt = sum(1 for e in SAMPLE_ETF_DATA if e["category"] == cat)
    print(f"  - {cat}: {cnt}개")

# ============================================================
# 공통 설정: 디렉토리, 벡터 스토어, 평가 데이터, BM25, 검색 함수
# ============================================================
os.makedirs("project2_data/raw", exist_ok=True)
os.makedirs("project2_data/evaluation", exist_ok=True)
os.makedirs("project2_data/checkpoints", exist_ok=True)

# 벡터 스토어 구축
etf_documents = []
documents = []
for doc_id, etf in enumerate(SAMPLE_ETF_DATA):
    text = (
        f"ETF명: {etf['name']}\n"
        f"카테고리: {etf['category']}\n"
        f"설명: {etf['description']}\n"
        f"위험도: {etf['risk_level']}\n"
        f"키워드: {', '.join(etf['keywords'])}"
    )
    etf_documents.append({
        "name": etf["name"],
        "category": etf["category"],
        "market": "국내" if etf["category"] in ["국내주식", "채권", "머니마켓", "부동산"] else "글로벌",
        "content": etf["description"],
    })
    documents.append(Document(
        page_content=text,
        metadata={"doc_id": doc_id, "name": etf["name"], "category": etf["category"], "ticker": etf["ticker"]},
    ))

vectorstore = FAISS.from_documents(documents, embeddings)
vs = vectorstore
print(f"✅ 벡터 스토어 구축 완료: {len(documents)}개 문서")

# 평가 데이터 구성
ticker_to_id = {etf["ticker"]: doc_id for doc_id, etf in enumerate(SAMPLE_ETF_DATA)}
eval_data = []
for item in SAMPLE_EVAL_QUERIES:
    relevant_ids = [ticker_to_id[t] for t in item["expected_tickers"] if t in ticker_to_id]
    eval_data.append({
        "query": item["query"],
        "relevant_doc_ids": relevant_ids,
        "relevant_doc_names": item["expected_tickers"],
        "category": SAMPLE_ETF_DATA[relevant_ids[0]]["category"] if relevant_ids else "기타",
    })

# BM25 설정
corpus = [doc.page_content.split() for doc in documents]
bm25 = BM25Okapi(corpus)

def bm25_search(query, k=5):
    scores = bm25.get_scores(query.split())
    top_k = np.argsort(scores)[::-1][:k]
    return [{"doc_id": int(i), "score": float(scores[i]), "name": SAMPLE_ETF_DATA[i]["name"]} for i in top_k]

# 하이브리드 검색
def hybrid_search(query, alpha=0.5, k=5):
    faiss_results = vectorstore.similarity_search_with_score(query, k=len(documents))
    faiss_scores = {doc.metadata["doc_id"]: 1 / (1 + score) for doc, score in faiss_results}

    bm25_results = bm25_search(query, k=len(documents))
    bm25_max = max((r["score"] for r in bm25_results), default=1)
    bm25_scores = {r["doc_id"]: r["score"] / bm25_max if bm25_max > 0 else 0 for r in bm25_results}

    all_ids = set(faiss_scores) | set(bm25_scores)
    combined = {i: alpha * faiss_scores.get(i, 0) + (1 - alpha) * bm25_scores.get(i, 0) for i in all_ids}
    top_k = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(doc_id, score, SAMPLE_ETF_DATA[doc_id]["name"]) for doc_id, score in top_k]

# 평가 지표 함수
def hit_rate_at_k(vs_, eval_data_, k=5):
    hits = sum(
        1 for item in eval_data_
        if any(r.metadata["doc_id"] in item["relevant_doc_ids"]
               for r in vs_.similarity_search(item["query"], k=k))
    )
    return hits / len(eval_data_)

def mrr_at_k(vs_, eval_data_, k=5):
    rr_sum = 0
    for item in eval_data_:
        for rank, r in enumerate(vs_.similarity_search(item["query"], k=k), 1):
            if r.metadata["doc_id"] in item["relevant_doc_ids"]:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(eval_data_)

def ndcg_at_k(rels, k):
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rels[:k]))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(rels, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0


# ============================================================
# ✅ 문제 1: ETF 사용자 페르소나 정의
# ============================================================
print("\n" + "=" * 60)
print("문제 1: ETF 사용자 페르소나 정의")
print("=" * 60)

personas = {
    "초보": {
        "description": "투자 경험이 거의 없고, 원금 보존을 최우선시하는 투자자",
        "queries": [
            {"query": "안전한 투자 상품 추천해주세요", "expected_category": ["채권", "머니마켓"]},
            {"query": "원금 손실 위험 없는 ETF", "expected_category": ["머니마켓"]},
            {"query": "적금보다 나은 안전한 투자", "expected_category": ["채권", "배당"]},
        ],
    },
    "중급": {
        "description": "기본적인 투자 지식이 있고, 적절한 위험을 감수하며 자산 증식을 추구하는 투자자",
        "queries": [
            {"query": "S&P500 추종 ETF 비교", "expected_category": ["해외주식"]},
            {"query": "배당과 성장 균형 잡힌 포트폴리오", "expected_category": ["배당", "국내주식"]},
            {"query": "분산투자 가능한 ETF 조합", "expected_category": ["국내주식", "해외주식", "채권"]},
        ],
    },
    "전문": {
        "description": "투자 경험이 풍부하고, 높은 수익을 위해 높은 변동성을 감수할 수 있는 투자자",
        "queries": [
            {"query": "AI 반도체 관련 ETF 섹터 분석", "expected_category": ["테마", "해외주식"]},
            {"query": "레버리지 ETF 단기 트레이딩 전략", "expected_category": ["레버리지"]},
            {"query": "나스닥100 vs 코스닥150 변동성 비교", "expected_category": ["해외주식", "국내주식"]},
        ],
    },
}

with open("project2_data/query_set.json", "w") as f:
    json.dump(personas, f, ensure_ascii=False, indent=2)

print(f"✅ {sum(len(p['queries']) for p in personas.values())}개 질의 저장 완료")


# ============================================================
# ✅ 문제 2: 추가 ETF 문서 생성
# ============================================================
print("\n" + "=" * 60)
print("문제 2: 추가 ETF 문서 생성")
print("=" * 60)

additional_etfs = [
    {"name": "TIGER ESG리더스", "category": "ESG", "market": "국내"},
    {"name": "KODEX 2차전지산업", "category": "2차전지", "market": "국내"},
    {"name": "TIGER 헬스케어", "category": "헬스케어", "market": "국내"},
    {"name": "KODEX 한국부동산리츠인프라", "category": "리츠", "market": "국내"},
    {"name": "KODEX 골드선물", "category": "원자재", "market": "글로벌"},
]

risk_map = {
    "ESG": 3, "2차전지": 4, "헬스케어": 3, "리츠": 2, "원자재": 3,
    "인덱스": 2, "섹터": 4, "배당": 2, "레버리지": 5, "테마": 4, "채권": 1, "자산배분": 2,
}

for add_etf in additional_etfs:
    prompt = f"""다음 ETF에 대한 상세 설명을 작성해주세요:
- ETF명: {add_etf['name']}
- 카테고리: {add_etf['category']}
- 시장: {add_etf['market']}

다음 항목을 포함해주세요:
1. 투자 전략 (3-4문장)
2. 주요 편입 종목 (5개)
3. 수수료 및 비용 (총보수)
4. 적합한 투자자 유형
5. 주의사항

한국어로 300-400자 내외로 작성해주세요."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    doc_text = response.choices[0].message.content
    etf_documents.append({
        "name": add_etf["name"],
        "category": add_etf["category"],
        "market": add_etf["market"],
        "content": doc_text,
    })
    print(f"✅ {add_etf['name']} 문서 생성 완료 (위험도: {risk_map.get(add_etf['category'], '?')})")

with open("project2_data/raw/etf_documents.json", "w") as f:
    json.dump(etf_documents, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(etf_documents)
print(f"\n카테고리별 수:\n{df['category'].value_counts()}")


# ============================================================
# ✅ 문제 3: 멀티-관련성 질의 생성
# ============================================================
print("\n" + "=" * 60)
print("문제 3: 멀티-관련성 질의 생성")
print("=" * 60)

with open("project2_data/raw/etf_documents.json") as f:
    etf_docs = json.load(f)

eval_dataset = []
for i, doc in enumerate(etf_docs[:10]):
    prompt = f"""다음 ETF 문서를 읽고, 이 ETF를 찾기 위한 자연스러운 질의 3개를 만들어주세요.

ETF: {doc['name']}
카테고리: {doc['category']}
내용: {doc['content'][:300]}

JSON 형태로 반환: {{"queries": ["질의1", "질의2", "질의3"]}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(response.choices[0].message.content)
        queries = result.get("queries", list(result.values())[0])
        if isinstance(queries, list):
            for q in queries:
                eval_dataset.append({
                    "query": q,
                    "relevant_doc_ids": [i],
                    "relevant_doc_names": [doc["name"]],
                    "category": doc["category"],
                })
    except Exception:
        pass
    print(f"✅ {doc['name']}: 질의 생성 완료")

with open("project2_data/evaluation/eval_queries.json", "w") as f:
    json.dump(eval_dataset, f, ensure_ascii=False, indent=2)

print(f"\n📊 총 {len(eval_dataset)}개 평가 질의 생성")

multi_queries = [
    {
        "query": "분산투자에 좋은 안전한 포트폴리오 구성 추천",
        "relevant": [{"doc_id": 0, "score": 2}, {"doc_id": 8, "score": 3}, {"doc_id": 9, "score": 2}],
        "irrelevant": [6],
    },
    {
        "query": "미국 시장에 투자할 수 있는 ETF 비교",
        "relevant": [{"doc_id": 1, "score": 3}, {"doc_id": 2, "score": 3}, {"doc_id": 5, "score": 2}],
        "irrelevant": [3, 8],
    },
    {
        "query": "배당 수익과 안정성을 동시에 추구하는 ETF",
        "relevant": [{"doc_id": 4, "score": 3}, {"doc_id": 5, "score": 2}, {"doc_id": 8, "score": 2}],
        "irrelevant": [2, 6],
    },
]

cats = [item["category"] for item in eval_dataset]
cat_counts = pd.Series(cats).value_counts()

plt.figure(figsize=(8, 4))
cat_counts.plot(kind="bar")
plt.title("평가 질의 카테고리 분포")
plt.xlabel("카테고리")
plt.ylabel("질의 수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

pd.DataFrame(eval_dataset).to_csv("project2_data/evaluation/eval_queries.csv", index=False)
print("✅ CSV 저장 완료")


# ============================================================
# ✅ 문제 4: MMR 검색 구현
# ============================================================
print("\n" + "=" * 60)
print("문제 4: MMR 검색 구현")
print("=" * 60)


def format_results(results, method_name):
    print(f"\n🔍 {method_name} 결과:")
    for i, item in enumerate(results, 1):
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
            print(f"  {i}. [{score:.4f}] {doc.metadata['name']} ({doc.metadata['category']})")
        else:
            print(f"  {i}. {item.metadata['name']} ({item.metadata['category']})")


query = "분산 투자에 적합한 ETF"

start = time.time()
sim_results = vectorstore.similarity_search_with_score(query, k=5)
sim_time = time.time() - start
format_results(sim_results, f"Similarity Search ({sim_time:.3f}s)")

start = time.time()
mmr_results = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=10)
mmr_time = time.time() - start
format_results([(doc, 0) for doc in mmr_results], f"MMR Search ({mmr_time:.3f}s)")

print("\n📊 k값별 카테고리 다양성:")
for k in [3, 5, 10]:
    sim = vectorstore.similarity_search(query, k=k)
    mmr = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=max(k * 2, 10))
    sim_cats = set(d.metadata["category"] for d in sim)
    mmr_cats = set(d.metadata["category"] for d in mmr)
    print(f"  k={k}: Similarity {len(sim_cats)}개 카테고리 vs MMR {len(mmr_cats)}개 카테고리")


# ============================================================
# ✅ 문제 5: Precision@K와 Recall@K 구현
# ============================================================
print("\n" + "=" * 60)
print("문제 5: Precision@K와 Recall@K 구현")
print("=" * 60)


def precision_at_k(vs_, eval_data_, k=5):
    precisions = []
    for item in eval_data_:
        results = vs_.similarity_search(item["query"], k=k)
        retrieved = [r.metadata["doc_id"] for r in results]
        hit = sum(1 for rid in retrieved if rid in item["relevant_doc_ids"])
        precisions.append(hit / k)
    return np.mean(precisions)


def recall_at_k(vs_, eval_data_, k=5):
    recalls = []
    for item in eval_data_:
        results = vs_.similarity_search(item["query"], k=k)
        retrieved = [r.metadata["doc_id"] for r in results]
        hit = sum(1 for rid in retrieved if rid in item["relevant_doc_ids"])
        total = len(item["relevant_doc_ids"])
        recalls.append(hit / total if total > 0 else 0)
    return np.mean(recalls)


ks = range(1, 11)
hrs   = [hit_rate_at_k(vectorstore, eval_data, k) for k in ks]
mrrs  = [mrr_at_k(vectorstore, eval_data, k) for k in ks]
precs = [precision_at_k(vectorstore, eval_data, k) for k in ks]
recs  = [recall_at_k(vectorstore, eval_data, k) for k in ks]

plt.figure(figsize=(10, 6))
plt.plot(ks, hrs,   "o-", label="Hit Rate")
plt.plot(ks, mrrs,  "s-", label="MRR")
plt.plot(ks, precs, "^-", label="Precision")
plt.plot(ks, recs,  "D-", label="Recall")
plt.xlabel("K")
plt.ylabel("Score")
plt.title("검색 평가 지표 비교 (K=1~10)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# ✅ 문제 6: NDCG 평가 리포트
# ============================================================
print("\n" + "=" * 60)
print("문제 6: NDCG 평가 리포트")
print("=" * 60)

report = {"baseline": {}}

for k in [1, 3, 5, 10]:
    hr  = hit_rate_at_k(vectorstore, eval_data, k)
    mrr = mrr_at_k(vectorstore, eval_data, k)

    ndcg_scores = []
    for item in eval_data:
        results = vs.similarity_search(item["query"], k=k)
        rels = [1 if r.metadata["doc_id"] in item["relevant_doc_ids"] else 0 for r in results]
        ndcg_scores.append(ndcg_at_k(rels, k))

    report["baseline"][f"k={k}"] = {
        "hit_rate": round(hr, 4),
        "mrr":      round(mrr, 4),
        "ndcg":     round(float(np.mean(ndcg_scores)), 4),
    }

with open("project2_data/evaluation/baseline_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"{'K':<6} | {'Hit Rate':>10} | {'MRR':>10} | {'NDCG':>10}")
print("-" * 45)
for k_str, metrics in report["baseline"].items():
    print(f"{k_str:<6} | {metrics['hit_rate']:>10.4f} | {metrics['mrr']:>10.4f} | {metrics['ndcg']:>10.4f}")
print("\n✅ baseline_report.json 저장 완료")


# ============================================================
# ✅ 문제 7: BM25 vs FAISS 비교 평가
# ============================================================
print("\n" + "=" * 60)
print("문제 7: BM25 vs FAISS 비교 평가")
print("=" * 60)


def bm25_hit_rate(eval_data_, k=5):
    hits = 0
    for item in eval_data_:
        retrieved = [r["doc_id"] for r in bm25_search(item["query"], k)]
        if any(rid in item["relevant_doc_ids"] for rid in retrieved):
            hits += 1
    return hits / len(eval_data_)


def bm25_mrr(eval_data_, k=5):
    rr_sum = 0
    for item in eval_data_:
        for rank, r in enumerate(bm25_search(item["query"], k), 1):
            if r["doc_id"] in item["relevant_doc_ids"]:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(eval_data_)


print(f"{'방법':<12} | {'K':>3} | {'Hit Rate':>10} | {'MRR':>10}")
print("-" * 45)
for k in [1, 3, 5, 10]:
    f_hr  = hit_rate_at_k(vectorstore, eval_data, k)
    f_mrr = mrr_at_k(vectorstore, eval_data, k)
    b_hr  = bm25_hit_rate(eval_data, k)
    b_mrr = bm25_mrr(eval_data, k)
    print(f"{'FAISS':<12} | {k:>3} | {f_hr:>10.4f} | {f_mrr:>10.4f}")
    print(f"{'BM25':<12}  | {k:>3} | {b_hr:>10.4f} | {b_mrr:>10.4f}")
    print("-" * 45)


# ============================================================
# ✅ 문제 8: Alpha 최적화
# ============================================================
print("\n" + "=" * 60)
print("문제 8: Alpha 최적화")
print("=" * 60)

best_alpha, best_hr = 0, 0

print(f"{'Alpha':>6} | {'Hit Rate@5':>12}")
print("-" * 25)

for a in np.arange(0, 1.1, 0.1):
    a = round(a, 1)
    hits = 0
    for item in eval_data:
        results = hybrid_search(item["query"], alpha=a, k=5)
        if any(r[0] in item["relevant_doc_ids"] for r in results):
            hits += 1
    hr = hits / len(eval_data)
    print(f"{a:>6.1f} | {hr:>12.4f}")
    if hr > best_hr:
        best_alpha, best_hr = a, hr

checkpoint = {"best_alpha": best_alpha, "best_hr": round(best_hr, 4)}
with open("project2_data/checkpoints/hybrid_config.json", "w") as f:
    json.dump(checkpoint, f, ensure_ascii=False, indent=2)

print(f"\n🏆 최적 alpha: {best_alpha} (Hit Rate: {best_hr:.4f})")
print("✅ hybrid_config.json 저장 완료")


# ============================================================
# ✅ 문제 9: 도메인 특화 동의어 사전
# ============================================================
print("\n" + "=" * 60)
print("문제 9: 도메인 특화 동의어 사전")
print("=" * 60)

finance_synonyms = {
    "ETF":  ["상장지수펀드", "인덱스펀드", "지수추종"],
    "배당":  ["분배금", "배당금", "인컴", "이자수익"],
    "안정":  ["안전", "보수적", "저위험", "원금보존"],
    "성장":  ["그로스", "공격적", "고수익", "고성장"],
    "미국":  ["해외", "글로벌", "나스닥", "S&P"],
}


def synonym_expand(query):
    expanded = [query]
    for keyword, synonyms in finance_synonyms.items():
        if keyword in query:
            for syn in synonyms:
                expanded.append(query.replace(keyword, syn))
    return expanded


baseline_hits, synonym_hits = 0, 0
for item in eval_data:
    results = hybrid_search(item["query"], alpha=0.5, k=5)
    if any(r[0] in item["relevant_doc_ids"] for r in results):
        baseline_hits += 1

    all_results = {}
    for eq in synonym_expand(item["query"]):
        for did, score, name in hybrid_search(eq, alpha=0.5, k=5):
            if did not in all_results or score > all_results[did][1]:
                all_results[did] = (did, score, name)
    top5 = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:5]
    if any(r[0] in item["relevant_doc_ids"] for r in top5):
        synonym_hits += 1

print(f"Baseline Hit Rate: {baseline_hits / len(eval_data):.4f}")
print(f"Synonym  Hit Rate: {synonym_hits / len(eval_data):.4f}")


# ============================================================
# ✅ 문제 10: 커스텀 Multi-Query Retriever
# ============================================================
print("\n" + "=" * 60)
print("문제 10: 커스텀 Multi-Query Retriever")
print("=" * 60)

custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""당신은 ETF 금융 상품 검색 전문가입니다.
다음 질문을 서로 다른 관점에서 3가지로 재작성하세요.
각 질의는 ETF 검색에 최적화되어야 합니다.

원래 질문: {question}

재작성된 질의 (한 줄에 하나씩):""",
)

retriever_custom = MultiQueryRetriever.from_llm(
    retriever=vs.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
    prompt=custom_prompt,
)

results_custom = retriever_custom.invoke("노후 대비 안정적 투자")
print(f"🔍 커스텀 Multi-Query 결과: {len(results_custom)}개")
for doc in results_custom:
    print(f"  - {doc.metadata['name']}: {doc.page_content[:80]}...")


# ============================================================
# ✅ 문제 11: 검색 비교 대시보드
# ============================================================
print("\n" + "=" * 60)
print("문제 11: 검색 비교 대시보드")
print("=" * 60)

search_history = []


def full_comparison(query, top_k):
    top_k = int(top_k)
    search_history.append(query)
    output = f"🔍 질의: {query}\n{'=' * 60}\n\n"

    faiss_results = vs.similarity_search_with_score(query, k=top_k)
    output += "📌 FAISS 벡터 검색:\n"
    for i, (doc, score) in enumerate(faiss_results, 1):
        output += f"  {i}. [{score:.4f}] {doc.metadata['name']} ({doc.metadata['category']})\n"

    bm25_results = bm25_search(query, top_k)
    output += "\n📌 BM25 키워드 검색:\n"
    for i, r in enumerate(bm25_results, 1):
        output += f"  {i}. [{r['score']:.4f}] {r['name']}\n"

    hybrid_results = hybrid_search(query, alpha=0.5, k=top_k)
    output += "\n📌 하이브리드 검색 (α=0.5):\n"
    for i, (did, score, name) in enumerate(hybrid_results, 1):
        output += f"  {i}. [{score:.4f}] {name}\n"

    return output


def show_history():
    if not search_history:
        return "검색 이력이 없습니다."
    return "\n".join(f"{i + 1}. {q}" for i, q in enumerate(search_history))


with gr.Blocks(title="ETF 검색 비교 대시보드") as demo:
    with gr.Tab("검색"):
        query_input  = gr.Textbox(label="검색 질의", placeholder="예: 배당 수익률 높은 안전한 ETF")
        top_k_slider = gr.Slider(1, 10, value=5, step=1, label="결과 수 (K)")
        search_btn   = gr.Button("검색")
        output_box   = gr.Textbox(label="비교 결과", lines=20)
        search_btn.click(full_comparison, [query_input, top_k_slider], output_box)

    with gr.Tab("검색 이력"):
        history_btn    = gr.Button("이력 조회")
        history_output = gr.Textbox(label="최근 검색 이력", lines=10)
        history_btn.click(show_history, [], history_output)

demo.launch(share=True)