# ============================================================
#  p2_weekend3_realdata_deploy_260418_김민석.py
#  프로젝트 2 - Weekend 3: 실전 데이터 기반 추천 시스템
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import json
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from dataclasses import dataclass, asdict
from numpy.linalg import norm
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document

load_dotenv()

MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("✅ 환경 설정 완료")

# ============================================================
# 스타터 코드 1/5: ETF 데이터
# ============================================================
etf_data = [
    {"name": "KODEX 200",             "category": "국내주식", "expense_ratio": 0.15, "return_1y":  8.5, "risk_level": "중간", "dividend_yield": 1.8, "volatility": 15.2,
     "keywords": ["코스피", "대형주", "인덱스", "분산투자", "국내주식"],
     "description": "KOSPI 200 지수를 추종하는 대표 국내 ETF. 대형주 중심 분산투자에 적합합니다."},
    {"name": "KODEX 미국S&P500TR",    "category": "해외주식", "expense_ratio": 0.05, "return_1y": 25.3, "risk_level": "중간", "dividend_yield": 0.0, "volatility": 18.5,
     "keywords": ["S&P500", "미국", "대형주", "패시브", "해외주식"],
     "description": "미국 S&P500 지수의 총수익(TR)을 추종. 저비용으로 미국 대형주 전체에 분산투자합니다."},
    {"name": "ACE 미국배당다우존스",   "category": "배당",     "expense_ratio": 0.01, "return_1y": 12.1, "risk_level": "낮음", "dividend_yield": 3.5, "volatility": 10.3,
     "keywords": ["미국배당", "다우존스", "고배당", "월배당", "안정"],
     "description": "미국 고배당 대형주 중심. 최저 수수료(0.01%)로 안정적 배당수익을 추구합니다."},
    {"name": "TIGER 2차전지테마",     "category": "테마",     "expense_ratio": 0.45, "return_1y":-15.2, "risk_level": "높음", "dividend_yield": 0.0, "volatility": 35.7,
     "keywords": ["2차전지", "배터리", "성장주", "고위험", "테마"],
     "description": "2차전지·배터리 관련 기업에 집중 투자. 고성장 기대 대신 높은 변동성을 감수해야 합니다."},
    {"name": "TIGER 국고채10년",      "category": "채권",     "expense_ratio": 0.07, "return_1y":  4.2, "risk_level": "낮음", "dividend_yield": 2.8, "volatility":  5.1,
     "keywords": ["국고채", "10년", "안전자산", "금리", "채권"],
     "description": "한국 10년 국고채 지수를 추종하는 안전자산 ETF. 금리 하락기에 유리합니다."},
    {"name": "KODEX 골드선물(H)",     "category": "원자재",   "expense_ratio": 0.68, "return_1y": 18.7, "risk_level": "중간", "dividend_yield": 0.0, "volatility": 20.4,
     "keywords": ["금", "골드", "인플레이션", "안전자산", "원자재"],
     "description": "금 선물 가격을 추종하며 환헤지 적용. 인플레이션 방어 및 포트폴리오 분산용입니다."},
    {"name": "KODEX 미국나스닥100TR", "category": "해외주식", "expense_ratio": 0.05, "return_1y": 32.1, "risk_level": "높음", "dividend_yield": 0.0, "volatility": 25.3,
     "keywords": ["나스닥100", "미국", "기술주", "성장", "해외주식"],
     "description": "나스닥100 기술주 중심 총수익 ETF. 높은 성장성과 높은 변동성을 동반합니다."},
    {"name": "TIGER 미국반도체",      "category": "섹터",     "expense_ratio": 0.49, "return_1y": 45.0, "risk_level": "높음", "dividend_yield": 0.0, "volatility": 38.2,
     "keywords": ["반도체", "미국", "필라델피아", "기술", "섹터"],
     "description": "미국 필라델피아 반도체 지수 추종. AI/반도체 수요 수혜 기대되나 변동성 극심합니다."},
    {"name": "KODEX 단기채권PLUS",    "category": "채권",     "expense_ratio": 0.03, "return_1y":  3.8, "risk_level": "낮음", "dividend_yield": 3.2, "volatility":  1.5,
     "keywords": ["단기채", "안전", "저위험", "현금성", "채권"],
     "description": "단기 채권 중심의 초저위험 ETF. 현금성 자산 대용으로 안정적 수익을 제공합니다."},
    {"name": "TIGER 리츠부동산",      "category": "리츠",     "expense_ratio": 0.29, "return_1y":  6.5, "risk_level": "중간", "dividend_yield": 4.8, "volatility": 12.8,
     "keywords": ["리츠", "부동산", "배당", "인프라", "실물자산"],
     "description": "국내 리츠·부동산 인프라에 투자. 높은 배당수익률(4.8%)과 실물자산 분산 효과가 있습니다."},
    {"name": "KODEX 200TR",           "category": "국내주식", "expense_ratio": 0.12, "return_1y":  9.1, "risk_level": "중간", "dividend_yield": 0.0, "volatility": 14.9,
     "keywords": ["코스피200", "TR", "총수익", "인덱스", "국내주식"],
     "description": "KODEX 200의 총수익(TR) 버전. 배당 재투자 효과를 포함하여 장기 성과가 우수합니다."},
    {"name": "TIGER 고배당저변동",    "category": "배당",     "expense_ratio": 0.30, "return_1y":  7.2, "risk_level": "낮음", "dividend_yield": 5.2, "volatility":  8.7,
     "keywords": ["고배당", "저변동", "방어적", "배당", "안정"],
     "description": "고배당+저변동성 종목을 선별. 배당수익률 5.2%로 방어적 포트폴리오에 적합합니다."},
]

print(f"✅ ETF 데이터 {len(etf_data)}개 로드")

# ============================================================
# 스타터 코드 2/5: Document 구성 + FAISS + Kiwi BM25
# ============================================================
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

documents = []
for etf in etf_data:
    text = (f"{etf['name']} ({etf['category']}): {etf['description']} "
            f"키워드: {', '.join(etf['keywords'])} "
            f"수수료 {etf['expense_ratio']}% 배당 {etf['dividend_yield']}% "
            f"수익률 {etf['return_1y']:.1f}% 변동성 {etf['volatility']:.1f}%")
    documents.append(Document(page_content=text, metadata=etf))

vectorstore = LangchainFAISS.from_documents(documents, embeddings)
print(f"✅ FAISS: {vectorstore.index.ntotal}개 벡터")

kiwi = Kiwi()

def kiwi_tokenize(text):
    """Kiwi 형태소 분석: NNG/NNP/SL/SN만 추출"""
    return [t.form.lower() for t in kiwi.tokenize(text) if t.tag in ("NNG", "NNP", "SL", "SN")]

corpus = [kiwi_tokenize(doc.page_content) for doc in documents]
bm25 = BM25Okapi(corpus)

def bm25_search(query, k=5):
    """BM25 키워드 검색"""
    tokens = kiwi_tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(documents[i].metadata["name"], scores[i]) for i in top_idx if scores[i] > 0]

print(f"✅ BM25: {len(corpus)}개 문서 인덱싱")

# ============================================================
# 스타터 코드 3/5: 하이브리드 검색 + 메타데이터 필터
# ============================================================
def hybrid_search(query, alpha=0.5, k=5):
    """벡터 + BM25 하이브리드 (Min-Max 정규화)"""
    vec_results = vectorstore.similarity_search_with_score(query, k=20)
    vec_scores = {doc.metadata["name"]: 1 / (1 + score) for doc, score in vec_results}

    tokens = kiwi_tokenize(query)
    bm25_raw = bm25.get_scores(tokens)
    bm25_scores = {documents[i].metadata["name"]: bm25_raw[i] for i in range(len(documents))}

    def minmax(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        return {k: (v - mn) / (mx - mn + 1e-8) for k, v in d.items()}

    vn, bn = minmax(vec_scores), minmax(bm25_scores)
    combined = {n: alpha * vn.get(n, 0) + (1 - alpha) * bn.get(n, 0) for n in set(vn) | set(bn)}
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]


def filtered_search(vectorstore, query, filters=None, k=5, fetch_k=20):
    """벡터 검색 + 메타데이터 필터"""
    results = vectorstore.similarity_search_with_score(query, k=fetch_k)
    if not filters:
        return results[:k]
    filtered = []
    for doc, score in results:
        m = doc.metadata
        ok = True
        for key, val in filters.items():
            if isinstance(val, dict):
                if "less_than" in val and m.get(key, float("inf")) >= val["less_than"]:
                    ok = False
                if "greater_than" in val and m.get(key, 0) <= val["greater_than"]:
                    ok = False
            else:
                if m.get(key) != val:
                    ok = False
        if ok:
            filtered.append((doc, score))
    return filtered[:k]


eval_queries = [
    {"query": "안전한 배당 ETF", "relevant": ["ACE 미국배당다우존스", "TIGER 국고채10년", "KODEX 단기채권PLUS"]},
    {"query": "미국 기술주 투자", "relevant": ["KODEX 미국나스닥100TR", "TIGER 미국반도체"]},
    {"query": "KODEX 200", "relevant": ["KODEX 200", "KODEX 200TR"]},
    {"query": "인플레이션 방어", "relevant": ["KODEX 골드선물(H)"]},
    {"query": "2차전지 테마", "relevant": ["TIGER 2차전지테마"]},
    {"query": "저비용 해외 ETF", "relevant": ["KODEX 미국S&P500TR", "KODEX 미국나스닥100TR"]},
    {"query": "고배당 저위험", "relevant": ["TIGER 고배당저변동", "ACE 미국배당다우존스"]},
    {"query": "부동산 투자", "relevant": ["TIGER 리츠부동산"]},
]

print(f"✅ 하이브리드 검색 + 필터 + 평가 쿼리 {len(eval_queries)}개 준비")

# ============================================================
# 스타터 코드 4/5: CBF + 협업 필터링 (CF)
# ============================================================
risk_map = {"낮음": 1, "중간": 2, "높음": 3}


def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


def etf_to_vector(etf):
    return np.array([
        risk_map[etf["risk_level"]] / 3,
        (etf["return_1y"] + 30) / 80,
        etf["dividend_yield"] / 6,
        etf["expense_ratio"] / 1,
        etf["volatility"] / 40,
    ])


item_vectors = np.array([etf_to_vector(e) for e in etf_data])
item_names = [e["name"] for e in etf_data]

n = len(item_vectors)
item_sim = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        item_sim[i, j] = cosine_sim(item_vectors[i], item_vectors[j])


def cbf_similar_items(target_name, top_k=5):
    """item-item 코사인 유사도 Top-K"""
    idx = item_names.index(target_name)
    scores = [(item_names[j], item_sim[idx, j]) for j in range(n) if j != idx]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


# 협업 필터링 평점 행렬 구성
np.random.seed(42)
user_types = ["보수적_배당", "보수적_배당", "중립_인덱스", "중립_인덱스",
              "공격적_성장", "공격적_성장", "공격적_성장",
              "중립_균형", "배당매니아", "글로벌투자자"]


def seed_rating(ut, etf):
    cat = etf["category"]
    prefs = {
        "보수적_배당":  {"채권": 5, "배당": 5, "리츠": 4, "국내주식": 2, "해외주식": 1},
        "중립_인덱스":  {"국내주식": 5, "해외주식": 5, "배당": 3, "채권": 3, "리츠": 2, "원자재": 2},
        "공격적_성장":  {"해외주식": 5, "섹터": 5, "테마": 4, "원자재": 3, "국내주식": 3, "배당": 1},
        "배당매니아":   {"배당": 5, "리츠": 5, "채권": 3, "국내주식": 3, "해외주식": 2},
        "글로벌투자자": {"해외주식": 5, "원자재": 4, "섹터": 4, "국내주식": 2, "배당": 2, "테마": 3},
    }
    if ut == "중립_균형":
        return 3 + np.random.randint(-1, 2)
    return prefs.get(ut, {}).get(cat, 0)


R = np.zeros((len(user_types), len(etf_data)))
for u, ut in enumerate(user_types):
    for i, etf in enumerate(etf_data):
        r = seed_rating(ut, etf)
        if r > 0 and np.random.random() > 0.3:
            R[u, i] = r

user_sim = np.zeros((R.shape[0], R.shape[0]))
for u in range(R.shape[0]):
    for v in range(R.shape[0]):
        if u == v:
            user_sim[u, v] = 1.0
            continue
        mask = (R[u] > 0) & (R[v] > 0)
        user_sim[u, v] = cosine_sim(R[u][mask], R[v][mask]) if mask.sum() >= 2 else 0


def cf_recommend(user_idx, top_k=5):
    """User-based CF: 유사 사용자 가중평균"""
    sims = user_sim[user_idx]
    pred = np.zeros(R.shape[1])
    for i in range(R.shape[1]):
        if R[user_idx, i] > 0:
            pred[i] = -1
            continue
        raters = np.where(R[:, i] > 0)[0]
        if len(raters) > 0 and sims[raters].sum() > 0:
            pred[i] = np.dot(sims[raters], R[raters, i]) / sims[raters].sum()
    top_idx = np.argsort(pred)[::-1][:top_k]
    return [(item_names[i], round(pred[i], 3)) for i in top_idx if pred[i] > 0]


print(f"✅ CBF: {n}×{n} 유사도 행렬 | CF: {R.shape} 평점 행렬 (밀도 {(R > 0).sum() / R.size:.0%})")

# ============================================================
# 스타터 코드 5/5: LLM 추천 헬퍼
# ============================================================
@dataclass
class InvestorProfile:
    name: str
    risk_tolerance: str
    investment_goal: str
    investment_horizon: int
    monthly_budget: int


def extract_profile(user_text):
    """자연어 → InvestorProfile"""
    prompt = f"""사용자 메시지에서 투자자 프로필을 JSON으로 추출하세요.
메시지: {user_text}
JSON: {{"name": "사용자", "risk_tolerance": "보수적|중립|공격적", "investment_goal": "안정수익|자산증식|배당수익", "investment_horizon": 숫자, "monthly_budget": 숫자}}"""
    resp = llm.invoke([{"role": "user", "content": prompt}]).content
    try:
        data = json.loads(resp)
    except Exception:
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    return InvestorProfile(
        name=data.get("name", "사용자"),
        risk_tolerance=data.get("risk_tolerance", "중립"),
        investment_goal=data.get("investment_goal", "자산증식"),
        investment_horizon=int(data.get("investment_horizon") or 5),
        monthly_budget=int(data.get("monthly_budget") or 100),
    )


def rule_based_filter(risk_tolerance, etf_data):
    allowed = {"보수적": ["낮음"], "중립": ["낮음", "중간"], "공격적": ["낮음", "중간", "높음"]}
    return [e for e in etf_data if e["risk_level"] in allowed[risk_tolerance]]


def llm_recommend(profile, candidates, top_k=3):
    """LLM 추천 + allocation + reason"""
    etf_list = "\n".join([
        f"- {e['name']}: {e['category']}, 수익률 {e['return_1y']}%, 배당 {e['dividend_yield']}%, "
        f"위험 {e['risk_level']}, 수수료 {e['expense_ratio']}%"
        for e in candidates
    ])
    prompt = (f"투자자: {profile.name} / {profile.risk_tolerance} / {profile.investment_goal} / "
              f"{profile.investment_horizon}년 / 월 {profile.monthly_budget}만원\n"
              f"후보: {etf_list}\n"
              f"적합한 ETF {top_k}개를 JSON 배열로. "
              f'[{{"name":"ETF명","allocation":비중(%),"reason":"이유"}}] 합계 100%. 순수 JSON만.')
    resp = llm.invoke([{"role": "system", "content": "ETF 투자 전문가."}, {"role": "user", "content": prompt}]).content
    try:
        return json.loads(resp)
    except Exception:
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        return json.loads(match.group()) if match else []


print("✅ LLM 헬퍼 준비 완료")
print("\n🎯 스타터 코드 로드 완료!")

# ============================================================
# ✅ 실습 1: split() vs Kiwi 토큰화 품질 비교
# ============================================================
print("\n" + "=" * 60)
print("실습 1: split() vs Kiwi 토큰화 품질 비교")
print("=" * 60)


def split_tokenize(text):
    """단순 공백 분리 토크나이저 (특수문자 제거 + 2자 이상만)"""
    return [t.lower() for t in re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text).split() if len(t) > 1]


corpus_split = [split_tokenize(doc.page_content) for doc in documents]
bm25_split = BM25Okapi(corpus_split)


def bm25_split_search(query, k=5):
    tokens = split_tokenize(query)
    scores = bm25_split.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(documents[i].metadata["name"], scores[i]) for i in top_idx if scores[i] > 0]


def hit_rate(eval_data, search_fn, k=5):
    hits = 0
    for item in eval_data:
        found = [r[0] for r in search_fn(item["query"], k=k)]
        if any(rel in found for rel in item["relevant"]):
            hits += 1
    return hits / len(eval_data)


split_hr = hit_rate(eval_queries, bm25_split_search, k=3)
kiwi_hr = hit_rate(eval_queries, bm25_search, k=3)
print(f"Hit Rate@3:  split={split_hr:.3f}  vs  Kiwi={kiwi_hr:.3f}")
print(f"개선율: {(kiwi_hr - split_hr) / (split_hr + 1e-8) * 100:+.0f}%\n")

print(f"{'쿼리':20s} | {'split Top-1':20s} | {'Kiwi Top-1':20s} | {'split':5s} | {'Kiwi':5s}")
print("-" * 85)
for item in eval_queries:
    q = item["query"]
    s_res = bm25_split_search(q, k=3)
    k_res = bm25_search(q, k=3)
    s_top = s_res[0][0] if s_res else "-"
    k_top = k_res[0][0] if k_res else "-"
    s_hit = "✅" if any(r in item["relevant"] for r in [x[0] for x in s_res]) else "❌"
    k_hit = "✅" if any(r in item["relevant"] for r in [x[0] for x in k_res]) else "❌"
    print(f"{q:20s} | {s_top:20s} | {k_top:20s} | {s_hit:5s} | {k_hit:5s}")

print("\n💡 Kiwi가 복합어(배당수익률→배당+수익률)를 분리하여 BM25 매칭률이 높아집니다.")

# ============================================================
# ✅ 실습 2: 쿼리 의도 분류 + 스마트 검색 라우터
# ============================================================
print("\n" + "=" * 60)
print("실습 2: 쿼리 의도 분류 + 스마트 검색 라우터")
print("=" * 60)

BRANDS = ["KODEX", "TIGER", "ACE", "ARIRANG", "HANARO", "SOL", "KBSTAR"]


def detect_intent(query):
    """쿼리 → '조건' | '키워드' | '의미'"""
    # 숫자+%, 이상/이하/초과/미만 → 조건 검색
    if re.search(r"\d+\s*%|이상|이하|초과|미만", query):
        return "조건"
    # 브랜드명(대문자) 포함 → 키워드 검색
    for b in BRANDS:
        if b in query.upper():
            return "키워드"
    # 나머지 → 의미 검색(하이브리드)
    return "의미"


def extract_filters(query):
    """LLM으로 자연어에서 필터 JSON 추출"""
    prompt = f"""사용자 쿼리에서 ETF 검색 필터를 추출하세요.
쿼리: {query}
필터 키: category(국내주식/해외주식/배당/테마/채권/원자재/섹터/리츠), risk_level(낮음/중간/높음),
expense_ratio({{"less_than":숫자}}), dividend_yield({{"greater_than":숫자}})
순수 JSON만. 해당 없으면 {{}}"""
    resp = llm.invoke([{"role": "user", "content": prompt}]).content
    try:
        return json.loads(resp)
    except Exception:
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        return json.loads(match.group()) if match else {}


def smart_router(query, k=3):
    """의도 분류 → 적절한 검색 함수 자동 선택"""
    intent = detect_intent(query)
    if intent == "조건":
        filters = extract_filters(query)
        results = filtered_search(vectorstore, query, filters=filters, k=k)
        return [(doc.metadata["name"], score) for doc, score in results]
    elif intent == "키워드":
        return bm25_search(query, k=k)
    else:
        return hybrid_search(query, alpha=0.5, k=k)


test_queries = [
    "수수료 0.1% 이하 ETF",
    "KODEX 200",
    "안전한 장기투자",
    "배당 3% 이상 해외 ETF",
    "2차전지 관련 성장주",
]
for q in test_queries:
    intent = detect_intent(q)
    results = smart_router(q, k=3)
    names = [r[0] for r in results] if results else []
    print(f"[{intent:3s}] {q:25s} → {names}")

# ============================================================
# ✅ 실습 3: 하이브리드 α 최적화 + 3방식 비교표
# ============================================================
print("\n" + "=" * 60)
print("실습 3: 하이브리드 α 최적화 + 3방식 비교")
print("=" * 60)

best_alpha, best_hr = 0, 0
print("α 스윕:")
for a in np.arange(0, 1.1, 0.1):
    a = round(a, 1)

    def _fn(q, k=5, _a=a):
        return hybrid_search(q, alpha=_a, k=k)

    hr = hit_rate(eval_queries, _fn, k=3)
    if hr > best_hr:
        best_alpha, best_hr = a, hr
    print(f"  α={a:.1f}: HR@3={hr:.3f}")

print(f"\n✅ 최적: α={best_alpha:.1f}, HR@3={best_hr:.3f}\n")


def vec_fn(q, k=5):
    r = vectorstore.similarity_search(q, k=k)
    return [(d.metadata["name"], 1.0) for d in r]


def hybrid_best_fn(q, k=5):
    return hybrid_search(q, alpha=best_alpha, k=k)


print(f"{'방법':25s} | HR@3   | HR@5")
print("-" * 50)
for name, fn in [
    ("FAISS (벡터)", vec_fn),
    ("BM25 (Kiwi)", bm25_search),
    (f"Hybrid (α={best_alpha})", hybrid_best_fn),
]:
    hr3 = hit_rate(eval_queries, fn, k=3)
    hr5 = hit_rate(eval_queries, fn, k=5)
    print(f"{name:25s} | {hr3:.3f}  | {hr5:.3f}")

# α=0 vs α=1 실패 쿼리 분석
print("\n실패 쿼리 분석:")
for item in eval_queries:
    q = item["query"]
    v_hit = any(r[0] in item["relevant"] for r in vec_fn(q, k=3))
    b_hit = any(r[0] in item["relevant"] for r in bm25_search(q, k=3))
    if not v_hit or not b_hit:
        fail_tags = []
        if not v_hit:
            fail_tags.append("벡터❌")
        if not b_hit:
            fail_tags.append("BM25❌")
        print(f"  {q:20s} → {' '.join(fail_tags)}")

# ============================================================
# ✅ 실습 4: MAP@K + LLM-as-Judge 검색 품질 평가
# ============================================================
print("\n" + "=" * 60)
print("실습 4: MAP@K + LLM-as-Judge 검색 품질 평가")
print("=" * 60)


def average_precision(ranked_names, relevant_set, k=5):
    """단일 쿼리의 Average Precision@K"""
    ranked = ranked_names[:k]
    hits, sum_prec = 0, 0
    for i, name in enumerate(ranked, 1):
        if name in relevant_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(k, len(relevant_set)) if relevant_set else 0


def map_at_k(eval_data, search_fn, k=5):
    """Mean Average Precision@K"""
    aps = []
    for item in eval_data:
        results = search_fn(item["query"], k=k)
        ranked = [r[0] for r in results]
        aps.append(average_precision(ranked, set(item["relevant"]), k))
    return sum(aps) / len(aps)


def llm_judge_search(query, results, k=5):
    """LLM이 검색 결과의 관련성을 1~5점으로 채점"""
    etf_list = "\n".join([f"{i+1}. {r[0]}" for i, r in enumerate(results[:k])])
    prompt = f"""검색 쿼리에 대한 각 ETF의 관련성을 1~5점으로 채점하세요.
쿼리: {query}
검색 결과:
{etf_list}
JSON: [{{"name": "ETF명", "relevance": 점수, "reason": "이유"}}]"""
    resp = llm.invoke([
        {"role": "system", "content": "검색 품질 평가자."},
        {"role": "user", "content": prompt},
    ]).content
    try:
        return json.loads(resp)
    except Exception:
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        return json.loads(match.group()) if match else []


# MAP@5 비교
print(f"{'방법':25s} | MAP@5")
print("-" * 40)
for name, fn in [
    ("FAISS (벡터)", vec_fn),
    ("BM25 (Kiwi)", bm25_search),
    (f"Hybrid (α={best_alpha})", hybrid_best_fn),
]:
    print(f"{name:25s} | {map_at_k(eval_queries, fn, 5):.3f}")

# LLM Judge 샘플 (2개 쿼리)
print("\nLLM-as-Judge (Hybrid 결과):")
for q in ["안전한 배당 ETF", "미국 기술주 투자"]:
    results = hybrid_search(q, alpha=best_alpha, k=3)
    scores = llm_judge_search(q, results)
    avg = np.mean([s["relevance"] for s in scores]) if scores else 0
    print(f"  {q}: 평균 {avg:.1f}/5")
    for s in scores:
        print(f"    {s['name']}: {s['relevance']}/5 — {s.get('reason', '')[:40]}")

# ============================================================
# ✅ 실습 5: CBF 다양성 확보 + Cold-Start 폴백
# ============================================================
print("\n" + "=" * 60)
print("실습 5: CBF 다양성 확보 + Cold-Start 폴백")
print("=" * 60)


def cbf_diverse(target_name, top_k=5):
    """카테고리 중복 없이 유사 ETF 추천"""
    idx = item_names.index(target_name)
    scored = [(j, item_sim[idx, j]) for j in range(n) if j != idx]
    scored.sort(key=lambda x: x[1], reverse=True)

    result, seen_cats = [], set()
    for j, score in scored:
        cat = etf_data[j]["category"]
        if cat in seen_cats:
            continue
        result.append((item_names[j], round(score, 3), cat))
        seen_cats.add(cat)
        if len(result) >= top_k:
            break
    return result


def cold_start_recommend(R, user_idx, favorite_etf=None, top_k=5):
    """신규 사용자(평점 < 2개)면 CBF 폴백, 기존 사용자면 CF"""
    n_ratings = (R[user_idx] > 0).sum() if user_idx < R.shape[0] else 0
    if n_ratings < 2:
        if favorite_etf is None:
            return [("(선호 ETF를 알려주세요)", 0, "CBF-불가")]
        return [(name, score, "CBF-폴백") for name, score in cbf_similar_items(favorite_etf, top_k)]
    return [(name, score, "CF") for name, score in cf_recommend(user_idx, top_k)]


print("CBF 일반:")
for name, score in cbf_similar_items("KODEX 200", top_k=5):
    etf = next(e for e in etf_data if e["name"] == name)
    print(f"  {score:.3f} | {name} ({etf['category']})")

print("\nCBF 다양성:")
for name, score, cat in cbf_diverse("KODEX 200", top_k=5):
    print(f"  {score:.3f} | {name} ({cat})")

print("\nCold-start (신규, fav=KODEX 200):")
for row in cold_start_recommend(R, user_idx=99, favorite_etf="KODEX 200", top_k=3):
    print(f"  {row}")

print("기존 사용자 U0:")
for row in cold_start_recommend(R, user_idx=0, top_k=3):
    print(f"  {row}")

# ============================================================
# ✅ 실습 6: 프로필 기반 추천 + 포트폴리오 리스크 분석
# ============================================================
print("\n" + "=" * 60)
print("실습 6: 프로필 기반 추천 + 포트폴리오 리스크 분석")
print("=" * 60)


def analyze_risk(profile, recommendations, etf_data):
    """포트폴리오 리스크 경고 생성"""
    etf_dict = {e["name"]: e for e in etf_data}
    warnings = []
    rec_etfs = [etf_dict[r["name"]] for r in recommendations if r["name"] in etf_dict]
    if not rec_etfs:
        return warnings

    # 1) 카테고리 집중도 > 60%
    cat_counts = Counter(e["category"] for e in rec_etfs)
    top_cat, top_n = cat_counts.most_common(1)[0]
    max_pct = top_n / len(rec_etfs) * 100
    if max_pct > 60:
        warnings.append({
            "type": "집중투자", "severity": "warning",
            "message": f"카테고리 '{top_cat}'에 {max_pct:.0f}% 집중",
        })

    # 2) 위험 불일치 (추천 ETF 위험도 > 투자자 허용 위험도)
    risk_order = {"낮음": 1, "중간": 2, "높음": 3}
    tol_order = {"보수적": 1, "중립": 2, "공격적": 3}
    p_risk = tol_order[profile.risk_tolerance]
    for e in rec_etfs:
        if risk_order[e["risk_level"]] > p_risk:
            warnings.append({
                "type": "위험불일치", "severity": "danger",
                "message": f"{e['name']}({e['risk_level']}) > {profile.risk_tolerance}",
            })

    # 3) 평균 수수료 > 0.5%
    avg_exp = np.mean([e["expense_ratio"] for e in rec_etfs])
    if avg_exp > 0.5:
        warnings.append({
            "type": "고수수료", "severity": "info",
            "message": f"평균 수수료 {avg_exp:.2f}%",
        })

    return warnings


def explain_risks(profile, warnings):
    """LLM으로 리스크 100자 이내 설명"""
    if not warnings:
        return "특별한 리스크 경고가 없습니다."
    w_text = "\n".join([f"- [{w['type']}] {w['message']}" for w in warnings])
    prompt = f"""투자자 {profile.name}({profile.risk_tolerance})에게 리스크를 쉽게 설명. 100자 이내.
경고:
{w_text}"""
    return llm.invoke([
        {"role": "system", "content": "친절한 금융 어드바이저."},
        {"role": "user", "content": prompt},
    ]).content.strip()


test_users = [
    "30대 공격적 투자자, 월 200만원, 자산증식 목표",
    "55세 은퇴 준비, 월 100만원, 안정적 배당 중시",
]
for msg in test_users:
    profile = extract_profile(msg)
    cands = rule_based_filter(profile.risk_tolerance, etf_data)
    recs = llm_recommend(profile, cands, top_k=3)
    ws = analyze_risk(profile, recs, etf_data)
    print(f"\n[{profile.name}] {profile.risk_tolerance}/{profile.investment_goal}")
    for r in recs:
        print(f"  {r.get('allocation', '?')}% | {r['name']} — {r.get('reason', '')[:40]}")
    print(f"  경고 {len(ws)}개: {explain_risks(profile, ws)}")

# ============================================================
# ✅ 실습 7: 통합 파이프라인 + LLM 마크다운 리포트
# ============================================================
print("\n" + "=" * 60)
print("실습 7: 통합 파이프라인 + LLM 마크다운 리포트")
print("=" * 60)


def recommend_pipeline(user_text, top_k=3):
    """End-to-End 추천 파이프라인"""
    print(f"쿼리: {user_text}\n")

    # 1) 프로필 추출
    profile = extract_profile(user_text)
    print(f"1) 프로필: {profile.risk_tolerance} / {profile.investment_goal} / {profile.investment_horizon}년")

    # 2) 규칙 기반 필터 (위험도 맞지 않는 ETF 제거)
    candidates = rule_based_filter(profile.risk_tolerance, etf_data)
    print(f"2) 규칙 필터: {len(candidates)}개 통과")

    # 3) LLM 추천
    recs = llm_recommend(profile, candidates, top_k=top_k)
    print(f"3) 추천:")
    for r in recs:
        print(f"   - {r['name']} ({r.get('allocation', '?')}%)")

    # 4) 리스크 분석 + 설명
    warnings = analyze_risk(profile, recs, etf_data)
    risk_msg = explain_risks(profile, warnings)
    print(f"4) 리스크: {risk_msg}")

    return {
        "profile": asdict(profile),
        "recommendations": recs,
        "warnings": warnings,
        "risk_msg": risk_msg,
    }


def final_report(pipeline_result):
    """LLM 마크다운 리포트 생성"""
    p = pipeline_result["profile"]
    recs = pipeline_result["recommendations"]
    risk_msg = pipeline_result["risk_msg"]

    rec_text = "\n".join([
        f"| {r['name']} | {r.get('allocation', '?')}% | {r.get('reason', '')[:40]} |"
        for r in recs
    ])
    prompt = f"""ETF 추천 최종 리포트를 마크다운으로 작성하세요.
투자자: {p['name']} / {p['risk_tolerance']} / {p['investment_goal']} / {p['investment_horizon']}년 / 월 {p['monthly_budget']}만원

추천 포트폴리오:
| ETF | 비중 | 이유 |
|-----|------|------|
{rec_text}

리스크 메시지: {risk_msg}

구조:
1) 투자자 요약 (1줄)
2) 추천 포트폴리오 (테이블)
3) 리스크 설명
4) 다음 단계 (2줄)"""
    return llm.invoke([
        {"role": "system", "content": "전문 금융 어드바이저. 마크다운으로 작성."},
        {"role": "user", "content": prompt},
    ]).content


result = recommend_pipeline(
    "은퇴 준비 중인 55세입니다. 월 200만원 여유가 있고 안정적인 배당이 중요해요."
)
print("\n" + "=" * 60)
print(final_report(result))

# ============================================================
# ✅ Weekend 3 완료 요약
# ============================================================
print("\n" + "=" * 60)
print("Weekend 3 완료!")
print("=" * 60)
summary = [
    ("실습 1", "split vs Kiwi 정량 비교 — 형태소 분석이 BM25 성능 좌우"),
    ("실습 2", "쿼리 의도 분류(조건/키워드/의미) + 스마트 라우팅"),
    ("실습 3", "α 최적화 + FAISS/BM25/Hybrid 3방식 비교"),
    ("실습 4", "MAP@K + LLM-as-Judge 검색 품질 평가"),
    ("실습 5", "CBF 카테고리 다양성 + Cold-Start 폴백"),
    ("실습 6", "LLM 추천 + 포트폴리오 리스크 분석"),
    ("실습 7", "통합 파이프라인 + LLM 마크다운 리포트"),
]
for lab, desc in summary:
    print(f"  {lab}: {desc}")
