# ============================================================
#  p2_weekend2_multi_llm_quality_260411_김민석.py
#  프로젝트 2 - Weekend 2: 리랭킹과 답변 품질 평가
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MODEL = "gpt-4o-mini"

print("✅ 환경 설정 완료")

# ============================================================
# 데이터 준비
# ============================================================
SAMPLE_ETF_DATA = [
    {"ticker": "KODEX200", "name": "KODEX 200", "category": "국내주식",
     "description": "KOSPI 200 지수를 추종하는 국내 대표 ETF. 삼성전자, SK하이닉스 등 대형주 중심.",
     "expense_ratio": 0.15, "aum_billion": 58000, "risk_level": "중간"},
    {"ticker": "TIGER미국S&P500", "name": "TIGER 미국 S&P500", "category": "해외주식",
     "description": "미국 S&P 500 지수를 추종. 애플, 마이크로소프트 등 미국 대형주 투자.",
     "expense_ratio": 0.07, "aum_billion": 45000, "risk_level": "중간"},
    {"ticker": "KODEX배당가치", "name": "KODEX 배당가치", "category": "배당",
     "description": "고배당 가치주 중심 ETF. 은행, 통신 등 배당 수익률이 높은 종목.",
     "expense_ratio": 0.30, "aum_billion": 8500, "risk_level": "낮음"},
    {"ticker": "TIGER반도체", "name": "TIGER 반도체", "category": "섹터",
     "description": "국내 반도체 산업 ETF. 삼성전자, SK하이닉스 등 반도체 관련주.",
     "expense_ratio": 0.40, "aum_billion": 12000, "risk_level": "높음"},
    {"ticker": "KODEX국고채3년", "name": "KODEX 국고채 3년", "category": "채권",
     "description": "한국 3년 만기 국고채에 투자하는 안정형 ETF.",
     "expense_ratio": 0.05, "aum_billion": 25000, "risk_level": "매우낮음"},
    {"ticker": "TIGER차이나CSI300", "name": "TIGER 차이나 CSI300", "category": "해외주식",
     "description": "중국 CSI 300 지수 추종. 상해/심천 대형주 투자.",
     "expense_ratio": 0.25, "aum_billion": 3500, "risk_level": "높음"},
    {"ticker": "KODEX골드선물", "name": "KODEX 골드선물(H)", "category": "원자재",
     "description": "금 선물 가격을 추종하는 원자재 ETF. 인플레이션 헤지 수단.",
     "expense_ratio": 0.68, "aum_billion": 5200, "risk_level": "중간"},
    {"ticker": "TIGER리츠부동산", "name": "TIGER 리츠부동산인프라", "category": "부동산",
     "description": "국내 리츠 및 부동산 인프라 기업에 투자. 배당 수익 추구.",
     "expense_ratio": 0.29, "aum_billion": 4100, "risk_level": "중간"},
    {"ticker": "KODEX2차전지", "name": "KODEX 2차전지산업", "category": "섹터",
     "description": "2차전지 관련 기업에 투자. LG에너지솔루션, 삼성SDI 등.",
     "expense_ratio": 0.45, "aum_billion": 18000, "risk_level": "높음"},
    {"ticker": "TIGER단기통안채", "name": "TIGER 단기통안채", "category": "채권",
     "description": "초단기 통안채에 투자. 파킹 용도로 활용되는 안전 자산.",
     "expense_ratio": 0.03, "aum_billion": 32000, "risk_level": "매우낮음"},
]

EVAL_QUERIES = [
    {"query": "안정적인 배당 ETF를 추천해주세요",
     "reference": "KODEX 배당가치 ETF를 추천합니다. 은행, 통신 등 고배당 가치주에 투자하며 리스크가 낮습니다. 수수료는 0.30%입니다.",
     "relevant_etfs": ["KODEX배당가치", "TIGER리츠부동산"]},
    {"query": "미국 주식에 투자하고 싶어요",
     "reference": "TIGER 미국 S&P500 ETF를 추천합니다. 애플, 마이크로소프트 등 미국 대형주에 투자하며 수수료가 0.07%로 저렴합니다.",
     "relevant_etfs": ["TIGER미국S&P500"]},
    {"query": "원금 손실 위험이 적은 ETF는?",
     "reference": "TIGER 단기통안채 ETF를 추천합니다. 초단기 통안채에 투자하여 원금 손실 위험이 매우 낮으며 수수료도 0.03%입니다.",
     "relevant_etfs": ["KODEX국고채3년", "TIGER단기통안채"]},
    {"query": "반도체 섹터에 투자하려면?",
     "reference": "TIGER 반도체 ETF를 추천합니다. 삼성전자, SK하이닉스 등 반도체 관련주에 집중 투자합니다. 리스크가 높으니 주의하세요.",
     "relevant_etfs": ["TIGER반도체"]},
    {"query": "인플레이션 헤지 방법이 있을까요?",
     "reference": "KODEX 골드선물(H) ETF를 고려해보세요. 금 선물 가격을 추종하여 인플레이션 헤지 수단으로 활용됩니다.",
     "relevant_etfs": ["KODEX골드선물"]},
    {"query": "중국 시장에 투자하는 ETF는?",
     "reference": "TIGER 차이나 CSI300 ETF를 추천합니다. 중국 CSI 300 지수를 추종하며 상해/심천 대형주에 투자합니다.",
     "relevant_etfs": ["TIGER차이나CSI300"]},
]

print(f"📦 ETF 데이터 로드 완료: {len(SAMPLE_ETF_DATA)}개 상품, {len(EVAL_QUERIES)}개 평가 질의")

# ============================================================
# 공통 설정: 벡터 스토어, BM25, 하이브리드 검색
# ============================================================
os.makedirs("project2_data/raw", exist_ok=True)
os.makedirs("project2_data/evaluation", exist_ok=True)
os.makedirs("project2_data/checkpoints", exist_ok=True)

vs_path = "project2_data/vectorstore/faiss_baseline"
if os.path.exists(vs_path):
    vs = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
    with open("project2_data/raw/etf_documents.json") as f:
        etf_docs = json.load(f)
    print(f"✅ 체크포인트 복원: 벡터 스토어 {vs.index.ntotal}개 문서")
else:
    docs = []
    etf_docs = {}
    for etf in SAMPLE_ETF_DATA:
        content = (
            f"{etf['name']} ({etf['ticker']}): {etf['description']} "
            f"카테고리: {etf['category']}, 수수료: {etf['expense_ratio']}%, 리스크: {etf['risk_level']}"
        )
        docs.append(Document(page_content=content, metadata={"ticker": etf["ticker"]}))
        etf_docs[etf["ticker"]] = {"name": etf["name"], "content": content, "category": etf["category"]}
    vs = FAISS.from_documents(docs, embeddings)
    print(f"✅ 새로 구축: 벡터 스토어 {vs.index.ntotal}개 문서")

all_contents = [etf_docs[k]["content"] for k in etf_docs]
all_keys = list(etf_docs.keys())
bm25 = BM25Okapi([doc.split() for doc in all_contents])


def hybrid_search(query, k=5, alpha=0.5):
    """BM25 + 벡터 하이브리드 검색 (RRF)"""
    vec_results = vs.similarity_search_with_score(query, k=k)
    vec_ids = [(doc.metadata.get("ticker", ""), score) for doc, score in vec_results]

    bm25_scores = bm25.get_scores(query.split())
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k]
    bm25_ids = [(all_keys[idx], score) for idx, score in bm25_ranked]

    rrf_scores = {}
    for rank, (doc_id, _) in enumerate(vec_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (rank + 60)
    for rank, (doc_id, _) in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (rank + 60)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(doc_id, rrf_scores[doc_id], etf_docs.get(doc_id, {}).get("content", "")) for doc_id, _ in sorted_results]


print("✅ hybrid_search 준비 완료")

# ============================================================
# ✅ 실습 1: Hit Rate 계산
# ============================================================
print("\n" + "=" * 60)
print("실습 1: Hit Rate 계산")
print("=" * 60)


def hit_rate(eval_data, search_fn, k=5):
    hits = 0
    for item in eval_data:
        results = search_fn(item["query"], k=k)
        found_ids = [r[0] for r in results]
        if any(etf in found_ids for etf in item["relevant_etfs"]):
            hits += 1
    return hits / len(eval_data)


for k in [3, 5]:
    hr = hit_rate(EVAL_QUERIES, hybrid_search, k=k)
    print(f"Hit Rate@{k}: {hr:.3f}")

# ============================================================
# ✅ 실습 2: RAG 답변 생성과 temperature 비교
# ============================================================
print("\n" + "=" * 60)
print("실습 2: RAG 답변 생성과 temperature 비교")
print("=" * 60)


def ask_etf(query, k=3, verbose=False):
    """ETF 추천 RAG 파이프라인"""
    results = hybrid_search(query, k=k)
    context = "\n\n".join([f"[{doc_id}] {content}" for doc_id, score, content in results])
    _llm = ChatOpenAI(model=MODEL, temperature=0)
    answer = _llm.invoke([
        SystemMessage(content="ETF 전문가입니다. 검색된 문서만을 근거로 답변하세요. 추측하지 마세요."),
        HumanMessage(content=f"참고 문서:\n{context}\n\n질문: {query}")
    ]).content
    if verbose:
        print(f"Q: {query}\nA: {answer[:200]}...")
    return answer


# 전체 답변 생성 (이후 실습에서 재사용)
answers = {}
for item in EVAL_QUERIES:
    answers[item["query"]] = ask_etf(item["query"])
print(f"✅ {len(answers)}개 답변 생성 완료")


def ask_etf_v2(query, k=3, temperature=0, max_tokens=500):
    results = hybrid_search(query, k=k)
    context = "\n".join([f"[{did}] {c}" for did, _, c in results])
    start = time.time()
    _llm = ChatOpenAI(model=MODEL, temperature=temperature, max_tokens=max_tokens)
    answer = _llm.invoke([
        SystemMessage(content="ETF 전문가입니다."),
        HumanMessage(content=f"문서:\n{context}\n\n질문: {query}")
    ]).content
    elapsed = time.time() - start
    return answer, elapsed


q = "안정적인 배당 ETF를 추천해주세요"
a0, t0 = ask_etf_v2(q, temperature=0)
a7, t7 = ask_etf_v2(q, temperature=0.7)
print(f"temp=0  : {len(a0)}자, {t0:.2f}초")
print(f"temp=0.7: {len(a7)}자, {t7:.2f}초")

# ============================================================
# ✅ 실습 3: 리랭킹 전후 Hit Rate 비교
# ============================================================
print("\n" + "=" * 60)
print("실습 3: 리랭킹 전후 Hit Rate 비교")
print("=" * 60)


def llm_rerank(query, documents, top_k=3):
    """LLM으로 검색 결과를 질의 관련성 순으로 재정렬"""
    doc_list = "\n".join([
        f"[{i}] {doc_id}: {content[:150]}"
        for i, (doc_id, score, content) in enumerate(documents)
    ])

    _llm = ChatOpenAI(model=MODEL, temperature=0).bind(
        response_format={"type": "json_object"}
    )
    response = _llm.invoke([
        SystemMessage(content="ETF 검색 결과를 질의 관련성 순으로 재정렬하는 전문가입니다."),
        HumanMessage(content=f"""질문: {query}

검색 결과:
{doc_list}

위 검색 결과를 질문과의 관련성 순으로 정렬하세요.
각 문서에 1-10점 관련성 점수를 부여하세요.
JSON 형식으로 응답: {{"rankings": [{{"index": 0, "score": 9, "reason": "이유"}}]}}""")
    ])

    try:
        result = json.loads(response.content)
        rankings = result if isinstance(result, list) else result.get("rankings", result.get("results", []))
        rankings.sort(key=lambda x: x.get("score", 0), reverse=True)
        reranked = []
        for r in rankings[:top_k]:
            idx = r["index"]
            if 0 <= idx < len(documents):
                doc_id, _, content = documents[idx]
                reranked.append((doc_id, r["score"], content))
        return reranked
    except Exception as e:
        print(f"⚠️ 리랭킹 실패: {e}")
        return documents[:top_k]


def compare_reranking(eval_data, k_initial=7, k_rerank=3):
    hit_before, hit_after = 0, 0
    total_time = 0

    for item in eval_data:
        initial = hybrid_search(item["query"], k=k_initial)
        ids_before = [r[0] for r in initial[:k_rerank]]

        start = time.time()
        reranked = llm_rerank(item["query"], initial, top_k=k_rerank)
        total_time += time.time() - start
        ids_after = [r[0] for r in reranked]

        if any(e in ids_before for e in item["relevant_etfs"]):
            hit_before += 1
        if any(e in ids_after for e in item["relevant_etfs"]):
            hit_after += 1

        print(f"Q: {item['query'][:25]}... | before={ids_before} | after={ids_after}")

    n = len(eval_data)
    print(f"\nHit Rate@{k_rerank}: {hit_before/n:.3f} → {hit_after/n:.3f}")
    print(f"평균 리랭킹 시간: {total_time/n:.2f}초")


compare_reranking(EVAL_QUERIES)

# ============================================================
# ✅ 실습 4: 스코어 필터링 방법 비교
# ============================================================
print("\n" + "=" * 60)
print("실습 4: 스코어 필터링 방법 비교")
print("=" * 60)


def score_filter(results, method="dynamic"):
    """스코어 기반 필터링"""
    if not results:
        return results

    scores = [r[1] for r in results]

    if method == "fixed":
        threshold = 5.0
    elif method == "dynamic":
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        threshold = mean_s - std_s
    elif method == "gap":
        gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        if gaps:
            max_gap_idx = np.argmax(gaps)
            threshold = scores[max_gap_idx + 1] + 0.01
        else:
            threshold = 0
    else:
        threshold = 0

    filtered = [(doc_id, score, content) for doc_id, score, content in results if score >= threshold]
    return filtered if filtered else results[:1]


for method in ["fixed", "dynamic", "gap"]:
    total_docs = 0
    for item in EVAL_QUERIES:
        initial = hybrid_search(item["query"], k=7)
        reranked = llm_rerank(item["query"], initial, top_k=5)
        filtered = score_filter(reranked, method=method)
        total_docs += len(filtered)
    avg_docs = total_docs / len(EVAL_QUERIES)
    print(f"{method:8s}: 평균 {avg_docs:.1f}개 문서 통과")

# ============================================================
# ✅ 실습 5: Few-shot 프롬프트 설계
# ============================================================
print("\n" + "=" * 60)
print("실습 5: Few-shot 프롬프트 설계")
print("=" * 60)

fewshot_prompt = ChatPromptTemplate.from_messages([
    ("system", "ETF 전문가입니다. 아래 예시처럼 답변하세요."),
    ("human", """예시:
Q: 안전한 투자를 원합니다
A: ## 추천 ETF
- TIGER 단기통안채: 수수료 0.03%, 매우낮은 리스크
## 추천 이유
- 초단기 국채 투자로 원금 보존
## 주의사항
- 기대수익률이 낮을 수 있음

---
참고 문서:
{context}

질문: {query}""")
])

results = hybrid_search("안정적인 배당 ETF를 추천해주세요", k=3)
context = "\n".join([f"[{did}] {c}" for did, _, c in results])
msgs = fewshot_prompt.format_messages(context=context, query="안정적인 배당 ETF를 추천해주세요")
resp = llm.invoke(msgs)
print(resp.content[:400])

# ============================================================
# ✅ 실습 6: BLEU 스코어 계산
# ============================================================
print("\n" + "=" * 60)
print("실습 6: BLEU 스코어 계산")
print("=" * 60)


def get_ngrams(tokens, n):
    """토큰 리스트에서 n-gram 추출"""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def modified_precision(ref_tokens, cand_tokens, n):
    """Clipped n-gram Precision"""
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    cand_ngrams = Counter(get_ngrams(cand_tokens, n))
    clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
    total = sum(cand_ngrams.values())
    return clipped / total if total > 0 else 0.0


def brevity_penalty(ref_len, cand_len):
    """Brevity Penalty"""
    if cand_len >= ref_len:
        return 1.0
    return math.exp(1 - ref_len / cand_len)


def compute_bleu(reference, candidate, max_n=4):
    """BLEU Score 계산"""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    bp = brevity_penalty(len(ref_tokens), len(cand_tokens))
    precisions = []
    for n in range(1, max_n + 1):
        p = modified_precision(ref_tokens, cand_tokens, n)
        precisions.append(p)
    log_avg = 0.0
    for p in precisions:
        if p == 0:
            return 0.0
        log_avg += (1.0 / max_n) * math.log(p)
    return bp * math.exp(log_avg)


rows = []
for item in EVAL_QUERIES:
    answer = answers.get(item["query"], ask_etf(item["query"]))
    ref = item["reference"]
    b1 = compute_bleu(ref, answer, max_n=1)
    b2 = compute_bleu(ref, answer, max_n=2)
    b4 = compute_bleu(ref, answer, max_n=4)
    rows.append({"query": item["query"][:25], "BLEU-1": b1, "BLEU-2": b2, "BLEU-4": b4})

df_bleu = pd.DataFrame(rows)
print(df_bleu.to_string(index=False))
print(f"\n평균: BLEU-1={df_bleu['BLEU-1'].mean():.4f}, BLEU-4={df_bleu['BLEU-4'].mean():.4f}")

# ============================================================
# ✅ 실습 7: BLEU와 ROUGE 비교
# ============================================================
print("\n" + "=" * 60)
print("실습 7: BLEU와 ROUGE 비교")
print("=" * 60)


def rouge_n(reference, candidate, n=1):
    """ROUGE-N: Recall, Precision, F1"""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    cand_ngrams = Counter(get_ngrams(cand_tokens, n))
    overlap = sum(min(ref_ngrams[ng], cand_ngrams.get(ng, 0)) for ng in ref_ngrams)
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
    precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def lcs_length(seq1, seq2):
    """최장 공통 부분수열 길이 (DP)"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(reference, candidate):
    """ROUGE-L: LCS 기반 F1"""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    lcs = lcs_length(ref_tokens, cand_tokens)
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "lcs": lcs}


rows = []
for item in EVAL_QUERIES:
    a = answers.get(item["query"], ask_etf(item["query"]))
    b4 = compute_bleu(item["reference"], a, max_n=4)
    r1 = rouge_n(item["reference"], a, 1)["f1"]
    r2 = rouge_n(item["reference"], a, 2)["f1"]
    rl = rouge_l(item["reference"], a)["f1"]
    rows.append({"query": item["query"][:20], "BLEU-4": b4, "R-1": r1, "R-2": r2, "R-L": rl})

df_rouge = pd.DataFrame(rows)
print(df_rouge.to_string(index=False))
print(f"\nBLEU-4 vs ROUGE-1 상관계수: {df_rouge['BLEU-4'].corr(df_rouge['R-1']):.3f}")

# ============================================================
# ✅ 실습 8: BERTScore와 상관관계 분석
# ============================================================
print("\n" + "=" * 60)
print("실습 8: BERTScore와 상관관계 분석")
print("=" * 60)


def get_embedding(text):
    """OpenAI 임베딩 벡터"""
    return np.array(embeddings.embed_query(text))


def cosine_sim(a, b):
    """코사인 유사도"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def simple_bertscore(reference, candidate):
    """문장 임베딩 기반 BERTScore (간이)"""
    ref_emb = get_embedding(reference)
    cand_emb = get_embedding(candidate)
    return cosine_sim(ref_emb, cand_emb)


rows = []
for item in EVAL_QUERIES:
    a = answers.get(item["query"], ask_etf(item["query"]))
    b4 = compute_bleu(item["reference"], a, max_n=4)
    r1 = rouge_n(item["reference"], a, 1)["f1"]
    bs = simple_bertscore(item["reference"], a)
    rows.append({"query": item["query"][:20], "BLEU-4": b4, "ROUGE-1": r1, "BERTScore": bs})

df_bert = pd.DataFrame(rows)
print(df_bert.to_string(index=False))
print(f"\nBLEU-ROUGE 상관: {df_bert['BLEU-4'].corr(df_bert['ROUGE-1']):.3f}")
print(f"BLEU-BERT 상관:  {df_bert['BLEU-4'].corr(df_bert['BERTScore']):.3f}")
print(f"ROUGE-BERT 상관: {df_bert['ROUGE-1'].corr(df_bert['BERTScore']):.3f}")

# ============================================================
# ✅ 실습 9: 한국어 금융 토크나이저 개선
# ============================================================
print("\n" + "=" * 60)
print("실습 9: 한국어 금융 토크나이저 개선")
print("=" * 60)


def korean_financial_tokenize(text):
    """금융 도메인 특화 한국어 토큰 분리"""
    words = text.split()
    tokens = []
    particles = ['으로', '에서', '부터', '까지', '에게',
                 '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '로']

    for word in words:
        # 숫자+단위 분리: "0.15%", "58000억"
        m = re.match(r'^([\d,.]+)(%|억|만|원|개|배)(.*)$', word)
        if m:
            tokens.extend([m.group(1), m.group(2)])
            if m.group(3):
                tokens.append(m.group(3))
            continue

        # 영문+한글 분리: "ETF를"
        m = re.match(r'^([A-Za-z0-9&]+)([가-힣]+)$', word)
        if m:
            tokens.append(m.group(1))
            rest = m.group(2)
            separated = False
            for p in sorted(particles, key=len, reverse=True):
                if rest == p:
                    tokens.append(p)
                    separated = True
                    break
                if rest.endswith(p) and len(rest) > len(p):
                    tokens.append(rest[:-len(p)])
                    tokens.append(p)
                    separated = True
                    break
            if not separated:
                tokens.append(rest)
            continue

        # 일반 한국어 조사 분리
        separated = False
        for p in sorted(particles, key=len, reverse=True):
            if word.endswith(p) and len(word) > len(p):
                tokens.append(word[:-len(p)])
                tokens.append(p)
                separated = True
                break
        if not separated:
            tokens.append(word)

    return tokens


FINANCE_TERMS = {"2차전지", "S&P500", "CSI300", "KOSPI200", "KODEX200"}


def korean_financial_tokenize_v2(text):
    """전문 용어를 보존하는 개선된 토크나이저"""
    words = text.split()
    tokens = []
    for word in words:
        if word in FINANCE_TERMS:
            tokens.append(word)
        else:
            tokens.extend(korean_financial_tokenize(word))
    return tokens


test = "2차전지 ETF와 S&P500 ETF 비교"
print(f"원문: {test}")
print(f"기존: {korean_financial_tokenize(test)}")
print(f"개선: {korean_financial_tokenize_v2(test)}")

# ============================================================
# ✅ 실습 10: LLM-as-Judge 배치 평가
# ============================================================
print("\n" + "=" * 60)
print("실습 10: LLM-as-Judge 배치 평가")
print("=" * 60)


def llm_judge(query, answer, context, criteria=None):
    """LLM-as-Judge: 답변 품질 평가"""
    if criteria is None:
        criteria = {
            "정확성": "답변이 참고 문서의 사실과 일치하는가? (1-5)",
            "관련성": "질문의 의도와 범위에 적합한 답변인가? (1-5)",
            "완전성": "핵심 정보(ETF명, 수수료, 리스크 등)를 포함하는가? (1-5)",
            "안전성": "투자 위험을 적절히 고지하고 있는가? (1-5)",
            "명확성": "이해하기 쉽고 구조적인가? (1-5)",
        }

    criteria_text = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])

    _llm = ChatOpenAI(model=MODEL, temperature=0).bind(
        response_format={"type": "json_object"}
    )
    response = _llm.invoke([
        SystemMessage(content="금융 ETF 답변 품질 평가 전문가입니다. JSON으로 응답하세요."),
        HumanMessage(content=f"""다음 ETF 추천 답변을 평가해주세요.

질문: {query}
참고 문서: {context[:500]}
답변: {answer}

평가 기준:
{criteria_text}

JSON 형식: {{"scores": {{"정확성": 4, "관련성": 4, "완전성": 3, "안전성": 4, "명확성": 4}}, "총점": 19, "피드백": "..."}}""")
    ])

    try:
        return json.loads(response.content)
    except Exception:
        return {"error": "파싱 실패"}


all_judgments = []
for item in EVAL_QUERIES:
    a = answers.get(item["query"], ask_etf(item["query"]))
    results = hybrid_search(item["query"], k=3)
    ctx = "\n".join([c for _, _, c in results])
    j = llm_judge(item["query"], a, ctx)
    j["query"] = item["query"][:25]
    all_judgments.append(j)
    print(f"Q: {item['query'][:25]}... → 총점: {j.get('총점', 'N/A')}")

# 기준별 평균
criteria_avg = {}
for j in all_judgments:
    for k, v in j.get("scores", {}).items():
        criteria_avg.setdefault(k, []).append(v)

print("\n기준별 평균 점수:")
for k, vals in criteria_avg.items():
    print(f"  {k}: {np.mean(vals):.1f}")

# ============================================================
# ✅ 실습 11: Criteria 평가 레이더 차트
# ============================================================
print("\n" + "=" * 60)
print("실습 11: Criteria 평가 레이더 차트")
print("=" * 60)


def criteria_evaluation(query, answer, context):
    """금융 도메인 특화 Criteria 평가"""
    criteria = {
        "사실_정확성": {"desc": "답변이 참고 문서의 사실과 일치하는가?", "weight": 0.30},
        "투자_적합성": {"desc": "추천이 질문자의 투자 목적에 적합한가?", "weight": 0.25},
        "리스크_고지": {"desc": "투자 위험을 적절히 고지하고 있는가?", "weight": 0.20},
        "정보_완전성": {"desc": "수수료, 수익률 등 필요 정보가 포함되었는가?", "weight": 0.15},
        "표현_명확성": {"desc": "전문 용어를 이해하기 쉽게 설명하는가?", "weight": 0.10},
    }

    _llm = ChatOpenAI(model=MODEL, temperature=0).bind(
        response_format={"type": "json_object"}
    )
    results_dict = {}
    for name, info in criteria.items():
        response = _llm.invoke([
            SystemMessage(content="금융 답변 품질 평가 전문가입니다. JSON으로 응답하세요."),
            HumanMessage(content=f"""기준: {info['desc']}

질문: {query}
참고 문서: {context[:400]}
답변: {answer}

1-5점으로 평가하세요.
JSON: {{"score": 4, "reason": "이유"}}""")
        ])
        try:
            r = json.loads(response.content)
            results_dict[name] = {"score": r["score"], "reason": r.get("reason", ""), "weight": info["weight"]}
        except Exception:
            results_dict[name] = {"score": 0, "reason": "파싱 실패", "weight": info["weight"]}

    weighted_total = sum(r["score"] * r["weight"] for r in results_dict.values())
    normalized = weighted_total / 5.0 * 100

    return {"criteria": results_dict, "weighted_score": weighted_total, "normalized": normalized}


# 레이더 차트 시각화
labels = ["사실_정확성", "투자_적합성", "리스크_고지", "정보_완전성", "표현_명확성"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

avg_scores = [np.mean(criteria_avg.get(label, [3])) for label in labels]
values = avg_scores + avg_scores[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values, 'o-', linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 5)
plt.title("Criteria 평가 레이더 차트")
plt.tight_layout()
plt.savefig("project2_data/evaluation/criteria_radar.png", dpi=100, bbox_inches="tight")
plt.show()
print("✅ 레이더 차트 저장 완료")

# ============================================================
# ✅ 실습 12: 통합 평가 파이프라인 실행
# ============================================================
print("\n" + "=" * 60)
print("실습 12: 통합 평가 파이프라인 실행")
print("=" * 60)


class ETFEvaluationPipeline:
    """ETF 추천 답변 종합 평가 파이프라인"""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {
            "ROUGE-1": 0.3,
            "BERTScore": 0.7,
            "LLM_Judge": 15,
        }
        self.results = []

    def evaluate_single(self, query, answer, reference, context):
        """단일 답변 종합 평가"""
        bleu4 = compute_bleu(reference, answer, max_n=4)
        r1 = rouge_n(reference, answer, 1)["f1"]
        r2 = rouge_n(reference, answer, 2)["f1"]
        rl = rouge_l(reference, answer)["f1"]
        bs = simple_bertscore(reference, answer)

        judge = llm_judge(query, answer, context)
        judge_total = judge.get("총점", 0)

        gate_results = {
            "ROUGE-1": r1 >= self.thresholds["ROUGE-1"],
            "BERTScore": bs >= self.thresholds["BERTScore"],
            "LLM_Judge": judge_total >= self.thresholds["LLM_Judge"],
        }

        critical_fail = not gate_results["BERTScore"]
        all_pass = all(gate_results.values())

        if critical_fail:
            status = "🔴 FAIL"
        elif not all_pass:
            status = "🟡 WARN"
        else:
            status = "🟢 PASS"

        result = {
            "query": query[:30], "BLEU-4": bleu4,
            "ROUGE-1": r1, "ROUGE-2": r2, "ROUGE-L": rl,
            "BERTScore": bs, "LLM_Judge": judge_total,
            "status": status, "gate": gate_results,
        }
        self.results.append(result)
        return result

    def evaluate_batch(self, eval_data, answer_fn):
        """배치 평가"""
        for item in eval_data:
            answer = answer_fn(item["query"])
            context = "\n".join([c for _, _, c in hybrid_search(item["query"], k=3)])
            self.evaluate_single(item["query"], answer, item["reference"], context)
        return self

    def summary(self):
        """평가 요약"""
        df = pd.DataFrame(self.results)
        print("=== 종합 평가 결과 ===")
        print(df[["query", "BLEU-4", "ROUGE-1", "BERTScore", "LLM_Judge", "status"]].to_string(index=False))
        print(f"\n🟢 PASS: {sum(1 for r in self.results if '🟢' in r['status'])}")
        print(f"🟡 WARN: {sum(1 for r in self.results if '🟡' in r['status'])}")
        print(f"🔴 FAIL: {sum(1 for r in self.results if '🔴' in r['status'])}")
        return df


pipeline = ETFEvaluationPipeline()
pipeline.evaluate_batch(EVAL_QUERIES, lambda q: answers.get(q, ask_etf(q)))
df_pipeline = pipeline.summary()

os.makedirs("project2_data/evaluation", exist_ok=True)
df_pipeline.drop(columns=["gate"], errors="ignore").to_csv(
    "project2_data/evaluation/evaluation_results.csv", index=False, encoding="utf-8-sig"
)
print("\n✅ evaluation_results.csv 저장 완료")

# ============================================================
# ✅ 실습 14: 체크포인트 저장과 검증
# ============================================================
print("\n" + "=" * 60)
print("실습 14: 체크포인트 저장과 검증")
print("=" * 60)

weekend2_results = {
    "timestamp": datetime.now().isoformat(),
    "version": "2.0",
    "evaluation_metrics": ["BLEU", "ROUGE", "BERTScore", "LLM-as-Judge", "Criteria"],
    "pipeline_config": {
        "search_k": 7,
        "rerank_k": 3,
        "filter_method": "dynamic",
        "model": "gpt-4o-mini",
    },
    "quality_gate_thresholds": {
        "ROUGE-1": 0.3,
        "BERTScore": 0.7,
        "LLM_Judge": 15,
    },
}

with open("project2_data/checkpoints/weekend2_results.json", "w") as f:
    json.dump(weekend2_results, f, ensure_ascii=False, indent=2)

with open("project2_data/checkpoints/weekend2_answers.json", "w") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

weekend2_state = {
    "weekend": 2,
    "completed": True,
    "achievements": [
        "LLM 기반 리랭킹 파이프라인 구축",
        "스코어 필터링 (고정/동적/갭) 구현",
        "프롬프트 최적화 (기본/최적화/Few-shot)",
        "BLEU, ROUGE, BERTScore 자동 평가",
        "한국어 금융 도메인 토큰화",
        "LLM-as-Judge 평가 파이프라인",
        "Criteria 기반 다차원 평가",
        "종합 평가 파이프라인 + 품질 게이트",
        "Gradio 평가 대시보드",
    ],
}

with open("project2_data/checkpoints/weekend2_state.json", "w") as f:
    json.dump(weekend2_state, f, ensure_ascii=False, indent=2)

print("✅ Weekend 2 체크포인트 저장 완료")
for f_name in ["weekend2_results.json", "weekend2_answers.json", "weekend2_state.json"]:
    print(f"  📁 project2_data/checkpoints/{f_name}")

# 체크리스트 검증
checklist = [
    ("벡터 스토어", os.path.exists("project2_data/vectorstore/faiss_baseline")),
    ("ETF 문서", os.path.exists("project2_data/raw/etf_documents.json")),
    ("평가 결과 CSV", os.path.exists("project2_data/evaluation/evaluation_results.csv")),
    ("Criteria 레이더 차트", os.path.exists("project2_data/evaluation/criteria_radar.png")),
    ("Weekend 2 결과", os.path.exists("project2_data/checkpoints/weekend2_results.json")),
    ("Weekend 2 답변 캐시", os.path.exists("project2_data/checkpoints/weekend2_answers.json")),
    ("Weekend 2 상태", os.path.exists("project2_data/checkpoints/weekend2_state.json")),
]

print("\n📋 체크리스트:")
for item_name, exists in checklist:
    print(f"  {'✅' if exists else '❌'} {item_name}")

# ============================================================
# ✅ 실습 13: 평가 대시보드 확장 (Gradio)
# ============================================================
print("\n" + "=" * 60)
print("실습 13: 평가 대시보드 확장")
print("=" * 60)


def evaluate_query_extended(query, reference=""):
    """질의 평가 전체 파이프라인 (Criteria 포함)"""
    initial = hybrid_search(query, k=7)
    reranked = llm_rerank(query, initial, top_k=3)
    filtered = score_filter(reranked, method="dynamic")

    context = "\n".join([f"[{did}] {c}" for did, _, c in filtered])
    _llm = ChatOpenAI(model=MODEL, temperature=0)
    answer = _llm.invoke([
        SystemMessage(content="ETF 전문가입니다. 검색된 문서만을 근거로 답변하세요."),
        HumanMessage(content=f"참고 문서:\n{context}\n\n질문: {query}")
    ]).content

    search_info = "📋 검색 결과:\n"
    for did, score, content in filtered:
        search_info += f"  [{did}] score={score:.4f}\n"

    metrics_info = ""
    if reference.strip():
        b4 = compute_bleu(reference, answer, max_n=4)
        r1 = rouge_n(reference, answer, 1)["f1"]
        rl = rouge_l(reference, answer)["f1"]
        bs = simple_bertscore(reference, answer)
        metrics_info = (
            f"📊 자동 평가:\n"
            f"  BLEU-4: {b4:.4f}\n"
            f"  ROUGE-1: {r1:.4f}\n"
            f"  ROUGE-L: {rl:.4f}\n"
            f"  BERTScore: {bs:.4f}"
        )
        gate = "🟢 PASS" if bs >= 0.7 else "🔴 FAIL"
        metrics_info += f"\n\n품질 게이트(BERTScore): {gate}"
    else:
        metrics_info = "📊 참조 답변을 입력하면 자동 평가가 표시됩니다."

    judge = llm_judge(query, answer, context)
    judge_total = judge.get("총점", 0)
    judge_info = (
        f"🧑‍⚖️ LLM-as-Judge:\n"
        f"  총점: {judge_total}/25\n"
        f"  피드백: {judge.get('피드백', 'N/A')[:200]}"
    )
    judge_info += "\n  게이트: 🟢 PASS" if judge_total >= 15 else "\n  게이트: 🔴 FAIL"

    crit = criteria_evaluation(query, answer, context)
    crit_info = "📋 Criteria 평가:\n"
    for name, r in crit["criteria"].items():
        crit_info += f"  {name}: {r['score']}/5 — {r['reason'][:40]}\n"
    crit_info += f"\n  가중 점수: {crit['weighted_score']:.2f}/5.00 ({crit['normalized']:.0f}점)"

    return answer, search_info, metrics_info, judge_info, crit_info


demo_ext = gr.Interface(
    fn=evaluate_query_extended,
    inputs=[
        gr.Textbox(label="질문", placeholder="ETF 관련 질문을 입력하세요"),
        gr.Textbox(label="참조 답변 (선택)", placeholder="자동 평가를 위한 정답"),
    ],
    outputs=[
        gr.Textbox(label="📝 생성된 답변"),
        gr.Textbox(label="🔍 검색 결과"),
        gr.Textbox(label="📊 자동 메트릭"),
        gr.Textbox(label="🧑‍⚖️ LLM 평가"),
        gr.Textbox(label="📋 Criteria 평가"),
    ],
    title="ETF 추천 + 품질 평가 대시보드 (확장)",
    examples=[
        ["안정적인 배당 ETF를 추천해주세요",
         "KODEX 배당가치 ETF를 추천합니다. 고배당 가치주에 투자하며 리스크가 낮습니다."],
        ["미국 주식에 투자하고 싶어요", ""],
        ["원금 손실 위험이 적은 ETF는?", ""],
    ]
)

demo_ext.launch(share=False)
