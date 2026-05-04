"""
Day 32 기본 MCP 서버 (FastMCP)
==========================================
ETF 상품 검색/조회/비교 도구를 제공하는 MCP 서버

실행:
    python servers/01_basic_server.py
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("etf-basic-server")

# ── ETF 데이터베이스 ──────────────────────────────────────────
ETF_DB = [
    {"name": "KODEX 200", "ticker": "069500", "category": "국내주식",
     "expense_ratio": 0.15, "return_1y": 8.5, "aum": 52000,
     "description": "KOSPI 200 지수를 추종하는 대표 ETF"},
    {"name": "KODEX S&P500TR", "ticker": "379800", "category": "해외주식",
     "expense_ratio": 0.05, "return_1y": 25.3, "aum": 38000,
     "description": "S&P 500 Total Return 지수 추종"},
    {"name": "ACE 미국배당다우존스", "ticker": "402970", "category": "배당",
     "expense_ratio": 0.01, "return_1y": 12.1, "aum": 21000,
     "description": "다우존스 미국 배당 100 지수 추종"},
    {"name": "TIGER 미국나스닥100", "ticker": "133690", "category": "해외주식",
     "expense_ratio": 0.07, "return_1y": 30.2, "aum": 45000,
     "description": "나스닥 100 지수를 추종하는 ETF"},
    {"name": "KODEX 미국S&P500TR", "ticker": "379810", "category": "해외주식",
     "expense_ratio": 0.05, "return_1y": 24.8, "aum": 32000,
     "description": "S&P 500 Total Return 원화 환산"},
    {"name": "TIGER 단기통안채", "ticker": "157450", "category": "채권",
     "expense_ratio": 0.03, "return_1y": 3.2, "aum": 15000,
     "description": "1년 이하 통안채 투자"},
    {"name": "KODEX 골드선물", "ticker": "132030", "category": "원자재",
     "expense_ratio": 0.68, "return_1y": 18.5, "aum": 8000,
     "description": "금 선물 가격을 추종"},
    {"name": "ACE KRX금현물", "ticker": "411060", "category": "원자재",
     "expense_ratio": 0.01, "return_1y": 20.1, "aum": 5000,
     "description": "KRX 금 현물 시장 추종"},
]


@mcp.tool()
def search_etf(category: str, min_return: float = 0.0) -> str:
    """ETF 상품을 카테고리와 최소 수익률로 검색합니다.

    Args:
        category: ETF 카테고리 (국내주식, 해외주식, 배당, 채권, 원자재)
        min_return: 최소 1년 수익률 (%, 기본값 0)
    """
    results = [
        etf for etf in ETF_DB
        if etf["category"] == category and etf["return_1y"] >= min_return
    ]
    if not results:
        return f"'{category}' 카테고리에서 수익률 {min_return}% 이상인 ETF가 없습니다."
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def get_etf_detail(name: str) -> str:
    """ETF 상품의 상세 정보를 조회합니다.

    Args:
        name: ETF 상품명 (예: KODEX 200)
    """
    for etf in ETF_DB:
        if etf["name"] == name:
            return json.dumps(etf, ensure_ascii=False, indent=2)
    return f"'{name}' ETF를 찾을 수 없습니다."


@mcp.tool()
def compare_etf(name1: str, name2: str) -> str:
    """두 ETF 상품을 비교합니다.

    Args:
        name1: 첫 번째 ETF 상품명
        name2: 두 번째 ETF 상품명
    """
    etf1 = next((e for e in ETF_DB if e["name"] == name1), None)
    etf2 = next((e for e in ETF_DB if e["name"] == name2), None)

    if not etf1:
        return f"'{name1}' ETF를 찾을 수 없습니다."
    if not etf2:
        return f"'{name2}' ETF를 찾을 수 없습니다."

    comparison = {
        "비교 항목": ["수익률(1Y)", "보수율", "운용자산(억)"],
        etf1["name"]: [f"{etf1['return_1y']}%", f"{etf1['expense_ratio']}%", f"{etf1['aum']}억"],
        etf2["name"]: [f"{etf2['return_1y']}%", f"{etf2['expense_ratio']}%", f"{etf2['aum']}억"],
    }
    return json.dumps(comparison, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()  # stdio transport (기본)
