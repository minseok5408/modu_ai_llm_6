"""
Day 32 기본 MCP 클라이언트
=====================================
stdio 방식으로 MCP 서버에 연결하여 도구 목록 조회/호출

실행:
    python clients/01_basic_client.py

서버(servers/01_basic_server.py)를 자동으로 시작합니다.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # ── 1. 서버 연결 설정 (stdio — 서버를 자동으로 시작) ────────
    server_params = StdioServerParameters(
        command="python",
        args=["servers/01_basic_server.py"],
    )

    print("=" * 55)
    print("MCP 클라이언트 시작 (stdio transport)")
    print("서버: servers/01_basic_server.py")
    print("=" * 55)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # ── 2. 초기화 핸드셰이크 ──────────────────────────
            await session.initialize()
            print("\n✅ 서버 연결 및 초기화 완료")

            # ── 3. 도구 목록 조회 (tools/list) ─────────────────
            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"\n📦 사용 가능한 도구: {len(tools)}개")
            print("-" * 40)
            for tool in tools:
                print(f"  🔧 {tool.name}")
                print(f"     설명: {tool.description}")
                if tool.inputSchema.get("properties"):
                    params = list(tool.inputSchema["properties"].keys())
                    print(f"     파라미터: {params}")
                print()

            # ── 4. 도구 호출 (tools/call) ──────────────────────
            print("=" * 55)
            print("도구 호출 테스트")
            print("=" * 55)

            # 4-1. search_etf 호출
            print("\n📡 search_etf(category='해외주식', min_return=20)")
            result = await session.call_tool(
                "search_etf",
                arguments={"category": "해외주식", "min_return": 20},
            )
            print(f"결과:\n{result.content[0].text}")

            # 4-2. get_etf_detail 호출
            print("\n📡 get_etf_detail(name='KODEX 200')")
            result = await session.call_tool(
                "get_etf_detail",
                arguments={"name": "KODEX 200"},
            )
            print(f"결과:\n{result.content[0].text}")

            # 4-3. compare_etf 호출
            print("\n📡 compare_etf(name1='KODEX 200', name2='TIGER 미국나스닥100')")
            result = await session.call_tool(
                "compare_etf",
                arguments={
                    "name1": "KODEX 200",
                    "name2": "TIGER 미국나스닥100",
                },
            )
            print(f"결과:\n{result.content[0].text}")

            # ── 5. 리소스 목록 조회 ──────────────────────────
            resources_result = await session.list_resources()
            resources = resources_result.resources
            if resources:
                print(f"\n📚 사용 가능한 리소스: {len(resources)}개")
                for r in resources:
                    print(f"  📄 {r.uri} — {r.name}")
            else:
                print("\n📚 리소스 없음 (이 서버는 도구만 제공)")

            print("\n✅ 클라이언트 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
