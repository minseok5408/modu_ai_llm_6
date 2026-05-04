"""
Day 32 MCP + LangChain 통합 클라이언트
=================================================
MCP 도구를 LangChain StructuredTool로 변환하여 LLM과 연동

실행 :
    python clients/02_langchain_client.py

서버(servers/01_basic_server.py)를 자동으로 시작합니다.
⚠️  OPENAI_API_KEY 환경 변수가 필요합니다.
"""
import asyncio
import json
import os
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()


def mcp_tool_to_langchain(mcp_tool, session, loop):
    """MCP 도구를 LangChain StructuredTool로 변환합니다.

    """
    tool_name = mcp_tool.name  # 클로저로 캡처

    def invoke_tool(**kwargs):
        """MCP 도구를 동기 호출합니다."""
        result = loop.run_until_complete(
            session.call_tool(tool_name, arguments=kwargs)
        )
        return result.content[0].text

    # inputSchema에서 파라미터 정보 추출
    properties = mcp_tool.inputSchema.get("properties", {})
    required = mcp_tool.inputSchema.get("required", [])

    # 간단한 args_schema 없이 from_function 사용
    return StructuredTool.from_function(
        func=invoke_tool,
        name=tool_name,
        description=mcp_tool.description or f"MCP tool: {tool_name}",
    )


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["servers/01_basic_server.py"],
    )

    print("=" * 55)
    print("MCP + LangChain 통합 클라이언트")
    print("=" * 55)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP 서버 연결 완료")

            # ── 1. MCP 도구 → LangChain 도구 변환 ─────────────
            loop = asyncio.get_event_loop()
            mcp_tools = await session.list_tools()

            lc_tools = []
            for tool in mcp_tools.tools:
                lc_tool = mcp_tool_to_langchain(tool, session, loop)
                lc_tools.append(lc_tool)
                print(f"  🔧 변환 완료: {tool.name}")

            # ── 2. LLM에 도구 바인딩 ──────────────────────────
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            llm_with_tools = llm.bind_tools(lc_tools)
            print(f"\n🤖 LLM에 {len(lc_tools)}개 도구 바인딩 완료")

            # ── 3. 대화 루프 ──────────────────────────────────
            queries = [
                "해외주식 ETF 중 수익률 20% 이상인 것을 찾아줘",
                "KODEX 200과 TIGER 미국나스닥100을 비교해줘",
                "ACE 미국배당다우존스의 상세 정보를 알려줘",
            ]

            for query in queries:
                print(f"\n{'='*55}")
                print(f"👤 사용자: {query}")
                print(f"{'='*55}")

                messages = [HumanMessage(content=query)]

                # LLM 호출 → 도구 선택
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                # 도구 호출이 있으면 실행
                if response.tool_calls:
                    for tc in response.tool_calls:
                        print(f"\n  🔧 도구 호출: {tc['name']}")
                        print(f"     인자: {json.dumps(tc['args'], ensure_ascii=False)}")

                        # MCP 서버에서 도구 실행
                        result = await session.call_tool(
                            tc["name"], arguments=tc["args"]
                        )
                        result_text = result.content[0].text
                        print(f"     결과: {result_text[:100]}...")

                        messages.append(
                            ToolMessage(content=result_text, tool_call_id=tc["id"])
                        )

                    # 도구 결과를 포함하여 최종 응답 생성
                    final_response = llm_with_tools.invoke(messages)
                    print(f"\n  🤖 AI: {final_response.content}")
                else:
                    print(f"\n  🤖 AI: {response.content}")

            print(f"\n{'='*55}")
            print("✅ MCP + LangChain 통합 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
