import asyncio

from openshift_mcp_client.chatbot import build_mcp_server


async def main():
    server = build_mcp_server()

    try:
        await server.connect()
        print("\nConnected to OpenShift MCP server.\n")

        tools = await server.list_tools()
        print(f"Found {len(tools)} MCP tools:\n")
        for tool in tools:
            print(f"- {tool.name}")
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
