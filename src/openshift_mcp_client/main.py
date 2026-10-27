"""CLI entrypoint for the OpenShift MCP chatbot."""

from __future__ import annotations

import asyncio

from openshift_mcp_client.chatbot import ChatRuntime


async def async_main() -> None:
    runtime = await ChatRuntime.create()
    try:
        print("\nConnected to OpenShift MCP server.")
        print("\nOpenShift Llama Agent started.")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in {"exit", "quit"}:
                break
            reply = await runtime.ask(user_input)
            print(f"\nAssistant: {reply}")
    finally:
        await runtime.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
