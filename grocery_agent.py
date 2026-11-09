
import sys
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools


# Load environment variables from .env
load_dotenv()


async def main():

    # 1. Initialize the OpenAI model
    model_client = OpenAIChatCompletionClient(
        model="gemini-3.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
         model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": "unknown",
                "structured_output": False
            }
    )

    # 2. Configure the Grocery MCP server
    server = StdioServerParams(
        command=sys.executable,
        args=["my_mcp_server.py"],
        read_timeout_seconds=30,
    )

    try:
        # 3. Connect to MCP server and retrieve tools
        tools = await mcp_server_tools(server)

        print("Tools from MCP server:", [t.name for t in tools])

        # 4. Create AutoGen Assistant Agent
        shopper = AssistantAgent(
            name="shopper",
            model_client=model_client,
            tools=tools,
            reflect_on_tool_use=False,
            system_message=(
                "You help with grocery prices. "
                "Use your tools to check prices."
            ),
        )

        # 5. Ask the agent a question
        result = await shopper.run(
            task="What is the price of milk today?"
        )

        # 6. Display the response
        print("\nAgent Response:")
        print(result.messages[-1].content)

    finally:
        # 7. Close the OpenAI model client
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
