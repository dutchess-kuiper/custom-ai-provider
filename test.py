from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo


time_tool = MCPTool(
    name="octagon-transcripts-agent",
    description="A Tool to get financial data",
    server_info=StdioServerInfo(
        command="npx",
        args=["-y", "octagon-mcp@latest"],
        env={"OCTAGON_API_KEY": "sk_5d1tfws4KwT6pmDnUqldavx4Coow3akeU9Lcy1rBPkuyttiF02R0vQirsA5PtPwKiky5KDA1_3ElXz_KTVwSJw"},
    ),
)
pipeline = Pipeline()
pipeline.add_component(
    "llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[time_tool])
)
pipeline.add_component("tool_invoker", ToolInvoker(tools=[time_tool]))
pipeline.add_component(
    "adapter",
    OutputAdapter(
        template="{{ initial_msg + initial_tool_messages + tool_messages }}",
        output_type=list[ChatMessage],
        unsafe=True,
    ),
)
pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))
pipeline.connect("llm.replies", "tool_invoker.messages")
pipeline.connect("llm.replies", "adapter.initial_tool_messages")
pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
pipeline.connect("adapter.output", "response_llm.messages")

user_input = "What is the transcript for apple?"  # can be any city
user_input_msg = ChatMessage.from_user(text=user_input)

result = pipeline.run(
    {
        "llm": {"messages": [user_input_msg]},
        "adapter": {"initial_msg": [user_input_msg]},
    }
)

print(result["response_llm"]["replies"][0].text)
# The current time in New York is 1:57 PM.
