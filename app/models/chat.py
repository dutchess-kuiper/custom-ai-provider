from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ToolFunction(BaseModel):
    function: FunctionDefinition

class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str
    index: int
    function: FunctionCall

class ToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, Any] = Field(default_factory=dict)

class CompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    functions: Optional[List[FunctionDefinition]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None

class MessageWithToolCalls(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None

class CompletionChoice(BaseModel):
    index: int
    message: Union[Message, MessageWithToolCalls]
    finish_reason: str

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class StreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[StreamChoice] 