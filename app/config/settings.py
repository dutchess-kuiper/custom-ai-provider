import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API settings
API_VERSION = "v1"
API_PORT = int(os.getenv("API_PORT", "8001"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM model settings
DEFAULT_MODEL = "gpt-4o"
AVAILABLE_MODELS = [
    {"id": "gpt-4o-mini", "object": "model", "created": 1677610602, "owned_by": "custom-ai-provider"},
    {"id": "gpt-4o", "object": "model", "created": 1677649963, "owned_by": "custom-ai-provider"}
]

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant powered by GPT-4. 
You provide clear, concise, and accurate information. Be friendly and conversational in your responses. 
When appropriate, provide detailed explanations and examples to help users understand complex topics. 
You have access to a wide range of knowledge and can help with various tasks.""" 