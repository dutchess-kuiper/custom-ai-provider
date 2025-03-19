# OpenAI-Compatible API Server with Haystack Integration

A simplified FastAPI server providing an OpenAI-compatible API interface with Haystack integration for LLM capabilities.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Full integration with Haystack for advanced NLP pipelines
- Support for streaming responses
- Conversation history management
- Compatible with OpenAI SDKs and libraries
- Docker support for easy deployment

## Project Structure

```
.
├── app/                    # Main application package
│   ├── api/                # API routes and handlers
│   │   └── routes.py       # OpenAI-compatible API endpoints
│   ├── config/             # Configuration
│   │   └── settings.py     # App settings and environment variables
│   ├── core/               # Core functionality
│   ├── models/             # Data models
│   │   └── chat.py         # Pydantic models for chat API
│   ├── services/           # Service layer
│   │   └── llm.py          # Haystack LLM integration
│   └── utils/              # Utility functions
├── main.py                 # Application entry point
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not in repo)
```

## Installation

### Environment Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key
   API_PORT=8001
   API_HOST=0.0.0.0
   ```

## Running the Server

### Local Development

Start the server with:

```bash
# Option 1: Using the run.py script (recommended)
./run.py

# Option 2: Using the Python module
python -m app

# Option 3: Using the main script
python main.py
```

The API will be available at `http://localhost:8001/v1`.

### Docker

Build and run the Docker container:

```bash
# Build the Docker image
docker build -t openai-compatible-api .

# Run the container
docker run -p 8001:8001 --env-file .env openai-compatible-api
```

## API Usage

### List Available Models

```
GET /v1/models
```

### Create a Chat Completion

```
POST /v1/chat/completions
```

Request body:

```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" }
  ],
  "temperature": 0.7,
  "stream": false
}
```

### Streaming Example

Set `"stream": true` in your request to receive a streamed response.

## Client Examples

### Python

```python
import openai

client = openai.OpenAI(
    api_key="any-key-works",  # API key is ignored but required
    base_url="http://localhost:8001/v1"  # Point to your server
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is FastAPI?"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript

```javascript
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "any-key-works",
  baseURL: "http://localhost:8001/v1",
});

async function main() {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is FastAPI?" },
    ],
  });

  console.log(completion.choices[0].message.content);
}

main();
```

## Customizing the Haystack Pipeline

To extend the LLM capabilities:

1. Modify the `LLMService` class in `app/services/llm.py`
2. Customize the pipeline with additional Haystack components
3. Update the API endpoints as needed

## Security Notes

⚠️ **Important**: For production deployment:

1. Use environment variables for sensitive configuration
2. Implement proper authentication for your API
3. Restrict CORS settings to specific origins

## Haystack Integration

This server uses [Haystack](https://haystack.deepset.ai/) to power its language model capabilities, with the following features:

1. Uses a Haystack Pipeline with an OpenAIGenerator component
2. Maintains conversation history for stateful chats
3. Provides both standard and streaming response capabilities
4. Handles error cases gracefully

### Customizing the Haystack Pipeline

To customize the Haystack pipeline:

1. Modify the `HaystackPipeline` class in `main.py`
2. Change the generator configuration or add additional components
3. Implement your own processing logic as needed

Example of adding a document retrieval component:

```python
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Initialize document store and add documents
document_store = InMemoryDocumentStore()
document_store.write_documents(your_documents)

# Add retriever to pipeline
retriever = InMemoryBM25Retriever(document_store=document_store)
self.pipeline.add_component("retriever", retriever)

# Connect components
self.pipeline.connect("retriever", "generator")
```

## References

- This implementation is based on the article [How to Build an OpenAI-Compatible API](https://towardsdatascience.com/how-to-build-an-openai-compatible-api-87c8edea2f06) from Towards Data Science
- Haystack documentation: [Haystack](https://docs.haystack.deepset.ai/)
- UV documentation: [UV - Python Package Manager](https://github.com/astral-sh/uv)
