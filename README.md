# OpenAI-Compatible API Server with Haystack Integration

This is a FastAPI server that provides an API interface compatible with OpenAI's Chat Completions API. It integrates with Haystack, allowing you to build powerful NLP pipelines with any LLM while maintaining OpenAI API compatibility.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Full integration with Haystack for advanced NLP pipelines
- Support for streaming responses
- Conversation history management
- Compatible with OpenAI SDKs and libraries
- Uses UV package manager and virtual environment for faster dependency installation

## Installation

### Automatic Setup (Recommended)

Run the setup script which installs UV, creates a virtual environment, and installs all dependencies:

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### Manual Setup

If you prefer to set up manually:

1. Install UV (fast Python package installer):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment:

   ```bash
   uv venv .venv
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Running the Server

Start the server with the provided script (which automatically activates the virtual environment):

```bash
./start_server.sh
```

Or manually:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the server
python main.py
```

The API will be available at `http://localhost:8001/v1`.

## Testing the Server

With the virtual environment activated:

```bash
python test_client.py
```

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

## API Usage

The API mimics OpenAI's interface:

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
  "model": "gpt-3.5-turbo",
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
    model="gpt-3.5-turbo",
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
    model: "gpt-3.5-turbo",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is FastAPI?" },
    ],
  });

  console.log(completion.choices[0].message.content);
}

main();
```

## Frontend Integration with Vercel AI SDK

When using Vercel AI SDK or similar frontend libraries, ensure you're correctly specifying the base URL:

```typescript
import { OpenAICompatible } from "@ai-sdk/openai-compatible";

const model = new OpenAICompatible({
  // Use explicit IPv4 address to avoid IPv6 issues
  baseURL: "http://127.0.0.1:8001/v1", // Use 127.0.0.1 instead of localhost
  apiKey: "any-key-works",
  model: "gpt-3.5-turbo",
});
```

## Security Notes

⚠️ **Important**: The current implementation includes an API key directly in the code. In a production environment:

1. Use environment variables or a secure secret management solution
2. Follow proper security practices for API key handling
3. Consider implementing proper authentication for your API

## Troubleshooting

### Virtual Environment Issues

If you encounter issues with the virtual environment:

- Make sure you've run `./setup.sh` to create the virtual environment
- If the script fails, try manually creating and activating the environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### Connection Refused Errors

If you see `ECONNREFUSED` errors:

- Verify the server is running: `python test_client.py`
- Check that your frontend is using the correct base URL
- Use IPv4 address format explicitly: `http://127.0.0.1:8001/v1` instead of `http://localhost:8001/v1`
- Add a small delay before the first request if the server is starting up
- Try running both the server and client on the same network/environment

### IPv6 Issues

If you're seeing errors with `::1:8001` (IPv6 loopback address):

```
Error: connect ECONNREFUSED ::1:8001
```

This is usually because:

1. Your client is trying to use IPv6 but the server is only listening on IPv4
2. The IPv6 stack on your machine has issues

Solutions:

- Force IPv4 by using `127.0.0.1` instead of `localhost` in client code
- The server now tries to bind to both IPv4 and IPv6 interfaces
- If needed, disable IPv6 resolution in your DNS settings

### Haystack Issues

If you encounter issues with the Haystack pipeline:

- Check the console logs for detailed error messages
- Verify your API key is valid and has sufficient permissions
- Make sure all required Haystack components are properly initialized
- If using custom components, verify they are correctly configured

### Next.js API Route Issues

If using Next.js:

- Make sure your route handler is properly configured
- Check if the request is being made from the client or server-side
- If server-side, ensure the environment running Next.js can access the API server
- Refer to the `client-example.ts` file for a working Next.js route handler example

### CORS Issues

If you encounter CORS errors:

- The server has CORS middleware enabled with `allow_origins=["*"]`
- For production use, replace `["*"]` with specific allowed origins

## References

- This implementation is based on the article [How to Build an OpenAI-Compatible API](https://towardsdatascience.com/how-to-build-an-openai-compatible-api-87c8edea2f06) from Towards Data Science
- Haystack documentation: [Haystack](https://docs.haystack.deepset.ai/)
- UV documentation: [UV - Python Package Manager](https://github.com/astral-sh/uv)
