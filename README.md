# OpenAI-Compatible API Server

This is a FastAPI server that provides an API interface compatible with OpenAI's Chat Completions API. It allows you to wrap your own AI models with an OpenAI-compatible interface, making them usable with tools and libraries designed for OpenAI's API.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Support for streaming responses
- Mock implementation that can be replaced with your own language model
- Compatible with OpenAI SDKs and libraries

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

Start the server with the provided script:

```bash
./start_server.sh
```

Or run it directly:

```bash
python main.py
```

The API will be available at `http://localhost:8001/v1`.

## Testing the Server

Use the included test script to verify the server is running correctly:

```bash
python test_client.py
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

## Integrating Your Own Model

To integrate your own language model:

1. Modify the `generate_chat_completion` function in `main.py` to call your model
2. Adapt the `stream_chat_completion` function if you want to support streaming

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

## Troubleshooting

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

This implementation is based on the article [How to Build an OpenAI-Compatible API](https://towardsdatascience.com/how-to-build-an-openai-compatible-api-87c8edea2f06) from Towards Data Science.
