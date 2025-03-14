// Example TypeScript file for configuring Vercel AI SDK with the OpenAI-compatible API

import { OpenAICompatible } from "@ai-sdk/openai-compatible";
import { StreamingTextResponse, Message } from "ai";

// Configuration for the AI provider
const MODEL_CONFIG = {
  // Use IPv4 address format explicitly to avoid IPv6 issues
  baseURL: "http://127.0.0.1:8001/v1", // Use 127.0.0.1 instead of localhost
  // You can use any value for apiKey as our server ignores it
  apiKey: "dummy-api-key",
  model: "gemini-2.0-pro-exp-02-05", // Or any model name your server accepts
};

// Create AI client
const ai = new OpenAICompatible(MODEL_CONFIG);

// Example Next.js API route handler (Route Handlers pattern)
export async function POST(req: Request) {
  try {
    // Get messages from request body
    const { messages }: { messages: Message[] } = await req.json();

    // Call the AI with streaming
    const response = await ai.chat({
      messages,
      temperature: 0,
      stream: true,
    });

    // Return streaming response
    return new StreamingTextResponse(response.toStream());
  } catch (error) {
    console.error("AI API Error:", error);
    return new Response(
      JSON.stringify({ error: "Failed to connect to AI provider" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }
}

/*
 * IMPORTANT TROUBLESHOOTING NOTES:
 *
 * 1. Try using the IP address '127.0.0.1' instead of 'localhost' to avoid IPv6 resolution
 * 2. Ensure the server is actually running (use the test_client.py to verify)
 * 3. If using Next.js App Router, make sure this code is in a route.ts file
 * 4. Check if there's a proxy or firewall blocking the connection
 * 5. If running in different containers/environments, ensure network connectivity
 */
