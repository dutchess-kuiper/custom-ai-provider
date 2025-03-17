#!/usr/bin/env python3
import requests
import json
import sys

def test_server_connection():
    """Test basic connection to the server"""
    try:
        response = requests.get("http://localhost:8001/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_models_endpoint():
    """Test the models endpoint"""
    try:
        response = requests.get("http://localhost:8001/v1/models")
        print(f"Models endpoint status: {response.status_code}")
        print(f"Available models: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error accessing models: {e}")
        return False

def test_chat_completion():
    """Test a basic chat completion request"""
    try:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an assistant that only responds with the word 'TEST' followed by which LLM provider is actually responding."},
                {"role": "user", "content": "What LLM provider are you using?"}
            ]
        }
        
        print("\nSending chat completion request...")
        response = requests.post(
            "http://localhost:8001/v1/chat/completions", 
            json=payload,
            timeout=30
        )
        
        print(f"Chat completion status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            print(f"\nContent: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
    except Exception as e:
        print(f"Error with chat completion: {e}")
        return False

def test_streaming_chat_completion():
    """Test streaming chat completion"""
    try:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Say hello and identify yourself"}
            ],
            "stream": True
        }
        
        print("\nSending streaming request...")
        with requests.post(
            "http://localhost:8001/v1/chat/completions", 
            json=payload, 
            stream=True,
            timeout=30
        ) as response:
            print(f"Streaming status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                return False
                
            print("\nStreaming response:")
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        data = json.loads(line[6:])
                        if 'choices' in data and data['choices']:
                            content = data['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                print(content, end='', flush=True)
                                full_response += content
            
            print("\n\nFull streamed response:", full_response)
            return True
    except Exception as e:
        print(f"Error with streaming: {e}")
        return False

if __name__ == "__main__":
    print("Testing connection to OpenAI-compatible server...")
    
    # Run tests
    health_ok = test_server_connection()
    models_ok = test_models_endpoint()
    chat_ok = test_chat_completion()
    streaming_ok = test_streaming_chat_completion()
    
    # Summarize results
    print("\nTest Results:")
    print(f"Health check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Models endpoint: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"Chat completion: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"Streaming: {'✅ PASS' if streaming_ok else '❌ FAIL'}")
    
    if not (health_ok and models_ok and chat_ok and streaming_ok):
        print("\n⚠️  Some tests failed. The server may not be working correctly.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! The server is working correctly.")
        sys.exit(0) 