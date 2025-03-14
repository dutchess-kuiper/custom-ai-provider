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
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }
        
        response = requests.post(
            "http://localhost:8001/v1/chat/completions", 
            json=payload
        )
        
        print(f"Chat completion status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error with chat completion: {e}")
        return False

if __name__ == "__main__":
    print("Testing connection to OpenAI-compatible server...")
    
    # Run tests
    health_ok = test_server_connection()
    models_ok = test_models_endpoint()
    chat_ok = test_chat_completion()
    
    # Summarize results
    print("\nTest Results:")
    print(f"Health check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Models endpoint: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"Chat completion: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    
    if not (health_ok and models_ok and chat_ok):
        print("\n⚠️  Some tests failed. The server may not be working correctly.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! The server is working correctly.")
        sys.exit(0) 