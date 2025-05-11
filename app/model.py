import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:27b"

def generate_response(prompt: str, model: str = MODEL_NAME) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120, stream=True)
    response.raise_for_status()
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            full_response += data.get("response", "")
            if data.get("done", False):
                break
    return full_response

if __name__ == "__main__":
    print(generate_response("Hello, who are you?"))
