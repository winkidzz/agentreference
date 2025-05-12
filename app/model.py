import requests
import json
import base64
import time
import threading

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:27b"

def log(msg):
    now = time.time()
    ms = int((now - int(now)) * 1000)
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    thread = threading.current_thread().name
    print(f"[{ts}.{ms:03d}][{thread}] {msg}")

def generate_response(prompt: str, model: str = MODEL_NAME, file_bytes=None, file_name=None, file_mime=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    # If file is present, add it to the payload
    if file_bytes is not None and file_name is not None:
        # For images, Ollama expects a base64-encoded string in an 'images' array
        if file_mime and file_mime.startswith("image/"):
            b64img = base64.b64encode(file_bytes).decode("utf-8")
            payload["images"] = [b64img]
        else:
            # For other file types, try to send as base64-encoded 'file' (if supported)
            b64file = base64.b64encode(file_bytes).decode("utf-8")
            payload["file"] = {
                "name": file_name,
                "data": b64file,
                "mime": file_mime or "application/octet-stream"
            }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120, stream=True)
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            chunk = data.get("response", "")
            log(f"Streaming chunk: {chunk!r}")
            if chunk:
                yield chunk
            if data.get("done", False):
                break

if __name__ == "__main__":
    for chunk in generate_response("Hello, who are you?"):
        print(chunk, end="", flush=True)
