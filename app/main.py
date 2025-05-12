from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from app.chain import get_chain
import uvicorn
import os
import time
from app.model import generate_response
import uuid
import threading

app = FastAPI()

# Serve static files from the 'static' directory at /static
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str = None  # Allow None for new sessions

class ChatResponse(BaseModel):
    response: str
    session_id: str

def log(msg):
    now = time.time()
    ms = int((now - int(now)) * 1000)
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    thread = threading.current_thread().name
    print(f"[{ts}.{ms:03d}][{thread}] {msg}")

@app.post("/chat")
async def chat(request: Request, 
               message: str = Form(None),
               session_id: str = Form(None),
               file: UploadFile = File(None)):
    log("/chat endpoint called")
    # Support both JSON and multipart/form-data
    if request.headers.get("content-type", "").startswith("application/json"):
        data = await request.json()
        message = data.get("message", "")
        session_id = data.get("session_id")
        file = None
    # Generate a unique session_id if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
        log(f"Generated new session_id: {session_id}")
    else:
        log(f"Using provided session_id: {session_id}")
    file_bytes = None
    file_content = None
    file_mime = None
    file_name = None
    if file is not None:
        file_bytes = await file.read()
        file_mime = file.content_type
        file_name = file.filename
        log(f"File uploaded: {file_name} ({file_mime}), {len(file_bytes)} bytes")
        try:
            file_content = file_bytes.decode("utf-8")
        except Exception:
            file_content = None
    # Get the chain, memory, and session
    chain, memory, session_id = get_chain(session_id)
    log(f"Chain and memory loaded for session_id: {session_id}")
    memory_vars = memory.load_memory_variables({})
    history_text = memory_vars.get("history", "")
    # Compose question
    question = message or ""
    if file_content:
        question = f"[File uploaded: {file_name}]\n{file_content}\n\n{question}"
    elif file_name:
        question = f"[File uploaded: {file_name}]\n\n{question}"
    # Hybrid logic: if file, call generate_response directly; else use LangChain
    if file_bytes is not None:
        log("Using multimodal (file) path for response generation")
        # Build multimodal prompt with history
        multimodal_prompt = f"{history_text}\nUser: {question}\nAgent:"
        def stream_gen():
            agent_response = ""
            for chunk in generate_response(multimodal_prompt, file_bytes=file_bytes, file_name=file_name, file_mime=file_mime):
                agent_response += chunk
                log(f"Streaming multimodal chunk: {chunk!r}")
                yield chunk
            # After streaming, update memory with user and agent turns
            memory.save_context({"question": question}, {"output": agent_response})
            log("Multimodal response complete and memory updated.")
        return StreamingResponse(stream_gen(), media_type="text/plain", headers={"X-Session-ID": session_id})
    else:
        log("Using text-only (LangChain) path for response generation")
        # Use LangChain for text-only
        inputs = {"history": history_text, "question": question}
        def stream_gen():
            for chunk in chain.stream(inputs):
                # If chunk is a Generation object, yield chunk.text
                if hasattr(chunk, "text"):
                    log(f"Streaming text chunk: {chunk.text!r}")
                    yield chunk.text
                # If chunk is a generator, join and yield
                elif hasattr(chunk, "__iter__") and not isinstance(chunk, (str, bytes)):
                    joined = "".join(list(chunk))
                    log(f"Streaming joined generator chunk: {joined!r}")
                    yield joined
                else:
                    log(f"Streaming str chunk: {str(chunk)!r}")
                    yield str(chunk)
        return StreamingResponse(stream_gen(), media_type="text/plain", headers={"X-Session-ID": session_id})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, reload=True)
