from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.chain import get_chain
import uvicorn
import os

app = FastAPI()
chain = get_chain()

# In-memory conversation history: {session_id: [messages]}
conversation_history = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # For demo, default to a single session

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # Get or initialize history
        history = conversation_history.setdefault(req.session_id, [])
        # Build prompt: concatenate all previous messages and the new one
        prompt = "\n".join(history + [req.message])
        answer = chain.run({"question": prompt})
        # Update history
        history.append(req.message)
        history.append(answer)
        return ChatResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from the 'static' directory
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8081, reload=True)
