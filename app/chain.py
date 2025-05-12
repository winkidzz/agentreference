from langchain.llms.base import LLM
from typing import Optional, List
from app.model import generate_response
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import uuid

class OllamaGemmaLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_str = str(prompt)
        return ''.join(generate_response(prompt_str))

    def stream(self, prompt: str, stop: Optional[List[str]] = None):
        prompt_str = str(prompt)
        for chunk in generate_response(prompt_str):
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "ollama-gemma"

# Global session_id -> memory mapping
session_memories = {}

def get_chain(session_id=None):
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=False)
    memory = session_memories[session_id]
    llm = OllamaGemmaLLM()
    prompt = PromptTemplate(
        input_variables=["history", "question"],
        template="{history}\nUser: {question}\nAgent:"
    )
    chain = prompt | llm
    return chain, memory, session_id

if __name__ == "__main__":
    chain, memory, session_id = get_chain()
    memory_vars = memory.load_memory_variables({})
    result = chain.invoke({"history": memory_vars.get("history", ""), "question": "What is the capital of France?"})
    print(result)
