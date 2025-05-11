from langchain.llms.base import LLM
from typing import Optional, List
from app.model import generate_response
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class OllamaGemmaLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return generate_response(prompt)

    @property
    def _llm_type(self) -> str:
        return "ollama-gemma"

def get_chain():
    llm = OllamaGemmaLLM()
    prompt = PromptTemplate(input_variables=["question"], template="{question}")
    return LLMChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    chain = get_chain()
    result = chain.run({"question": "What is the capital of France?"})
    print(result)
