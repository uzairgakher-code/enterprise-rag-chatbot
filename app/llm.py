from langchain.llms import Ollama
import config

def get_llm():
    return Ollama(model=config.LLM_MODEL)
