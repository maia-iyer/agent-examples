from pydantic_settings import BaseSettings

class Configuration(BaseSettings):
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1")
    llm_api_base: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "dummy")
    llm_headers: str = os.getenv("LLM_HEADERS", "")

