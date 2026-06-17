"""LLM adapters: ILLMService and ICVGenerator implementations."""

from app.infrastructure.llm.llm_cv_generator import LLMCVGenerator
from app.infrastructure.llm.ollama_llm_service import OllamaLLMService

__all__ = ["LLMCVGenerator", "OllamaLLMService"]
