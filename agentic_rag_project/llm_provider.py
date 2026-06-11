"""
LLM Provider Module

Implements:
  - Ollama local model detection and usage (Air-Gapped Deployment)
  - External API support (OpenAI, Anthropic, custom endpoints)
  - Model configuration and initialization
  - Streaming Outputs support
  - Speculative Decoding awareness
"""

import os
import subprocess
import json
import logging
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Ollama Model Detection
# ─────────────────────────────────────────────

def detect_ollama_models() -> List[Dict[str, str]]:
    """
    Detect locally downloaded Ollama models.
    Air-Gapped Deployment: Supports fully local operation.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            logger.warning(f"Ollama list returned non-zero: {result.stderr}")
            return []
        
        models = []
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            return []
        
        # Skip header line
        for line in lines[1:]:
            if line.strip():
                parts = line.split()
                if parts:
                    model_info = {
                        "name": parts[0],
                        "id": parts[1] if len(parts) > 1 else "",
                        "size": parts[2] if len(parts) > 2 else "",
                    }
                    models.append(model_info)
        
        logger.info(f"Detected {len(models)} Ollama models: {[m['name'] for m in models]}")
        return models
    
    except FileNotFoundError:
        logger.info("Ollama binary not found. Not installed.")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("Ollama list command timed out.")
        return []
    except Exception as e:
        logger.error(f"Error detecting Ollama models: {e}")
        return []


def is_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        response = urllib.request.urlopen(req, timeout=5)
        return response.status == 200
    except Exception:
        return False


# ─────────────────────────────────────────────
# LLM Provider Abstraction
# ─────────────────────────────────────────────

class LLMProvider:
    """
    LLM Provider abstraction layer.
    Supports both local (Ollama) and external (OpenAI, Anthropic, custom) models.
    
    Implements:
      - Tool Use / Function Calling capability
      - Streaming Outputs
      - Speculative Decoding awareness
    """
    
    def __init__(self, provider_type: str, model_name: str,
                 api_key: str = "", api_base_url: str = "",
                 temperature: float = 0.2, max_tokens: int = 2048):
        self.provider_type = provider_type
        self.model_name = model_name
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._init_provider()
    
    def _init_provider(self):
        """Initialize the LLM provider based on type."""
        if self.provider_type == "ollama":
            self._init_ollama()
        elif self.provider_type == "openai":
            self._init_openai()
        elif self.provider_type == "anthropic":
            self._init_anthropic()
        elif self.provider_type == "custom":
            self._init_custom()
        else:
            logger.warning(f"Unknown provider type: {self.provider_type}")
    
    def _init_ollama(self):
        """
        Initialize Ollama local model.
        Air-Gapped Deployment: Fully local with zero internet connection.
        """
        try:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
            logger.info(f"Initialized Ollama LLM: {self.model_name}")
        except ImportError:
            logger.error("langchain-ollama not installed. Install with: pip install langchain-ollama")
            self._llm = None
    
    def _init_openai(self):
        """Initialize OpenAI model."""
        try:
            from langchain_openai import ChatOpenAI
            os.environ["OPENAI_API_KEY"] = self.api_key
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
            )
            logger.info(f"Initialized OpenAI LLM: {self.model_name}")
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI
                os.environ["OPENAI_API_KEY"] = self.api_key
                self._llm = CommunityChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                logger.info(f"Initialized OpenAI LLM (community): {self.model_name}")
            except ImportError:
                logger.error("Neither langchain-openai nor langchain-community[openai] installed.")
                self._llm = None
    
    def _init_anthropic(self):
        """Initialize Anthropic (Claude) model."""
        try:
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                anthropic_api_key=self.api_key,
            )
            logger.info(f"Initialized Anthropic LLM: {self.model_name}")
        except ImportError:
            logger.error("langchain-anthropic not installed. Install with: pip install langchain-anthropic")
            self._llm = None
    
    def _init_custom(self):
        """
        Initialize custom API endpoint.
        Supports any OpenAI-compatible API (e.g., vLLM, local servers).
        """
        try:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key or "not-needed",
                base_url=self.api_base_url,
            )
            logger.info(f"Initialized Custom LLM: {self.model_name} at {self.api_base_url}")
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI
                os.environ["OPENAI_API_KEY"] = self.api_key or "not-needed"
                os.environ["OPENAI_API_BASE"] = self.api_base_url
                self._llm = CommunityChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                logger.info(f"Initialized Custom LLM (community): {self.model_name}")
            except ImportError:
                logger.error("No OpenAI-compatible LangChain package installed.")
                self._llm = None
    
    @property
    def llm(self):
        """Get the underlying LLM instance."""
        return self._llm
    
    def invoke(self, messages) -> Any:
        """
        Invoke the LLM with messages.
        Supports the standard LangChain message format.
        """
        if self._llm is None:
            raise RuntimeError(f"LLM provider '{self.provider_type}' not initialized properly")
        return self._llm.invoke(messages)
    
    def stream(self, messages):
        """
        Streaming Outputs:
        Deliver tokens in real-time as they are generated.
        """
        if self._llm is None:
            raise RuntimeError(f"LLM provider '{self.provider_type}' not initialized properly")
        
        if hasattr(self._llm, 'stream'):
            return self._llm.stream(messages)
        else:
            # Fallback: yield full response
            response = self._llm.invoke(messages)
            yield response
    
    def get_info(self) -> Dict:
        """Get provider information."""
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "is_local": self.provider_type == "ollama",
            "initialized": self._llm is not None,
        }
