# llm_config.py — Hybrid LLM factory (Ollama / OpenAI)
# Handles provider switching, fallback logic, and client instantiation
"""
llm_config.py — LLM Configuration & Factory

Local-only LLM implementation using Ollama.
Provides a unified interface for text generation and summarization.

Default model: tinyllama (lightweight, fast, runs on CPU)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MODEL = "tinyllama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT = 60  # seconds


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class OllamaConnectionError(LLMError):
    """Raised when Ollama server is not reachable."""
    pass


class OllamaGenerationError(LLMError):
    """Raised when text generation fails."""
    pass


class ModelNotFoundError(LLMError):
    """Raised when requested model is not available."""
    pass


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM."""
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = REQUEST_TIMEOUT


class OllamaLLM:
    """
    Lightweight Ollama client for text generation.
    
    Uses only stdlib (urllib) - no external dependencies required.
    """
    
    def __init__(self, config: OllamaConfig | None = None):
        """
        Initialize Ollama client.
        
        Args:
            config: Optional OllamaConfig, uses defaults if not provided
        """
        self.config = config or OllamaConfig()
        self._verified = False
    
    @property
    def model(self) -> str:
        return self.config.model
    
    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")
    
    def _make_request(
        self,
        endpoint: str,
        payload: dict | None = None,
        method: str = "POST",
    ) -> dict:
        """
        Make HTTP request to Ollama API.
        
        Args:
            endpoint: API endpoint (e.g., "/api/generate")
            payload: Request body (JSON)
            method: HTTP method
            
        Returns:
            Response JSON as dict
            
        Raises:
            OllamaConnectionError: If server is not reachable
            OllamaGenerationError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            data = json.dumps(payload).encode("utf-8") if payload else None
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                # Handle streaming response (Ollama returns newline-delimited JSON)
                full_response = ""
                final_data = {}
                
                for line in response:
                    line_str = line.decode("utf-8").strip()
                    if line_str:
                        chunk = json.loads(line_str)
                        if "response" in chunk:
                            full_response += chunk.get("response", "")
                        if chunk.get("done", False):
                            final_data = chunk
                            final_data["response"] = full_response
                            break
                
                return final_data if final_data else {"response": full_response}
                
        except urllib.error.URLError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: `ollama serve`\n"
                f"Error: {str(e)}"
            )
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ModelNotFoundError(
                    f"Model '{self.config.model}' not found. "
                    f"Pull it with: `ollama pull {self.config.model}`"
                )
            raise OllamaGenerationError(f"HTTP error {e.code}: {e.reason}")
        except json.JSONDecodeError as e:
            raise OllamaGenerationError(f"Invalid response from Ollama: {str(e)}")
        except Exception as e:
            raise OllamaGenerationError(f"Unexpected error: {str(e)}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is running and reachable.
        
        Returns:
            True if server is available, False otherwise
        """
        try:
            url = f"{self.base_url}/api/tags"
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    
    def list_models(self) -> list[str]:
        """
        List available models on the Ollama server.
        
        Returns:
            List of model names
            
        Raises:
            OllamaConnectionError: If server is not reachable
        """
        try:
            url = f"{self.base_url}/api/tags"
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except urllib.error.URLError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama: {str(e)}")
        except Exception as e:
            raise OllamaGenerationError(f"Failed to list models: {str(e)}")
    
    def verify_model(self) -> bool:
        """
        Verify that the configured model is available.
        
        Returns:
            True if model is available
            
        Raises:
            ModelNotFoundError: If model is not available
            OllamaConnectionError: If server is not reachable
        """
        models = self.list_models()
        model_base = self.config.model.split(":")[0]  # Handle tags like "tinyllama:latest"
        
        available = any(
            model_base in m or self.config.model in m 
            for m in models
        )
        
        if not available:
            raise ModelNotFoundError(
                f"Model '{self.config.model}' not found. "
                f"Available models: {', '.join(models) or 'none'}. "
                f"Pull with: `ollama pull {self.config.model}`"
            )
        
        self._verified = True
        return True
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: User prompt / input text
            system_prompt: Optional system prompt for context
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text response
            
        Raises:
            OllamaConnectionError: If server is not reachable
            OllamaGenerationError: If generation fails
            ModelNotFoundError: If model is not available
        """
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,  # We handle streaming internally
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = self._make_request("/api/generate", payload)
        return response.get("response", "").strip()
    
    def summarize(
        self,
        text: str,
        max_length: int | None = None,
        style: str = "concise",
    ) -> str:
        """
        Summarize input text.
        
        Args:
            text: Text to summarize
            max_length: Optional max length hint (in words, approximately)
            style: Summarization style ("concise", "detailed", "bullet")
            
        Returns:
            Summarized text
        """
        style_instructions = {
            "concise": "Provide a brief, concise summary in 2-3 sentences.",
            "detailed": "Provide a detailed summary covering all key points.",
            "bullet": "Provide a summary as bullet points listing key insights.",
        }
        
        instruction = style_instructions.get(style, style_instructions["concise"])
        
        if max_length:
            instruction += f" Keep it under {max_length} words."
        
        prompt = f"""Summarize the following text.

{instruction}

Text to summarize:
{text}

Summary:"""
        
        return self.generate(
            prompt=prompt,
            system_prompt="You are a helpful assistant that provides clear, accurate summaries.",
            temperature=0.3,  # Lower temperature for more focused summaries
        )
    
    def analyze(
        self,
        data_description: str,
        question: str,
    ) -> str:
        """
        Analyze data based on a description and question.
        
        Args:
            data_description: Description of the data/statistics
            question: Analysis question to answer
            
        Returns:
            Analysis response
        """
        prompt = f"""Based on the following data analysis results, answer the question.

Data Summary:
{data_description}

Question: {question}

Provide a clear, data-driven answer:"""
        
        return self.generate(
            prompt=prompt,
            system_prompt=(
                "You are a data analyst assistant. Provide clear, actionable insights "
                "based on the data presented. Be specific and reference actual numbers."
            ),
            temperature=0.5,
        )
    
    def generate_insight(
        self,
        statistical_finding: str,
        context: str | None = None,
    ) -> str:
        """
        Transform a statistical finding into a business insight.
        
        Args:
            statistical_finding: Raw statistical observation
            context: Optional business context
            
        Returns:
            Business-friendly insight
        """
        context_str = f"\nBusiness context: {context}" if context else ""
        
        prompt = f"""Transform this statistical finding into a clear business insight.

Statistical finding: {statistical_finding}{context_str}

Write a 1-2 sentence business insight that:
1. Explains what this means in plain English
2. Suggests a potential action or implication

Business insight:"""
        
        return self.generate(
            prompt=prompt,
            system_prompt=(
                "You are a business analyst who translates statistics into actionable insights. "
                "Be clear, specific, and business-focused."
            ),
            temperature=0.6,
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    verify: bool = False,
) -> OllamaLLM:
    """
    Factory function to create an LLM instance.
    
    Args:
        model: Ollama model name (default: tinyllama)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        base_url: Ollama server URL
        verify: If True, verify model availability on creation
        
    Returns:
        Configured OllamaLLM instance
        
    Raises:
        OllamaConnectionError: If verify=True and server is not reachable
        ModelNotFoundError: If verify=True and model is not available
        
    Example:
        llm = get_llm(model="tinyllama", temperature=0.7)
        response = llm.generate("Explain this data trend...")
    """
    config = OllamaConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    llm = OllamaLLM(config)
    
    if verify:
        llm.verify_model()
    
    return llm


def check_ollama_status() -> dict:
    """
    Check Ollama server status and available models.
    
    Returns:
        Dict with status information:
        {
            "available": bool,
            "base_url": str,
            "models": list[str],
            "error": str | None
        }
    """
    llm = OllamaLLM()
    
    result = {
        "available": False,
        "base_url": llm.base_url,
        "models": [],
        "default_model": DEFAULT_MODEL,
        "error": None,
    }
    
    try:
        if llm.is_available():
            result["available"] = True
            result["models"] = llm.list_models()
    except OllamaConnectionError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_generate(prompt: str, **kwargs) -> str:
    """
    Quick one-off text generation.
    
    Args:
        prompt: Text prompt
        **kwargs: Passed to get_llm()
        
    Returns:
        Generated text
    """
    llm = get_llm(**kwargs)
    return llm.generate(prompt)


def quick_summarize(text: str, **kwargs) -> str:
    """
    Quick one-off summarization.
    
    Args:
        text: Text to summarize
        **kwargs: Passed to get_llm()
        
    Returns:
        Summary text
    """
    llm = get_llm(**kwargs)
    return llm.summarize(text)
