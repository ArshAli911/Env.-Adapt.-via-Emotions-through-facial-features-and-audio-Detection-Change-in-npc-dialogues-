"""
Ollama connectivity and model validation utilities.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union


class OllamaStatus(Enum):
    """Enumeration of possible Ollama connection states."""
    SUCCESS = "success"
    PACKAGE_NOT_INSTALLED = "package_not_installed"
    SERVER_NOT_RUNNING = "server_not_running"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_TEST_FAILED = "model_test_failed"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class OllamaCheckResult:
    """Result of Ollama connectivity check."""
    status: OllamaStatus
    is_available: bool
    message: Optional[str] = None
    available_models: Optional[List[str]] = None
    
    def to_tuple(self) -> Tuple[bool, Optional[str]]:
        """Convert to legacy tuple format for backward compatibility."""
        return self.is_available, self.message


class OllamaConfig:
    """Configuration constants for Ollama operations."""
    DEFAULT_MODEL = "deepseek-r1:latest"
    TEST_PROMPT = "Say hello in one word."
    TEST_MAX_TOKENS = 5
    CONNECTION_KEYWORDS = ["connection", "refused", "timeout"]


class OllamaValidator:
    """Handles Ollama connectivity validation with separation of concerns."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def check_ollama_and_model(self, model_name: str = OllamaConfig.DEFAULT_MODEL) -> OllamaCheckResult:
        """
        Comprehensive Ollama connectivity and model validation.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            OllamaCheckResult with detailed status information
        """
        try:
            ollama = self._import_ollama()
            if not ollama:
                return OllamaCheckResult(
                    status=OllamaStatus.PACKAGE_NOT_INSTALLED,
                    is_available=False,
                    message="Ollama package not installed. Run: pip install ollama"
                )
            
            available_models = self._get_available_models(ollama)
            if available_models is None:
                return OllamaCheckResult(
                    status=OllamaStatus.SERVER_NOT_RUNNING,
                    is_available=False,
                    message="Ollama server is not running. Start it with: ollama serve"
                )
            
            if model_name not in available_models:
                return self._handle_model_not_found(model_name, available_models)
            
            # Test model functionality
            test_result = self._test_model_generation(ollama, model_name)
            if test_result.is_available:
                self.logger.info(f"Ollama model '{model_name}' validated successfully")
            
            return test_result
            
        except Exception as e:
            return self._handle_unexpected_error(e)
    
    def _import_ollama(self) -> Optional[object]:
        """Safely import ollama package."""
        try:
            import ollama
            self.logger.debug("Ollama package imported successfully")
            return ollama
        except ImportError:
            self.logger.warning("Ollama package not available")
            return None
    
    def _get_available_models(self, ollama) -> Optional[List[str]]:
        """Get list of available models from Ollama server."""
        try:
            self.logger.debug("Checking Ollama server connection...")
            response = ollama.list()
            self.logger.debug("Ollama server responded successfully")
            
            models = self._extract_model_names(response)
            self.logger.debug(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama server: {e}")
            return None
    
    def _extract_model_names(self, response) -> List[str]:
        """Extract model names from Ollama response."""
        models = []
        if hasattr(response, 'models') and response.models:
            models = [
                m.model if hasattr(m, 'model') else str(m) 
                for m in response.models
            ]
        return models
    
    def _handle_model_not_found(self, model_name: str, available_models: List[str]) -> OllamaCheckResult:
        """Handle case where requested model is not found."""
        available_str = ", ".join(available_models) if available_models else "None"
        message = f"Model '{model_name}' not found. Available models: {available_str}. Run: ollama pull {model_name}"
        
        return OllamaCheckResult(
            status=OllamaStatus.MODEL_NOT_FOUND,
            is_available=False,
            message=message,
            available_models=available_models
        )
    
    def _test_model_generation(self, ollama, model_name: str) -> OllamaCheckResult:
        """Test model generation capability."""
        try:
            self.logger.debug(f"Testing model '{model_name}' generation...")
            
            test_response = ollama.generate(
                model=model_name,
                prompt=OllamaConfig.TEST_PROMPT,
                options={"num_predict": OllamaConfig.TEST_MAX_TOKENS}
            )
            
            if test_response.get('response'):
                self.logger.debug("Model generation test successful")
                return OllamaCheckResult(
                    status=OllamaStatus.SUCCESS,
                    is_available=True,
                    message=None
                )
            else:
                return OllamaCheckResult(
                    status=OllamaStatus.MODEL_TEST_FAILED,
                    is_available=False,
                    message=f"Model '{model_name}' exists but failed to generate response"
                )
                
        except Exception as e:
            self.logger.error(f"Model generation test failed: {e}")
            return OllamaCheckResult(
                status=OllamaStatus.MODEL_TEST_FAILED,
                is_available=False,
                message=f"Model '{model_name}' exists but failed test: {str(e)}"
            )
    
    def _handle_unexpected_error(self, error: Exception) -> OllamaCheckResult:
        """Handle unexpected errors with appropriate categorization."""
        error_msg = str(error).lower()
        
        # Check for connection-related errors
        if any(keyword in error_msg for keyword in OllamaConfig.CONNECTION_KEYWORDS):
            return OllamaCheckResult(
                status=OllamaStatus.SERVER_NOT_RUNNING,
                is_available=False,
                message="Ollama server is not running. Start it with: ollama serve"
            )
        
        # Generic error handling
        self.logger.error(f"Unexpected Ollama error: {error}")
        return OllamaCheckResult(
            status=OllamaStatus.UNKNOWN_ERROR,
            is_available=False,
            message=f"Ollama error: {str(error)}"
        )


# Global validator instance for backward compatibility
_validator = OllamaValidator()


def check_ollama_and_model(model_name: str = OllamaConfig.DEFAULT_MODEL) -> Tuple[bool, Optional[str]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        model_name: Name of the model to validate
        
    Returns:
        Tuple of (is_available, error_message)
    """
    result = _validator.check_ollama_and_model(model_name)
    return result.to_tuple() 