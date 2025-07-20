#!/usr/bin/env python3
"""
Ollama connectivity test script with improved architecture and error handling
"""
import sys
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class TestResult(Enum):
    """Enumeration of possible test results"""
    SUCCESS = "success"
    IMPORT_FAILED = "import_failed"
    CONNECTION_FAILED = "connection_failed"
    NO_MODELS = "no_models"
    GENERATION_FAILED = "generation_failed"


@dataclass
class OllamaTestConfig:
    """Configuration for Ollama testing"""
    TARGET_MODELS: List[str] = None
    TEST_PROMPT: str = "The user is feeling happy. Generate a short, supportive response."
    OLLAMA_PORT: int = 11434
    
    def __post_init__(self):
        if self.TARGET_MODELS is None:
            self.TARGET_MODELS = [
                "deepseek-r1:latest", 
                "deepseek-r1", 
                "qwen2.5:latest", 
                "llama2", 
                "llama3"
            ]


class OllamaConnectivityTester:
    """Handles Ollama connectivity testing with separation of concerns"""
    
    def __init__(self, config: Optional[OllamaTestConfig] = None, verbose: bool = True):
        self.config = config or OllamaTestConfig()
        self.verbose = verbose
        self.ollama = None
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ollama_test')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _print(self, message: str, level: str = "info"):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
        
        if level == "error":
            self.logger.error(message.replace("❌ ", "").replace("⚠️ ", ""))
        elif level == "success":
            self.logger.info(message.replace("✅ ", "").replace("🎉 ", ""))
        else:
            self.logger.info(message.replace("🔍 ", "").replace("🔗 ", "").replace("📋 ", ""))
    
    def test_import(self) -> bool:
        """Test if ollama package can be imported"""
        self._print("🔍 Testing Ollama package import...")
        
        try:
            import ollama
            self.ollama = ollama
            self._print("✅ Ollama package imported successfully", "success")
            return True
        except ImportError as e:
            self._print(f"❌ Failed to import ollama package: {e}", "error")
            self._print("💡 Install with: pip install ollama")
            return False
    
    def test_server_connection(self) -> Tuple[bool, List[str]]:
        """Test connection to Ollama server and get available models"""
        self._print("🔗 Testing Ollama server connection...")
        
        try:
            response = self.ollama.list()
            self._print("✅ Successfully connected to Ollama server", "success")
            
            models = self._extract_models(response)
            self._print(f"📋 Available models: {models}")
            return True, models
            
        except ConnectionError as e:
            self._print(f"❌ Cannot connect to Ollama server: {e}", "error")
            self._print("💡 Make sure Ollama is running with: ollama serve")
            return False, []
        except Exception as e:
            self._print(f"❌ Failed to connect to Ollama server: {e}", "error")
            self._print("💡 Make sure Ollama is running with: ollama serve")
            return False, []
    
    def find_compatible_models(self, available_models: List[str]) -> List[str]:
        """Find models that match our target list"""
        found_models = []
        self._print("\n🎯 Checking for target models:")
        
        for model in self.config.TARGET_MODELS:
            if model in available_models:
                self._print(f"✅ {model} - Available", "success")
                found_models.append(model)
            else:
                self._print(f"❌ {model} - Not found")
        
        return found_models
    
    def test_dialogue_generation(self, model: str) -> Tuple[bool, str]:
        """Test dialogue generation with specified model"""
        self._print(f"\n🧪 Testing dialogue generation with {model}...")
        
        try:
            response = self.ollama.generate(
                model=model, 
                prompt=self.config.TEST_PROMPT
            )
            dialogue = response.get('response', '').strip()
            
            if dialogue:
                self._print("✅ Dialogue generation successful!", "success")
                self._print(f"📝 Generated: {dialogue}")
                return True, dialogue
            else:
                self._print("⚠️ Dialogue generation returned empty response", "error")
                return False, ""
                
        except Exception as e:
            self._print(f"❌ Dialogue generation failed: {e}", "error")
            return False, ""
    
    def _extract_models(self, response) -> List[str]:
        """Extract model names from Ollama response"""
        models = []
        if hasattr(response, 'models') and response.models:
            models = [
                m.model if hasattr(m, 'model') else str(m) 
                for m in response.models
            ]
        else:
            self._print("⚠️ No models found or unexpected response format", "error")
            self._print(f"Raw response: {response}")
        return models
    
    def run_full_test(self) -> Tuple[TestResult, Optional[str]]:
        """Run complete Ollama connectivity test"""
        self._print("🔍 Testing Ollama connectivity...")
        
        # Test import
        if not self.test_import():
            return TestResult.IMPORT_FAILED, None
        
        # Test server connection
        connected, available_models = self.test_server_connection()
        if not connected:
            return TestResult.CONNECTION_FAILED, None
        
        # Find compatible models
        found_models = self.find_compatible_models(available_models)
        if not found_models:
            self._print("\n❌ No suitable models found!", "error")
            self._print("💡 Try installing a model with: ollama pull deepseek-r1:latest")
            return TestResult.NO_MODELS, None
        
        self._print(f"\n🎉 Found {len(found_models)} usable model(s): {found_models}", "success")
        
        # Test dialogue generation
        test_model = found_models[0]
        success, _ = self.test_dialogue_generation(test_model)
        
        if success:
            return TestResult.SUCCESS, test_model
        else:
            return TestResult.GENERATION_FAILED, test_model


class TroubleshootingGuide:
    """Provides troubleshooting guidance for common Ollama issues"""
    
    @staticmethod
    def suggest_fixes(result: TestResult):
        """Suggest fixes based on test result"""
        print("\n🔧 Troubleshooting Steps:")
        
        if result == TestResult.IMPORT_FAILED:
            print("1. Install Python ollama package: pip install ollama")
            print("2. Verify Python environment is correct")
            
        elif result == TestResult.CONNECTION_FAILED:
            print("1. Install Ollama: https://ollama.ai/download")
            print("2. Start Ollama server: ollama serve")
            print("3. Check if Ollama is running on port 11434")
            print("4. Verify firewall settings")
            
        elif result == TestResult.NO_MODELS:
            print("1. Pull a model: ollama pull deepseek-r1:latest")
            print("2. List available models: ollama list")
            print("3. Try alternative models: ollama pull llama2")
            
        elif result == TestResult.GENERATION_FAILED:
            print("1. Check model compatibility")
            print("2. Verify model is fully downloaded")
            print("3. Try with a different model")
            print("4. Check Ollama server logs")
        
        else:
            print("1. Install Ollama: https://ollama.ai/download")
            print("2. Start Ollama server: ollama serve")
            print("3. Pull a model: ollama pull deepseek-r1:latest")
            print("4. Install Python package: pip install ollama")
            print("5. Check if Ollama is running on port 11434")


def test_ollama_connection() -> Tuple[bool, Optional[str]]:
    """Legacy function for backward compatibility"""
    tester = OllamaConnectivityTester()
    result, model = tester.run_full_test()
    return result == TestResult.SUCCESS, model


def main():
    """Main entry point for the test script"""
    tester = OllamaConnectivityTester()
    result, model = tester.run_full_test()
    
    if result == TestResult.SUCCESS:
        print(f"\n🎉 Ollama is working correctly with model: {model}")
        print("✅ Your VR Emotion Adaptation app should now generate dialogue!")
        sys.exit(0)
    else:
        TroubleshootingGuide.suggest_fixes(result)
        sys.exit(1)


if __name__ == "__main__":
    main()