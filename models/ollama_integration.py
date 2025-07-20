import ollama
import random
from typing import Dict, List, Optional
from enum import Enum
import logging
import os
from dataclasses import dataclass

class OllamaIntegrationError(Exception):
    """Base exception for Ollama integration errors"""
    pass

class ModelNotAvailableError(OllamaIntegrationError):
    """Raised when the Ollama model is not available"""
    pass

class DialogueGenerationError(OllamaIntegrationError):
    """Raised when dialogue generation fails"""
    pass

class ConfigurationError(OllamaIntegrationError):
    """Raised when configuration is invalid"""
    pass

@dataclass
class OllamaConfig:
    """Configuration for Ollama integration"""
    default_model: str = "deepseek-r1:latest"
    max_dialogue_length: int = 200
    short_dialogue_length: int = 15
    fallback_dialogue_length: int = 150
    primary_type_weight: float = 0.7
    log_level: int = logging.DEBUG
    log_format: str = '%(asctime)s [%(levelname)s] %(message)s'
    log_file: str = 'logs/backend.log'

# Initialize logging with configuration
if not os.path.exists('logs'):
    os.makedirs('logs')

config = OllamaConfig()
logging.basicConfig(
    level=config.log_level, 
    format=config.log_format, 
    filename=config.log_file
)

class DialogueType(Enum):
    """Types of NPC dialogue responses"""
    GREETING = "greeting"
    REACTION = "reaction"
    COMFORT = "comfort"
    ENCOURAGEMENT = "encouragement"
    WARNING = "warning"
    QUESTION = "question"
    OBSERVATION = "observation"
    STORY = "story"
    JOKE = "joke"
    ADVICE = "advice"
    REFLECTION = "reflection"
    CELEBRATION = "celebration"
    CONCERN = "concern"
    CURIOSITY = "curiosity"
    SUPPORT = "support"
    CALMING = "calming"
    REASSURANCE = "reassurance"

class NPCPersonality(Enum):
    """Different NPC personality types"""
    FRIENDLY = "friendly"
    WISE = "wise"
    PLAYFUL = "playful"
    MYSTERIOUS = "mysterious"
    PROFESSIONAL = "professional"
    CARING = "caring"
    ADVENTUROUS = "adventurous"
    PHILOSOPHICAL = "philosophical"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    CELEBRATORY = "celebratory"
    SUPPORTIVE = "supportive"
    CALMING = "calming"
    REASSURING = "reassuring"
    PROTECTIVE = "protective"
    CURIOUS = "curious"
    EXCITED = "excited"
    UNDERSTANDING = "understanding"
    NEUTRAL = "neutral"
    HELPFUL = "helpful"
    OBSERVANT = "observant"
    ENGAGING = "engaging"

class OllamaIntegration:
    def __init__(self, model="deepseek-r1:latest"):
        self.model = model
        self.dialogue_history = []
        self.current_personality = NPCPersonality.FRIENDLY
        
        # Personality-specific dialogue templates
        self.personality_templates = self._initialize_personality_templates()
        
        # Emotion-specific dialogue strategies
        self.emotion_strategies = self._initialize_emotion_strategies()
    
    def _initialize_personality_templates(self) -> Dict[NPCPersonality, Dict]:
        """Initialize dialogue templates for different personalities"""
        return {
            NPCPersonality.FRIENDLY: {
                "tone": "warm and welcoming",
                "style": "casual and approachable",
                "vocabulary": "simple and clear",
                "examples": ["Hey there!", "How are you doing?", "It's great to see you!"]
            },
            NPCPersonality.WISE: {
                "tone": "thoughtful and insightful",
                "style": "philosophical and reflective",
                "vocabulary": "sophisticated and meaningful",
                "examples": ["Consider this...", "In times like these...", "Wisdom teaches us..."]
            },
            NPCPersonality.PLAYFUL: {
                "tone": "cheerful and energetic",
                "style": "fun and lighthearted",
                "vocabulary": "colorful and expressive",
                "examples": ["Oh wow!", "That's amazing!", "Let's have some fun!"]
            },
            NPCPersonality.MYSTERIOUS: {
                "tone": "enigmatic and intriguing",
                "style": "cryptic and suggestive",
                "vocabulary": "mysterious and poetic",
                "examples": ["Hmm, interesting...", "There's more than meets the eye...", "Secrets abound..."]
            },
            NPCPersonality.PROFESSIONAL: {
                "tone": "formal and respectful",
                "style": "business-like and efficient",
                "vocabulary": "precise and technical",
                "examples": ["I understand.", "Let me assist you.", "How may I help?"]
            },
            NPCPersonality.CARING: {
                "tone": "nurturing and compassionate",
                "style": "gentle and supportive",
                "vocabulary": "caring and empathetic",
                "examples": ["Are you okay?", "I'm here for you.", "Take care of yourself."]
            },
            NPCPersonality.ADVENTUROUS: {
                "tone": "excited and bold",
                "style": "daring and enthusiastic",
                "vocabulary": "dynamic and action-oriented",
                "examples": ["Let's explore!", "What an adventure!", "Ready for anything!"]
            },
            NPCPersonality.PHILOSOPHICAL: {
                "tone": "contemplative and deep",
                "style": "analytical and profound",
                "vocabulary": "intellectual and abstract",
                "examples": ["What does this mean?", "Consider the implications...", "Life is full of lessons..."]
            },
            NPCPersonality.HUMOROUS: {
                "tone": "witty and entertaining",
                "style": "funny and clever",
                "vocabulary": "playful and pun-filled",
                "examples": ["That's hilarious!", "Well, well, well...", "Oh, the irony!"]
            },
            NPCPersonality.SERIOUS: {
                "tone": "grave and focused",
                "style": "direct and no-nonsense",
                "vocabulary": "straightforward and clear",
                "examples": ["This is important.", "Listen carefully.", "Pay attention."]
            },
            NPCPersonality.CELEBRATORY: {
                "tone": "joyful and enthusiastic",
                "style": "festive and celebratory",
                "vocabulary": "uplifting and positive",
                "examples": ["Congratulations!", "This is wonderful!", "Let's celebrate!"]
            },
            NPCPersonality.SUPPORTIVE: {
                "tone": "encouraging and backing",
                "style": "supportive and helpful",
                "vocabulary": "encouraging and positive",
                "examples": ["I'm here for you.", "You can do this!", "I believe in you."]
            },
            NPCPersonality.CALMING: {
                "tone": "soothing and peaceful",
                "style": "calm and reassuring",
                "vocabulary": "gentle and soothing",
                "examples": ["Take a deep breath.", "Everything will be okay.", "Stay calm."]
            },
            NPCPersonality.REASSURING: {
                "tone": "comforting and confident",
                "style": "reassuring and supportive",
                "vocabulary": "comforting and confident",
                "examples": ["You're safe here.", "Don't worry.", "I've got you."]
            },
            NPCPersonality.PROTECTIVE: {
                "tone": "protective and caring",
                "style": "guardian-like and watchful",
                "vocabulary": "protective and caring",
                "examples": ["I'll keep you safe.", "No harm will come to you.", "I'm watching over you."]
            },
            NPCPersonality.CURIOUS: {
                "tone": "inquisitive and interested",
                "style": "curious and exploratory",
                "vocabulary": "questioning and engaging",
                "examples": ["Tell me more!", "What do you think?", "I'm curious about..."]
            },
            NPCPersonality.EXCITED: {
                "tone": "enthusiastic and energetic",
                "style": "excited and animated",
                "vocabulary": "energetic and enthusiastic",
                "examples": ["This is amazing!", "I can't wait!", "How exciting!"]
            },
            NPCPersonality.UNDERSTANDING: {
                "tone": "empathetic and comprehending",
                "style": "understanding and accepting",
                "vocabulary": "empathetic and clear",
                "examples": ["I understand.", "That makes sense.", "I see what you mean."]
            },
            NPCPersonality.NEUTRAL: {
                "tone": "balanced and objective",
                "style": "neutral and impartial",
                "vocabulary": "balanced and clear",
                "examples": ["I see.", "That's interesting.", "Tell me more."]
            },
            NPCPersonality.HELPFUL: {
                "tone": "assisting and supportive",
                "style": "helpful and cooperative",
                "vocabulary": "helpful and clear",
                "examples": ["How can I help?", "Let me assist you.", "I'm here to help."]
            },
            NPCPersonality.OBSERVANT: {
                "tone": "attentive and perceptive",
                "style": "observant and insightful",
                "vocabulary": "perceptive and clear",
                "examples": ["I notice that...", "It seems like...", "I can see that..."]
            },
            NPCPersonality.ENGAGING: {
                "tone": "involving and interactive",
                "style": "engaging and participatory",
                "vocabulary": "interactive and clear",
                "examples": ["What do you think?", "Let's explore this.", "How do you feel about..."]
            }
        }
    
    def _initialize_emotion_strategies(self) -> Dict[str, Dict]:
        """Initialize dialogue strategies for different emotions"""
        return {
            "happy": {
                "primary_types": [DialogueType.CELEBRATION, DialogueType.REACTION, DialogueType.JOKE],
                "secondary_types": [DialogueType.ENCOURAGEMENT, DialogueType.OBSERVATION],
                "tone_modifiers": ["excited", "joyful", "enthusiastic"],
                "avoid_types": [DialogueType.WARNING, DialogueType.CONCERN]
            },
            "sad": {
                "primary_types": [DialogueType.COMFORT, DialogueType.SUPPORT, DialogueType.REFLECTION],
                "secondary_types": [DialogueType.QUESTION, DialogueType.ADVICE],
                "tone_modifiers": ["gentle", "understanding", "compassionate"],
                "avoid_types": [DialogueType.JOKE, DialogueType.CELEBRATION]
            },
            "angry": {
                "primary_types": [DialogueType.CALMING, DialogueType.REFLECTION, DialogueType.ADVICE],
                "secondary_types": [DialogueType.QUESTION, DialogueType.OBSERVATION],
                "tone_modifiers": ["calm", "patient", "understanding"],
                "avoid_types": [DialogueType.JOKE, DialogueType.CELEBRATION]
            },
            "fear": {
                "primary_types": [DialogueType.COMFORT, DialogueType.REASSURANCE, DialogueType.SUPPORT],
                "secondary_types": [DialogueType.QUESTION, DialogueType.ADVICE],
                "tone_modifiers": ["reassuring", "protective", "gentle"],
                "avoid_types": [DialogueType.WARNING]
            },
            "surprise": {
                "primary_types": [DialogueType.REACTION, DialogueType.CURIOSITY, DialogueType.QUESTION],
                "secondary_types": [DialogueType.OBSERVATION, DialogueType.CELEBRATION],
                "tone_modifiers": ["amazed", "curious", "excited"],
                "avoid_types": [DialogueType.CONCERN, DialogueType.WARNING]
            },
            "disgust": {
                "primary_types": [DialogueType.REFLECTION, DialogueType.ADVICE, DialogueType.OBSERVATION],
                "secondary_types": [DialogueType.QUESTION, DialogueType.SUPPORT],
                "tone_modifiers": ["understanding", "helpful", "non-judgmental"],
                "avoid_types": [DialogueType.JOKE, DialogueType.CELEBRATION]
            },
            "neutral": {
                "primary_types": [DialogueType.GREETING, DialogueType.OBSERVATION, DialogueType.QUESTION],
                "secondary_types": [DialogueType.REFLECTION, DialogueType.CURIOSITY],
                "tone_modifiers": ["balanced", "observant", "engaging"],
                "avoid_types": [DialogueType.WARNING, DialogueType.CONCERN]
            }
        }

    def analyze_text_emotion(self, text: str) -> str:
        """Analyzes text to deduce an emotion."""
        prompt = f"""Analyze the emotion of the following text and return only a single emotion from: angry, disgust, fear, happy, neutral, sad, surprise.

Consider the context, tone, and emotional indicators in the text.

Text: "{text}"
Emotion:"""
        response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip().lower()

    def is_model_available(self) -> bool:
        """Checks if the configured Ollama model is available."""
        try:
            # List local models and check if self.model is in the list
            response = ollama.list()
            available_models = []
            
            if hasattr(response, 'models') and response.models:
                available_models = [m.model if hasattr(m, 'model') else str(m) for m in response.models]
            
            # Check exact match first, then base name match
            is_available = (self.model in available_models or 
                          self.model.split(':')[0] in [m.split(':')[0] for m in available_models])
            
            if not is_available:
                logging.error(f"[ERROR] Configured Ollama model '{self.model}' not found. Available models: {available_models}")
            else:
                logging.info(f"[INFO] Ollama model '{self.model}' is available")
                
            return is_available
        except Exception as e:
            logging.error(f"[ERROR] Error checking Ollama model availability: {e}")
            return False

    def generate_npc_dialogue(self, emotion: str, context: str = "", 
                            dialogue_type: Optional[DialogueType] = None,
                            personality: Optional[NPCPersonality] = None) -> str:
        """
        Generates NPC dialogue based on emotion, context, dialogue type, and personality.
        Utilizes Ollama for advanced dialogue generation.
        """
        logging.debug(f"[DEBUG] generate_npc_dialogue called with emotion='{emotion}', context='{context}', dialogue_type='{dialogue_type}', personality='{personality}'")

        if not self.is_model_available():
            logging.error("[ERROR] Ollama model not available, cannot generate dialogue.")
            return "Ollama model is not available for dialogue generation."

        try:
            # Select appropriate dialogue type and personality if not provided
            selected_personality = personality if personality else self._select_appropriate_personality(emotion)
            selected_dialogue_type = dialogue_type if dialogue_type else self._select_dialogue_type(emotion, selected_personality)
        
            personality_template = self.personality_templates.get(selected_personality, self.personality_templates[NPCPersonality.NEUTRAL])
            
            prompt = self._create_dialogue_prompt(
                emotion,
                context,
                selected_dialogue_type,
                personality_template
            )
            logging.debug(f"[DEBUG] Ollama prompt generated:\n'''{prompt}'''")

            # Generate dialogue using Ollama
            response = ollama.generate(model=self.model, prompt=prompt, stream=False)
            
            # Extract and clean up the dialogue from the response
            raw_dialogue = response.get('response', '').strip()
            dialogue = self._clean_dialogue_response(raw_dialogue)
            
            if not dialogue:
                logging.warning(f"[WARNING] Ollama returned empty response for emotion: {emotion}, dialogue_type: {selected_dialogue_type}. Full response: {response}")
                dialogue = self._generate_fallback_dialogue(emotion, selected_dialogue_type, personality_template)
            
            logging.debug(f"[DEBUG] Ollama raw response: {response}") # Raw response for full debugging
            logging.debug(f"[DEBUG] Generated dialogue: '{dialogue}'")
            
            if dialogue:
                self.dialogue_history.append({'role': 'assistant', 'content': dialogue, 'response': dialogue})
            return dialogue
        except Exception as e:
            logging.error(f"[ERROR] Unexpected error during dialogue generation: {e}")
            import traceback
            traceback.print_exc()
            return f"I'm a little confused. Can we try again? (Error: {str(e)[:50]}...)"

    def _select_appropriate_personality(self, emotion: str) -> NPCPersonality:
        """Select an appropriate personality based on the emotion."""
        personality_mapping = {
            "happy": [NPCPersonality.PLAYFUL, NPCPersonality.FRIENDLY, NPCPersonality.CELEBRATORY],
            "sad": [NPCPersonality.CARING, NPCPersonality.WISE, NPCPersonality.SUPPORTIVE],
            "angry": [NPCPersonality.WISE, NPCPersonality.CALMING, NPCPersonality.PROFESSIONAL],
            "fear": [NPCPersonality.CARING, NPCPersonality.REASSURING, NPCPersonality.PROTECTIVE],
            "surprise": [NPCPersonality.PLAYFUL, NPCPersonality.CURIOUS, NPCPersonality.EXCITED],
            "disgust": [NPCPersonality.UNDERSTANDING, NPCPersonality.NEUTRAL, NPCPersonality.HELPFUL],
            "neutral": [NPCPersonality.FRIENDLY, NPCPersonality.OBSERVANT, NPCPersonality.ENGAGING]
        }
        
        # Get available personalities for this emotion, with fallback to FRIENDLY
        available_personalities = personality_mapping.get(emotion, [NPCPersonality.FRIENDLY])
        
        # Filter out any personalities that don't exist in the templates
        valid_personalities = [p for p in available_personalities if p in self.personality_templates]
        
        # If no valid personalities found, use FRIENDLY as default
        if not valid_personalities:
            valid_personalities = [NPCPersonality.FRIENDLY]
        
        return random.choice(valid_personalities)

    def _select_dialogue_type(self, emotion: str, personality: NPCPersonality) -> DialogueType:
        """Select an appropriate dialogue type based on emotion and personality."""
        emotion_strategy = self.emotion_strategies.get(emotion, self.emotion_strategies["neutral"])
        
        try:
            # Weight primary types more heavily
            if random.random() < 0.7:
                return random.choice(emotion_strategy["primary_types"])
            else:
                return random.choice(emotion_strategy["secondary_types"])
        except Exception as e:
            logging.error(f"Error selecting dialogue type: {e}")
            # Return a safe default
            return DialogueType.REACTION

    def _create_dialogue_prompt(self, emotion: str, context: str, dialogue_type: DialogueType, 
                               personality_template: Dict) -> str:
        """Create a detailed prompt for dialogue generation."""
        
        dialogue_type_prompts = {
            DialogueType.GREETING: "greet the user warmly",
            DialogueType.REACTION: "react to the user's emotional state",
            DialogueType.COMFORT: "offer comfort and support",
            DialogueType.ENCOURAGEMENT: "provide encouragement and motivation",
            DialogueType.WARNING: "give a gentle warning or caution",
            DialogueType.QUESTION: "ask an engaging question",
            DialogueType.OBSERVATION: "make an insightful observation",
            DialogueType.STORY: "share a brief, relevant story or anecdote",
            DialogueType.JOKE: "make a light-hearted joke or comment",
            DialogueType.ADVICE: "offer helpful advice",
            DialogueType.REFLECTION: "reflect on the situation thoughtfully",
            DialogueType.CELEBRATION: "celebrate the user's positive state",
            DialogueType.CONCERN: "express gentle concern",
            DialogueType.CURIOSITY: "show genuine curiosity",
            DialogueType.SUPPORT: "offer unwavering support"
        }
        
        prompt = f"""You are an NPC in a VR game. Generate a SHORT dialogue response (maximum 15 words).

User emotion: {emotion}
Dialogue type: {dialogue_type.value}
Tone: {personality_template['tone']}
Context: {context}

Requirements:
- Maximum 15 words
- Be {personality_template['tone']}
- {dialogue_type_prompts.get(dialogue_type, 'respond appropriately')}
- Do not use thinking tags or explanations
- Just give the direct dialogue

NPC says:"""
        return prompt

    def _clean_dialogue_response(self, raw_dialogue: str) -> str:
        """Clean and validate dialogue response from Ollama."""
        if not raw_dialogue:
            return ""
        
        # Remove <think> tags and everything inside them
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', raw_dialogue, flags=re.DOTALL)
        
        # Remove common prefixes that Ollama might add
        prefixes_to_remove = [
            "Dialogue:", "Response:", "NPC:", "Assistant:", 
            "AI:", "Bot:", "Character:", "Reply:", "NPC says:"
        ]
        
        cleaned = cleaned.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove quotes if the entire response is quoted
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        
        # Take only the first sentence if multiple sentences
        sentences = cleaned.split('. ')
        if len(sentences) > 1:
            cleaned = sentences[0] + '.'
        
        # Ensure reasonable length (truncate if too long)
        max_length = 100
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + '...'
        
        return cleaned

    def _generate_fallback_dialogue(self, emotion: str, dialogue_type: DialogueType, 
                                   personality_template: Dict) -> str:
        """Generate a fallback dialogue using templates when Ollama fails."""
        
        fallback_templates = {
            DialogueType.GREETING: {
                "happy": "Hello there! You seem cheerful today!",
                "sad": "Hi... I'm here if you need to talk.",
                "angry": "Hello. I can see you're frustrated.",
                "fear": "Hey there. You're safe here.",
                "surprise": "Wow! You look amazed!",
                "disgust": "Hello. Something bothering you?",
                "neutral": "Hello! How are you doing?"
            },
            DialogueType.COMFORT: {
                "sad": "I'm here for you. It's okay.",
                "fear": "You're safe here with me.",
                "angry": "I understand you're upset.",
                "default": "I care about how you're feeling."
            },
            DialogueType.ENCOURAGEMENT: {
                "happy": "Your positive energy is amazing!",
                "sad": "You're stronger than you know.",
                "angry": "Your feelings are valid.",
                "fear": "You're braver than you think.",
                "default": "You've got this! I believe in you."
            }
        }
        
        # Get appropriate template
        templates = fallback_templates.get(dialogue_type, fallback_templates.get(DialogueType.GREETING, {}))
        dialogue = templates.get(emotion, templates.get("default", "Hello! How are you feeling?"))
        
        return dialogue

    def get_dialogue_history(self) -> List[Dict]:
        """Get the history of generated dialogues."""
        return self.dialogue_history.copy()

    def clear_dialogue_history(self):
        """Clear the dialogue history."""
        self.dialogue_history.clear()

    def get_dialogue_statistics(self) -> Dict:
        """Get statistics about generated dialogues."""
        if not self.dialogue_history:
            return {"total": 0, "emotions": {}, "types": {}, "personalities": {}}
        
        stats = {
            "total": len(self.dialogue_history),
            "emotions": {},
            "types": {},
            "personalities": {},
            "average_length": 0
        }
        
        # Calculate statistics from dialogue history
        total_length = 0
        for entry in self.dialogue_history:
            content = entry.get('content', '')
            total_length += len(content)
        
        if self.dialogue_history:
            stats["average_length"] = total_length / len(self.dialogue_history)
        
        return stats