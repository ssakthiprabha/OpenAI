#!/usr/bin/env python3
"""
API Configuration for Logo Generation Pipeline

Set your OpenAI API key here to use the pipeline.
"""

# Set your OpenAI API key here
OPENAI_API_KEY = "My Open API key"

# OpenAI Model Configuration
OPENAI_MODEL = "gpt-3.5-turbo" ## "gpt-4"
DALLE_MODEL = "dall-e-3"

# Pipeline Configuration
MAX_LOGO_CANDIDATES = 5
EVALUATION_CRITERIA_WEIGHTS = {
    "clarity": 0.25,
    "relevance": 0.25,
    "creativity": 0.25,
    "simplicity": 0.25
}

def get_api_key():
    """Get the OpenAI API key."""
    if OPENAI_API_KEY == "your_openai_api_key_here":
        print("⚠️  Warning: Please set your OpenAI API key in api_config.py")
        print("   Replace 'your_openai_api_key_here' with your actual API key")
        return None
    return OPENAI_API_KEY

def validate_config():
    """Validate the configuration."""
    if not get_api_key():
        print("❌ Configuration validation failed: API key not set")
        return False
    
    # Check evaluation weights sum to 1.0
    weight_sum = sum(EVALUATION_CRITERIA_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.001:
        print(f"❌ Evaluation weights must sum to 1.0, got {weight_sum}")
        return False
    
    print("✅ Configuration validation passed")
    return True

if __name__ == "__main__":
    print("🔧 API Configuration")
    print("=" * 30)
    print(f"OpenAI Model: {OPENAI_MODEL}")
    print(f"DALL-E Model: {DALLE_MODEL}")
    print(f"API Key Set: {'Yes' if get_api_key() else 'No'}")
    print(f"Logo Candidates: {MAX_LOGO_CANDIDATES}")
    print(f"Evaluation Weights: {EVALUATION_CRITERIA_WEIGHTS}")
    print("=" * 30)
    
    validate_config()
