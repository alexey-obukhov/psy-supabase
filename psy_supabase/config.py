"""Configuration file for the psy_supabase programm."""

DEFAULT_TOPIC = "supportive_listening"
DEFAULT_EMOTION = "concern"
DEFAULT_APPROACH = "empathy_validation"
DEFAULT_THEME = "general_support"

TEXT_GENERATING_MODEL = "rasyosef/Phi-1_5-Instruct-v0.1"  # Beaware templates should be adjusted regarding this model
TOXIC_CLASSIFICATION_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
