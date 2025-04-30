import os

class Config:
    # File paths (adjusted for Google Colab)
    BASE_DIR = "/content/drive/MyDrive/CRRT_Medical_System"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    VECTORS_DIR = os.path.join(BASE_DIR, "vectors")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    TEST_FILE = "test_questions.json"
    VECTOR_STORE_PATH = VECTORS_DIR
    
    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Use PyTorch model for fine-tuning
    LLM_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # GGUF for inference
    MODEL_PATH = os.path.join(MODELS_DIR, LLM_FILE)
    FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, "finetuned_crrt_model")
    
    # RAG parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K_MATCHES = 10
    MAX_CONTEXT_TOKENS = 1200
    MAX_ANSWER_TOKENS = 150
    LLM_CONTEXT_LENGTH = 2048
    
    # System parameters
    DEVICE = "cpu"  # Use CPU for simplicity in Colab's free tier
    MIN_MEMORY_GB = 2.8
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_name in [cls.DATA_DIR, cls.MODELS_DIR, cls.VECTORS_DIR, cls.RESULTS_DIR]:
            os.makedirs(dir_name, exist_ok=True)