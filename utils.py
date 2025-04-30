"""
Utility functions for the Medical CRRT LLM project.
"""

import os
import json
import logging
import torch
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crrt_llm")

def check_cuda() -> Tuple[bool, str]:
    """Check if CUDA is available and return device."""
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    logger.info(f"Using device: {device}")
    return has_cuda, device

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def load_pdfs_from_directory(directory: str) -> Dict[str, str]:
    """Load all PDFs from a directory and extract their text."""
    pdf_texts = {}
    
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return pdf_texts
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            logger.info(f"Processing PDF: {filename}")
            pdf_texts[filename] = extract_text_from_pdf(filepath)
    
    logger.info(f"Loaded {len(pdf_texts)} PDF documents")
    return pdf_texts

def load_json(file_path: str) -> Any:
    """Load JSON data from a file."""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def save_json(data: Any, file_path: str) -> bool:
    """Save data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False

def format_sources(sources: List[Dict[str, Any]]) -> List[str]:
    """Format source information for display."""
    formatted_sources = []
    
    for i, source in enumerate(sources, 1):
        source_text = f"Source {i}: "
        
        if "metadata" in source:
            metadata = source["metadata"]
            if "source" in metadata:
                source_text += f"{os.path.basename(metadata['source'])}"
            if "page" in metadata:
                source_text += f" (Page {metadata['page']})"
        
        formatted_sources.append(source_text)
    
    return formatted_sources

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and special characters."""
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    
    # Remove special characters that might affect comparison
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    return text.strip()

def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Calculate accuracy based on exact matches."""
    if not predictions or not ground_truth:
        return 0.0
    
    if len(predictions) != len(ground_truth):
        logger.warning(f"Prediction length ({len(predictions)}) doesn't match ground truth length ({len(ground_truth)})")
        # Truncate to the shorter length
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
    
    # Normalize answers for comparison
    norm_predictions = [normalize_text(p) for p in predictions]
    norm_ground_truth = [normalize_text(g) for g in ground_truth]
    
    # Calculate exact matches
    correct = sum(1 for p, g in zip(norm_predictions, norm_ground_truth) if p == g)
    return correct / len(ground_truth)

def extract_answer_from_text(text: str) -> str:
    """Extract the answer part from a generated text."""
    # This function helps extract just the answer part from the model's response
    # which might contain prompt artifacts
    
    # If the text contains "Your answer:" or similar markers, extract the part after that
    markers = ["Your answer:", "Answer:", "Assistant:", "AI:"]
    
    for marker in markers:
        if marker in text:
            answer_part = text.split(marker, 1)[1].strip()
            return answer_part
    
    # If no markers found, return the original text
    return text.strip()

def similarity_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    embedding1 = embedding1 / norm1
    embedding2 = embedding2 / norm2
    
    # Calculate cosine similarity
    return np.dot(embedding1, embedding2)

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    if not prediction or not ground_truth:
        return 0.0
    
    # Normalize and tokenize
    pred_tokens = set(normalize_text(prediction).lower().split())
    truth_tokens = set(normalize_text(ground_truth).lower().split())
    
    # Calculate precision, recall, F1
    common_tokens = pred_tokens.intersection(truth_tokens)
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def timer_decorator(func):
    """Decorator to time function execution."""
    import time
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result
    
    return wrapper

def validate_model_paths(model_paths: Dict[str, Dict[str, str]]) -> Dict[str, bool]:
    """Validate that model paths exist."""
    results = {}
    
    for model_name, model_info in model_paths.items():
        if model_info.get("is_hf_model", False):
            # HF models don't need local path check
            results[model_name] = True
        else:
            path = model_info.get("path")
            results[model_name] = os.path.exists(path) if path else False
    
    return results

def create_sample_qa_data() -> List[Dict[str, str]]:
    """Create sample QA data for testing."""
    return [
        {
            "question": "What is CRRT?",
            "answer": "Continuous Renal Replacement Therapy (CRRT) is a dialysis modality used for critically ill patients with acute kidney injury (AKI) who cannot tolerate conventional intermittent hemodialysis. It provides continuous, gradual fluid and solute removal over 24 hours, making it suitable for hemodynamically unstable patients."
        },
        {
            "question": "What are the main indications for CRRT?",
            "answer": "The main indications for CRRT include: acute kidney injury (AKI) in hemodynamically unstable patients, fluid overload unresponsive to diuretics, severe metabolic acidosis, certain drug overdoses, hyperkalemia refractory to medical management, and selected cases of rhabdomyolysis with AKI."
        },
        {
            "question": "What are the different modalities of CRRT?",
            "answer": "The main CRRT modalities include: CVVH (Continuous Veno-Venous Hemofiltration), CVVHD (Continuous Veno-Venous Hemodialysis), CVVHDF (Continuous Veno-Venous Hemodiafiltration), and SCUF (Slow Continuous Ultrafiltration). Each modality uses different principles of diffusion and convection for solute clearance."
        }
    ]

def get_requirements() -> List[str]:
    """Return a list of requirements for the project."""
    return [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "langchain>=0.0.267",
        "langchain-community>=0.0.10",
        "sentence-transformers>=2.2.2",
        "llama-cpp-python>=0.2.0",
        "pymupdf>=1.23.0",
        "faiss-cpu>=1.7.4",
        "peft>=0.4.0",
        "datasets>=2.12.0",
        "bitsandbytes>=0.40.0",
        "streamlit>=1.22.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "scikit-learn>=1.2.0",
        "accelerate>=0.21.0"
    ]