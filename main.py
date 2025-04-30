import os
import sys
import json
import logging
import torch
from datetime import datetime
from typing import List, Dict, Any

from crrt_config import Config
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from model_trainer import CRRTModelTrainer
from model_evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.BASE_DIR, "crrt_system.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment, create directories, etc."""
    Config.ensure_directories()
    
    # Check GPU availability
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU memory: {total_memory:.2f} GB")
        
        if total_memory > Config.MIN_MEMORY_GB:
            Config.DEVICE = "cuda"
            logger.info(f"Using device: {Config.DEVICE}")
        else:
            logger.warning(f"GPU memory ({total_memory:.2f} GB) less than required minimum ({Config.MIN_MEMORY_GB} GB). Using CPU.")
    else:
        logger.info("No GPU detected. Using CPU.")

def process_data():
    """Process the PDF documents and create vector store."""
    processor = DocumentProcessor(Config)
    
    # Check if we have already processed data
    chunks_path = os.path.join(Config.VECTORS_DIR, "chunks.json")
    faiss_path = os.path.join(Config.VECTORS_DIR, "faiss_index")
    
    if os.path.exists(chunks_path) and os.path.exists(faiss_path):
        logger.info("Loading existing processed data...")
        processor.load_chunks(chunks_path)
        processor.load_faiss_index(faiss_path)
        logger.info("Existing data loaded successfully.")
    else:
        logger.info("Processing PDF documents...")
        chunks, faiss_index = processor.process_pdfs(Config.DATA_DIR)
        processor.save_chunks(chunks_path)
        processor.save_faiss_index(faiss_path)
        logger.info(f"Processed {len(chunks)} chunks from PDFs.")
    
    return processor

def train_model():
    """Fine-tune the base model on CRRT data."""
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = CRRTModelTrainer(Config)
    
    # Check if we already have a fine-tuned model
    finetuned_model_path = Config.FINETUNED_MODEL_PATH
    
    if os.path.exists(finetuned_model_path):
        logger.info("Loading existing fine-tuned model...")
        trainer.load_model(finetuned_model_path)
    else:
        logger.info("Fine-tuning model on CRRT data...")
        # Process data for training
        processor = process_data()
        training_data = trainer.prepare_training_data(processor.chunks)
        
        # Train the model
        trained_model = trainer.train(training_data)
        
        # Save the fine-tuned model
        trainer.save_model(finetuned_model_path)
        logger.info(f"Model fine-tuned and saved to {finetuned_model_path}")
    
    return trainer

def evaluate_models():
    """Evaluate the fine-tuned model against others."""
    logger.info("Starting model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(Config)
    
    # Load test questions
    test_file_path = os.path.join(Config.DATA_DIR, Config.TEST_FILE)
    if not os.path.exists(test_file_path):
        logger.error(f"Test file not found at {test_file_path}")
        create_test_questions(test_file_path)
    
    # Set up reference answers
    evaluator.setup_reference_answers()
    
    # Load our fine-tuned model
    crrt_system = RAGSystem(Config)
    
    # Assign the processed FAISS index
    processor = process_data()
    crrt_system.vector_store = processor.faiss_index
    
    # Evaluate our model
    logger.info("Evaluating CRRT specialized model...")
    crrt_results = evaluator.evaluate_model(crrt_system, "CRRT-Specialized")
    
    # Evaluate general LLMs
    general_llm_results = evaluator.evaluate_general_llms()
    
    # Evaluate medical LLMs
    medical_llm_results = evaluator.evaluate_medical_llms()
    
    # Compare results
    all_results = [crrt_results] + general_llm_results + medical_llm_results
    comparison_results = evaluator.compare_results(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Generate evaluation report
    report_file = os.path.join(Config.RESULTS_DIR, f"evaluation_report_{timestamp}.md")
    evaluator.generate_evaluation_report(comparison_results, report_file)
    
    logger.info(f"Evaluation results saved to {results_file}")
    logger.info(f"Evaluation report saved to {report_file}")
    
    return comparison_results

def create_test_questions(output_path):
    """Create a set of test questions for CRRT evaluation."""
    logger.info("Creating test question set...")
    
    # Basic set of test questions for CRRT
    test_questions = [
        {
            "id": 1,
            "question": "What are the primary indications for initiating CRRT?",
            "category": "indications",
            "difficulty": "basic"
        },
        {
            "id": 2,
            "question": "Describe the difference between CVVH, CVVHD, and CVVHDF modes of CRRT.",
            "category": "operational",
            "difficulty": "intermediate"
        },
        {
            "id": 3,
            "question": "What is the recommended anticoagulation protocol for patients on CRRT with heparin-induced thrombocytopenia?",
            "category": "anticoagulation",
            "difficulty": "advanced"
        },
        {
            "id": 4,
            "question": "How should fluid removal targets be calculated for a patient on CRRT?",
            "category": "fluid_management",
            "difficulty": "intermediate"
        },
        {
            "id": 5,
            "question": "What are the common complications associated with CRRT and how can they be prevented?",
            "category": "complications",
            "difficulty": "intermediate"
        },
        {
            "id": 6,
            "question": "Explain the significance of filter pressure monitoring during CRRT.",
            "category": "monitoring",
            "difficulty": "intermediate"
        },
        {
            "id": 7,
            "question": "What is the optimal dosing for CRRT in septic patients?",
            "category": "dosing",
            "difficulty": "advanced"
        },
        {
            "id": 8,
            "question": "How does CRRT affect drug dosing and pharmacokinetics?",
            "category": "pharmacology",
            "difficulty": "advanced"
        },
        {
            "id": 9,
            "question": "What are the nursing responsibilities in managing a patient on CRRT?",
            "category": "nursing",
            "difficulty": "basic"
        },
        {
            "id": 10,
            "question": "Compare and contrast the different vascular access sites for CRRT.",
            "category": "vascular_access",
            "difficulty": "intermediate"
        },
        {
            "id": 11,
            "question": "How should electrolyte abnormalities be managed during CRRT?",
            "category": "electrolytes",
            "difficulty": "intermediate"
        },
        {
            "id": 12,
            "question": "What are the criteria for CRRT discontinuation?",
            "category": "discontinuation",
            "difficulty": "basic"
        },
        {
            "id": 13,
            "question": "Describe the setup procedure for a CRRT machine.",
            "category": "operational",
            "difficulty": "basic"
        },
        {
            "id": 14,
            "question": "What are the differences between CRRT and intermittent hemodialysis?",
            "category": "comparison",
            "difficulty": "basic"
        },
        {
            "id": 15,
            "question": "How should nutrition be managed for patients on CRRT?",
            "category": "nutrition",
            "difficulty": "intermediate"
        },
        {
            "id": 16,
            "question": "Explain the mechanisms of solute clearance in CRRT.",
            "category": "principles",
            "difficulty": "advanced"
        },
        {
            "id": 17,
            "question": "What are the recommended replacement fluid compositions for CRRT?",
            "category": "fluids",
            "difficulty": "intermediate"
        },
        {
            "id": 18,
            "question": "How does CRRT impact acid-base balance?",
            "category": "acid_base",
            "difficulty": "advanced"
        },
        {
            "id": 19,
            "question": "What troubleshooting steps should be taken for high transmembrane pressure alarms?",
            "category": "troubleshooting",
            "difficulty": "intermediate"
        },
        {
            "id": 20,
            "question": "Describe the principles of regional citrate anticoagulation in CRRT.",
            "category": "anticoagulation",
            "difficulty": "advanced"
        }
    ]
    
    # Save test questions to file
    with open(output_path, 'w') as f:
        json.dump(test_questions, f, indent=2)
    
    logger.info(f"Created {len(test_questions)} test questions and saved to {output_path}")

def interactive_mode(rag_system: RAGSystem):
    """Run interactive mode to allow user queries."""
    chat_history = []
    print("Entering Interactive Mode. Type 'exit' to quit, 'debug' to see context, or 'save' to save chat history.")
    
    while True:
        query = input("Ask about CRRT: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'save':
            history_path = os.path.join(Config.RESULTS_DIR, 'chat_history.json')
            try:
                with open(history_path, 'w') as f:
                    json.dump(chat_history, f, indent=4)
                print(f"Chat history saved to {history_path}")
            except Exception as e:
                print(f"Error saving chat history: {e}")
            continue
        elif query.lower() == 'debug':
            if chat_history:
                last_entry = chat_history[-1]
                print("Context for last query:")
                print(last_entry.get('context', 'No context available'))
            else:
                print("No previous queries to debug.")
            continue
        
        try:
            result = rag_system.answer_question(query)
            print(f"Answer: {result['answer']}\n")
            print(f"Retrieval Time: {result['retrieval_time_seconds']}s")
            print(f"Generation Time: {result['generation_time_seconds']}s")
            print(f"Total Time: {result['total_time_seconds']}s\n")
            
            chat_history.append({
                'query': query,
                'answer': result['answer'],
                'context': result['context_used']
            })
        except Exception as e:
            print(f"Error generating answer: {e}")
            chat_history.append({
                'query': query,
                'answer': f"Error: {e}",
                'context': ''
            })

def main():
    """Main function to run the CRRT Medical LLM system."""
    logger.info("Starting CRRT Medical LLM System")
    
    # Set up environment
    setup_environment()
    
    # Process data
    processor = process_data()
    
    # Train model
    trainer = train_model()
    
    # Initialize RAG system
    rag_system = RAGSystem(Config)
    rag_system.vector_store = processor.faiss_index
    
    # Evaluate models
    results = evaluate_models()
    
    # Print summary of results
    print("\n=== EVALUATION RESULTS SUMMARY ===")
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and 'accuracy' in model_results:
            print(f"{model_name}: Accuracy {model_results['accuracy']*100:.2f}%")
    
    # Run interactive mode
    interactive_mode(rag_system)
    
    logger.info("CRRT Medical LLM System completed successfully")

if __name__ == "__main__":
    main()