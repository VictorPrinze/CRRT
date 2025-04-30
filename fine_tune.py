"""
Fine-tuning the LLM on medical CRRT data.
"""

import os
import json
import torch
from typing import Dict, List, Any
import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

from config import TRAIN_CONFIG, MODEL_CONFIG, OUTPUT_DIR
from utils import logger, check_cuda, load_json

class ModelFineTuner:
    """Fine-tune the LLM model on medical CRRT data."""
    
    def __init__(self, train_config: Dict[str, Any] = None):
        """Initialize the fine-tuner."""
        self.train_config = train_config or TRAIN_CONFIG
        self.has_cuda, self.device = check_cuda()
        
        # Set up paths
        self.output_dir = os.path.join(OUTPUT_DIR, "fine_tuned_model")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # We'll use HF Transformers instead of llama.cpp for fine-tuning
        # This might need modifications based on your available GPU memory
        self.model_name = MODEL_CONFIG["model_name"].split("/")[-1]
        if "/" not in self.model_name:
            # If we're using a local GGUF model, we need to use a corresponding HF model
            if "llama" in self.model_name.lower():
                self.model_name = "meta-llama/Llama-3-8B-Instruct"
            elif "mistral" in self.model_name.lower():
                self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            else:
                self.model_name = "Qwen/Qwen1.5-7B-Chat"  # Fallback
    
    def prepare_training_data(self, qa_data_path: str) -> Dataset:
        """Prepare training data from QA pairs."""
        try:
            qa_data = load_json(qa_data_path)
            
            # Format the data for instruction fine-tuning
            formatted_data = []
            for item in qa_data:
                if "question" in item and "answer" in item:
                    formatted_data.append({
                        "text": f"""<|im_start|>system
You are a medical AI assistant specializing in Continuous Renal Replacement Therapy (CRRT).
<|im_end|>
<|im_start|>user
{item['question']}
<|im_end|>
<|im_start|>assistant
{item['answer']}
<|im_end|>"""
                    })
            
            logger.info(f"Prepared {len(formatted_data)} examples for training")
            return Dataset.from_list(formatted_data)
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return Dataset.from_list([])
    
    def fine_tune(self, dataset: Dataset) -> None:
        """Fine-tune the model using PEFT/LoRA."""
        if len(dataset) == 0:
            logger.error("No training data available")
            return
        
        try:
            # Use 4-bit quantization to reduce memory requirements
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
            
            # Set up LoRA configuration
            lora_config = LoraConfig(
                r=16,  # Rank
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Get PEFT model
            model = get_peft_model(model, lora_config)
            
            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=2048  # Adjust based on your GPU memory
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.train_config["epochs"],
                per_device_train_batch_size=1,  # Adjust based on your GPU memory
                gradient_accumulation_steps=4,
                learning_rate=self.train_config["learning_rate"],
                weight_decay=self.train_config["weight_decay"],
                warmup_steps=self.train_config["warmup_steps"],
                save_strategy="steps",
                save_steps=self.train_config["save_steps"],
                logging_dir=os.path.join(self.output_dir, "logs"),
                logging_steps=10,
                fp16=True,
                gradient_checkpointing=True,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                )
            )
            
            # Train model
            logger.info("Starting fine-tuning...")
            trainer.train()
            
            # Save model and adapter
            model.save_pretrained(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Model fine-tuned and saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise
    
    def convert_to_gguf(self, output_path: str = None) -> None:
        """Convert the fine-tuned model to GGUF format for inference."""
        if not output_path:
            output_path = os.path.join(self.output_dir, f"{self.model_name.split('/')[-1]}-finetuned.gguf")
        
        try:
            logger.info(f"Converting model to GGUF format: {output_path}")
            # This is a placeholder - actual conversion would require llama.cpp or similar tool
            # In practice, you might call a subprocess to run the conversion
            # subprocess.run(["python", "-m", "llama_cpp.convert", ...])
            
            # For now, we'll just log that this would need to be done manually
            logger.info("GGUF conversion needs to be done manually using llama.cpp tools")
            logger.info("Example command: python -m llama_cpp.convert /path/to/hf/model /path/to/output.gguf")
            
            return output_path
        except Exception as e:
            logger.error(f"Error converting model to GGUF format: {e}")
            return None

if __name__ == "__main__":
    from config import DATA_DIR
    
    fine_tuner = ModelFineTuner()
    qa_data_path = os.path.join(DATA_DIR, "test_questions.json")
    
    # Prepare training data
    dataset = fine_tuner.prepare_training_data(qa_data_path)
    
    # Fine-tune model
    if len(dataset) > 0:
        fine_tuner.fine_tune(dataset)
        
        # Optionally convert to GGUF format
        # fine_tuner.convert_to_gguf()
    else:
        logger.error("No training data available, skipping fine-tuning")