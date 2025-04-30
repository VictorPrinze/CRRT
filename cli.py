import os
import time
import logging
import json
from typing import List
import PyPDF2
from colorama import init, Fore, Style
from crrt_config import Config
from rag_system import RAGSystem
from document_processor import DocumentProcessor

# Initialize colorama for terminal colors
init()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CRRTCLI:
    def __init__(self, config: Config):
        """Initialize CLI with configuration, RAG system, and document processor."""
        self.config = config
        self.rag_system = RAGSystem()
        self.doc_processor = DocumentProcessor(config)
        self.is_initialized = False
        self.chat_history = []

    def load_pdf_documents(self, data_dir: str) -> List[str]:
        """Load and extract text from PDF files in the data directory."""
        documents = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, "rb") as file:
                        pdf = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text() or ""
                            text += page_text
                        documents.append(text)
                        logging.info(f"Loaded document: {filename}")
                except Exception as e:
                    logging.error(f"Error loading {filename}: {e}")
        return documents

    def setup_system(self, pdf_dir: str = None):
        """Set up the RAG system by loading or processing PDFs."""
        if pdf_dir is None:
            pdf_dir = self.config.DATA_DIR
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"PDF directory {pdf_dir} does not exist")

        # Check for existing vector store
        chunks_path = os.path.join(self.config.VECTORS_DIR, "chunks.json")
        faiss_path = os.path.join(self.config.VECTORS_DIR, "faiss_index.bin")
        if os.path.exists(chunks_path) and os.path.exists(faiss_path):
            try:
                self.doc_processor.load_chunks(chunks_path)
                self.doc_processor.load_faiss_index(faiss_path)
                self.doc_processor.load_embedding_model()
                self.rag_system.load_vector_store(self.doc_processor.chunks, self.doc_processor.faiss_index, self.doc_processor.embedding_model)
                self.is_initialized = True
                logging.info("Loaded existing vector store")
                print(f"{Fore.GREEN}✓ Loaded existing knowledge base{Fore.RESET}")
                return
            except Exception as e:
                logging.warning(f"Failed to load vector store: {e}. Reprocessing PDFs...")

        # Load and process PDFs if no valid vector store
        print(f"{Fore.YELLOW}Processing CRRT documents...{Fore.RESET}")
        documents = self.load_pdf_documents(pdf_dir)
        if not documents:
            raise ValueError("No documents loaded")

        # Index documents
        try:
            chunks, faiss_index = self.doc_processor.process_pdfs(pdf_dir)
            self.rag_system.load_vector_store(chunks, faiss_index, self.doc_processor.embedding_model)
            self.doc_processor.save_chunks(chunks_path)
            self.doc_processor.save_faiss_index(faiss_path)
            self.is_initialized = True
            logging.info("PDFs processed and indexed successfully")
            print(f"{Fore.GREEN}✓ Knowledge base created successfully{Fore.RESET}")
        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise

    def interactive_mode(self):
        """Run the interactive CLI mode for question answering with enhanced formatting."""
        print(f"\n{Fore.CYAN}🌟 CRRT Medical AI Assistant 🌟{Fore.RESET}")
        print("Specialized in Continuous Renal Replacement Therapy")
        print("Commands: 'exit' to quit, 'help' for info, 'debug' to toggle logs, 'setup' to reprocess PDFs, 'save' to save chat history")
        print(f"{Fore.MAGENTA}------------------------------------{Fore.RESET}\n")

        debug_mode = False
        if not self.is_initialized:
            print("Initializing CRRT knowledge base...")
            try:
                self.setup_system()
                print("Knowledge base ready!")
            except Exception as e:
                print(f"{Fore.RED}Setup failed: {e}{Fore.RESET}")
                return

        while True:
            try:
                user_input = input(f"{Fore.BLUE}Ask about CRRT: {Fore.RESET}").strip()
                if user_input.lower() == 'exit':
                    print(f"\n{Fore.GREEN}Thank you for using CRRT Medical AI. Goodbye!{Fore.RESET}")
                    break
                elif user_input.lower() == 'help':
                    print(f"\n{Fore.CYAN}CRRT Medical AI Assistant{Fore.RESET}")
                    print("Ask questions about CRRT (e.g., 'What is CRRT?').")
                    print("Commands: 'exit' to quit, 'help' for this message, 'debug' to toggle logs, 'setup' to reprocess PDFs, 'save' to save chat history")
                    continue
                elif user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
                    continue
                elif user_input.lower() == 'setup':
                    pdf_dir = input("Enter CRRT PDF directory (e.g., /home/user/data): ")
                    try:
                        self.setup_system(pdf_dir)
                        print(f"{Fore.GREEN}Knowledge base updated!{Fore.RESET}")
                    except Exception as e:
                        print(f"{Fore.RED}Setup failed: {e}{Fore.RESET}")
                    continue
                elif user_input.lower() == 'save':
                    history_file = os.path.join(self.config.RESULTS_DIR, "chat_history.json")
                    with open(history_file, "w") as f:
                        json.dump(self.chat_history, f, indent=4)
                    print(f"{Fore.GREEN}Chat history saved to {history_file}{Fore.RESET}")
                    continue
                elif not user_input:
                    print(f"{Fore.YELLOW}Please ask a question or use a command.{Fore.RESET}")
                    continue

                print(f"\n{Fore.YELLOW}Analyzing", end="")
                for _ in range(3):
                    print(".", end="", flush=True)
                    time.sleep(0.5)
                print(f"{Fore.RESET}\n")

                # Process the query
                result = self.rag_system.process_query(user_input)

                # Display the Q&A in a well-organized format
                print(f"{Fore.BLUE}Question: {Fore.RESET}{user_input}")
                print(f"{Fore.GREEN}Answer: {Fore.RESET}{result['answer']}\n")

                # Creative Thinking: Suggest a related insight or alternative approach
                related_insight = self.generate_related_insight(user_input, result['context'])
                print(f"{Fore.MAGENTA}Did you know?{Fore.RESET} {related_insight}\n")

                if debug_mode:
                    print(f"{Fore.CYAN}Context:{Fore.RESET}")
                    for i, ctx in enumerate(result['context'], 1):
                        print(f"{i}. {ctx}")
                    print()

                # Save to chat history
                self.chat_history.append({
                    "question": user_input,
                    "answer": result['answer'],
                    "context": result['context'],
                    "insight": related_insight
                })

            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Fore.RESET}")
                print("Please try again or type 'help'.")
                logging.error(f"CLI error: {e}")

    def generate_related_insight(self, query: str, context: List[str]) -> str:
        """Generate a creative insight or suggestion based on the query and context."""
        context_str = "\n".join(context)
        insight_prompt = f"""<|startoftext|>You are a medical AI assistant specializing in Continuous Renal Replacement Therapy (CRRT). Based on the following context and query, provide a concise, creative insight or suggestion related to CRRT (e.g., an alternative approach, a clinical tip, or a summary of a related concept). Keep it under 50 words.

**Context:**
{context_str}

**Query:**
{query}

**Insight (max 50 words):**"""
        
        try:
            response = self.rag_system.llm(
                insight_prompt,
                max_tokens=50,
                temperature=0.5,
                top_p=0.9,
                top_k=50,
                stop=["<|endoftext|>"]
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logging.error(f"Error generating insight: {e}")
            return "Consider consulting recent CRRT guidelines for the latest best practices."

if __name__ == "__main__":
    Config.ensure_directories()
    cli = CRRTCLI(Config)
    cli.interactive_mode()