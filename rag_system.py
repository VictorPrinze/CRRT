import os
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama
from crrt_config import Config

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = Config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'device': self.config.DEVICE}
        )
        self.vector_store = None
        self.llm = self._load_language_model()
        
    def _load_language_model(self) -> Llama:
        """Load the Mistral-7B model with llama-cpp-python, enabling GPU offloading."""
        try:
            llm = Llama(
                model_path=self.config.MODEL_PATH,
                n_ctx=self.config.LLM_CONTEXT_LENGTH,
                n_threads=1,
                n_gpu_layers=35,  # Offload layers to GPU (T4 can handle this)
                verbose=False
            )
            self.logger.info("Language model loaded successfully with GPU support")
            return llm
        except Exception as e:
            self.logger.error(f"Error loading language model: {e}")
            raise
    
    def load_vector_store(self, documents: List[Any], faiss_index: FAISS, embedding_model: Any):
        """Load an existing FAISS vector store."""
        try:
            self.vector_store = faiss_index
            self.logger.info("Vector store loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            raise

    def load_and_index_documents(self, documents: List[str]):
        """Split and index documents into the vector store."""
        try:
            chunks = self.text_splitter.create_documents(documents)
            self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
            self.logger.info("Documents indexed successfully")
        except Exception as e:
            self.logger.error(f"Error indexing documents: {e}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, k: int = Config.TOP_K_MATCHES) -> List[str]:
        """Retrieve top-k relevant document chunks for a query."""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {e}")
            raise
    
    def truncate_context(self, context: List[str], max_tokens: int) -> str:
        """Truncate the context to fit within max_tokens (approximated by characters)."""
        context_str = "\n".join(context)
        max_chars = max_tokens * 4  # Approximate: 1 token ≈ 4 characters
        if len(context_str) > max_chars:
            context_str = context_str[:max_chars] + "..."
        return context_str

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate an answer using the Mistral model with a refined prompt."""
        try:
            context_str = self.truncate_context(context, self.config.MAX_CONTEXT_TOKENS)
            prompt = f"""<|startoftext|>You are a medical AI assistant specializing in Continuous Renal Replacement Therapy (CRRT). Based on the following context, provide a concise and accurate answer to the query. Use precise medical terminology and focus on key terms relevant to the query (e.g., use "citrate" instead of "regional citrate anticoagulation" if appropriate). Ensure completeness (e.g., list all modalities if asked).

**Context:**
{context_str}

**Query:**
{query}

**Answer (max {self.config.MAX_ANSWER_TOKENS} words):**"""
            
            response = self.llm(
                prompt,
                max_tokens=self.config.MAX_ANSWER_TOKENS,
                temperature=0.2,  # Lowered for more deterministic answers
                top_p=0.9,
                top_k=40,
                stop=["<|endoftext|>"]
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        try:
            self.logger.debug(f"Retrieving chunks for query: {query}")
            context = self.retrieve_relevant_chunks(query)
            self.logger.debug(f"Generating answer for query: {query}")
            answer = self.generate_answer(query, context)
            self.logger.debug(f"Answer generated: {answer}")
            return {
                "query": query,
                "answer": answer,
                "context": context
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise