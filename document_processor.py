import os
import json
import logging
import PyPDF2
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from crrt_config import Config

class DocumentProcessor:
    def __init__(self, config: Config):
        """Initialize DocumentProcessor with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        self.embedding_model = None
        self.chunks = None
        self.faiss_index = None
        self.load_embedding_model()

    def load_embedding_model(self):
        """Load the embedding model."""
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': self.config.DEVICE}
            )
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise

    def process_pdfs(self, pdf_dir: str) -> Tuple[List[Document], FAISS]:
        """Process PDFs and create a vector store."""
        documents = []
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_dir, filename)
                try:
                    with open(filepath, "rb") as file:
                        pdf = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text() or ""
                            text += page_text
                        documents.append(text)
                        self.logger.info(f"Processed PDF: {filename}")
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
        
        if not documents:
            raise ValueError("No documents processed")
        
        # Split documents into chunks
        self.chunks = self.text_splitter.create_documents(documents)
        # Create FAISS index
        self.faiss_index = FAISS.from_documents(self.chunks, self.embedding_model)
        return self.chunks, self.faiss_index

    def load_chunks(self, chunks_path: str):
        """Load saved chunks."""
        try:
            with open(chunks_path, "r") as f:
                chunks_data = json.load(f)
            self.chunks = [Document(page_content=doc) for doc in chunks_data]
            self.logger.info("Chunks loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading chunks: {e}")
            raise

    def save_chunks(self, chunks_path: str):
        """Save chunks to a file."""
        try:
            chunks_data = [doc.page_content for doc in self.chunks]
            with open(chunks_path, "w") as f:
                json.dump(chunks_data, f)
            self.logger.info("Chunks saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving chunks: {e}")
            raise

    def load_faiss_index(self, faiss_path: str):
        """Load FAISS index from file."""
        try:
            self.faiss_index = FAISS.load_local(faiss_path, self.embedding_model, allow_dangerous_deserialization=True)
            self.logger.info("FAISS index loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {e}")
            raise

    def save_faiss_index(self, faiss_path: str):
        """Save FAISS index to file."""
        try:
            self.faiss_index.save_local(faiss_path)
            self.logger.info("FAISS index saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving FAISS index: {e}")
            raise