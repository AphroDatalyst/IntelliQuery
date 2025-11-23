"""
Document Loader - Handles loading of PDF, TXT, and Markdown files.
Supports automatic OCR for image-based PDFs.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
)


class DocumentLoader:
    """Loader for various document formats with automatic OCR support."""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".markdown": UnstructuredMarkdownLoader,
    }
    
    # Threshold for triggering OCR (average chars per page)
    OCR_THRESHOLD = 50
    
    def _load_pdf_with_ocr_fallback(self, file_path: str) -> List[Document]:
        """
        Load PDF with automatic OCR fallback.
        First tries PyPDFLoader, then falls back to OCR if insufficient text.
        """
        documents: List[Document] = []
        
        # Try PyPDFLoader first (fast for text-based PDFs)
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Check if we got sufficient text
            if documents:
                total_chars = sum(len(doc.page_content.strip()) for doc in documents)
                avg_chars_per_page = total_chars / len(documents) if documents else 0
                
                if avg_chars_per_page >= self.OCR_THRESHOLD:
                    # Sufficient text extracted, no OCR needed
                    return documents
                else:
                    # Insufficient text, likely image-based PDF
                    print(f"📷 Low text detected ({avg_chars_per_page:.0f} chars/page), using OCR...")
            
        except Exception as e:
            print(f"⚠️  PyPDFLoader failed: {e}, trying OCR...")
        
        # Fall back to UnstructuredPDFLoader with OCR
        try:
            loader = UnstructuredPDFLoader(str(file_path), mode="elements")
            documents = loader.load()
            
            # Mark documents as OCR-processed
            for doc in documents:
                doc.metadata["ocr_used"] = True
            
            print("✅ OCR extraction successful")
            return documents
            
        except Exception as e:
            # If OCR also fails, return original PyPDFLoader result or raise
            if documents:
                print(f"⚠️  OCR failed, using partial text extraction: {e}")
                return documents
            else:
                raise RuntimeError(f"Both PyPDFLoader and OCR failed: {e}") from e
    
    def load_directory(self, directory_path: str, exclude_patterns: List[str] = None) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
            exclude_patterns: List of filename patterns to exclude (e.g., ['README.md', '*.txt'])
        """
        if exclude_patterns is None:
            exclude_patterns = ['README.md', 'readme.md', 'LICENSE', 'CHANGELOG.md']
        
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory_path}")
            return documents
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                # Skip excluded files
                if any(file_path.name == pattern or file_path.match(pattern) for pattern in exclude_patterns):
                    print(f"⏭️  Skipping excluded file: {file_path.name}")
                    continue
                try:
                    # Special handling for PDFs with OCR fallback
                    if file_path.suffix.lower() == ".pdf":
                        loaded_docs = self._load_pdf_with_ocr_fallback(str(file_path))
                    else:
                        loader_class = self.SUPPORTED_EXTENSIONS[file_path.suffix.lower()]
                        loader = loader_class(str(file_path))
                        loaded_docs = loader.load()
                    
                    # Add metadata
                    for doc in loaded_docs:
                        doc.metadata["source"] = str(file_path)
                        doc.metadata["filename"] = file_path.name
                    
                    documents.extend(loaded_docs)
                    ocr_status = " (OCR)" if any(doc.metadata.get("ocr_used", False) for doc in loaded_docs) else ""
                    print(f"✅ Loaded: {file_path.name} ({len(loaded_docs)} pages{ocr_status})")
                except Exception as e:
                    print(f"❌ Error loading {file_path.name}: {e}")
                    continue
        
        return documents
    
    def load_file(self, file_path: str) -> List[Document]:
        """Load a single document file."""
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path_obj.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_path_obj.suffix}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        # Special handling for PDFs with OCR fallback
        if file_path_obj.suffix.lower() == ".pdf":
            documents = self._load_pdf_with_ocr_fallback(str(file_path))
        else:
            loader_class = self.SUPPORTED_EXTENSIONS[file_path_obj.suffix.lower()]
            loader = loader_class(file_path)
            documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["filename"] = file_path_obj.name
        
        return documents

