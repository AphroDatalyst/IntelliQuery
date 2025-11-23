"""
Utility functions for setup verification.
"""

import sys
import subprocess
from pathlib import Path


def check_ollama():
    """Check if Ollama is installed and running."""
    print("🔍 Checking Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0:
            print(f"✅ Ollama: {result.stdout.strip()}")
            
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    if models:
                        print(f"✅ Service running. Models: {', '.join([m.get('name', '') for m in models])}")
                        return True, models
                    else:
                        print("⚠️  No models. Run: ollama pull llama2")
                        return True, []
                else:
                    print("❌ Service not responding")
                    return False, []
            except ImportError:
                print("⚠️  Install 'requests' to check service")
                return True, []
            except Exception as e:
                print(f"❌ Cannot connect: {e}")
                return False, []
        else:
            print("❌ Ollama not installed")
            return False, []
    except FileNotFoundError:
        print("❌ Ollama not found. Install from: https://ollama.ai/")
        return False, []
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, []


def check_dependencies():
    """Check Python dependencies."""
    print("\n🔍 Checking dependencies...")
    required = ["langchain", "langchain_community", "ollama", "chromadb"]
    missing = []
    
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    # Check for unstructured (OCR support)
    try:
        import unstructured  # noqa: F401
        print("✅ unstructured (OCR support)")
    except ImportError:
        print("⚠️  unstructured (OCR support) - optional but recommended")
    
    # Check for Tesseract OCR (system dependency)
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0] if result.stdout else "installed"
            print(f"✅ tesseract-ocr ({version})")
        else:
            print("⚠️  tesseract-ocr - not found (required for OCR)")
    except (FileNotFoundError, Exception):
        print("⚠️  tesseract-ocr - not found (required for OCR)")
        print("   Install: See README.md for system-specific instructions")
    
    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}")
        print("   Install: pip install -r requirements.txt")
        return False
    
    print("✅ All core dependencies installed")
    return True


def check_documents():
    """Check documents folder."""
    print("\n🔍 Checking documents...")
    docs_folder = Path("docs")
    
    if not docs_folder.exists():
        docs_folder.mkdir(exist_ok=True)
        print(f"⚠️  Created: {docs_folder}")
        return False
    
    files = []
    for ext in [".pdf", ".txt", ".md"]:
        files.extend(list(docs_folder.glob(f"*{ext}")))
        files.extend(list(docs_folder.glob(f"**/*{ext}")))
    
    if not files:
        print(f"⚠️  No documents in {docs_folder}")
        return False
    
    print(f"✅ Found {len(files)} document(s)")
    for f in files[:5]:
        print(f"   - {f.name}")
    return True


def create_test_document():
    """Create test document if needed."""
    docs_folder = Path("docs")
    test_file = docs_folder / "test_document.txt"
    
    if not test_file.exists():
        test_content = """Test Document for RAG Agent

This document tests the RAG (Retrieval-Augmented Generation) system.

Key Points:
1. RAG uses Ollama for local LLM inference
2. Documents are stored in a vector database
3. The system retrieves relevant context for questions
4. LLM generates answers based on retrieved context

Technologies:
- Ollama: Local LLM
- LangChain: Orchestration
- ChromaDB: Vector database
- Supports PDF, TXT, and Markdown files
"""
        test_file.write_text(test_content)
        print(f"✅ Created: {test_file}")
    else:
        print(f"✅ Test document exists: {test_file}")


def main():
    """Run all checks."""
    print("=" * 60)
    print("🧪 RAG Agent Setup Test")
    print("=" * 60)
    
    ollama_ok, _ = check_ollama()
    deps_ok = check_dependencies()
    has_docs = check_documents()
    
    if not has_docs:
        create_test_document()
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    if ollama_ok and deps_ok:
        print("✅ Setup complete!")
        print("\nUsage:")
        print("  rag-agent --mode ingest")
        print("  rag-agent --mode query --question 'What is RAG?'")
        print("  rag-agent --mode interactive")
    else:
        print("❌ Setup incomplete. Fix issues above.")
    
    print("=" * 60)
    return ollama_ok and deps_ok


if __name__ == "__main__":
    try:
        import requests  # noqa: F401
    except ImportError:
        print("⚠️  Installing 'requests'...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=False)
    
    success = main()
    sys.exit(0 if success else 1)

