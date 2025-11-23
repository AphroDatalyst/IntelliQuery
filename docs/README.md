# Documents Folder

Place your documents (PDF, TXT, MD) in this folder for the RAG agent to process.

## Supported Formats

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`, `.markdown`)

## Usage

1. Add your documents to this folder
2. Run: `python main.py --mode ingest`
3. Start querying: `python main.py --mode interactive`

The agent will automatically:
- Load all supported documents
- Split them into chunks
- Create embeddings
- Store them in the vector database



