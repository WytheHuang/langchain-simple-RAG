# langchain-simple-RAG

A simple Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama, and PyPDF. This system allows users to ask questions about PDF documents and receive AI-generated answers based on the document content.

## Features

- PDF document loading and processing
- Document chunking for efficient retrieval
- Vector embeddings using Ollama's BGE-M3 model
- Question answering using Llama 3.2 Vision (11B parameters)
- Interactive command-line interface
- Source attribution for answers

## Prerequisites

- Python 3.12 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Required models pulled in Ollama:
  - `bge-m3` for embeddings
  - `llama3.2-vision:11b` for text generation

## Installation

1. Clone this repository:
```bash
git clone https://github.com/WytheHuang/langchain-simple-RAG.git
cd langchain-simple-RAG
```

1. Set up a Python virtual and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

1. Install the required packages:
```bash
uv sync
```

## Usage

1. Create a `documents` directory in your project root and place your PDF files there:
```bash
mkdir documents
cp /path/to/your/pdfs/*.pdf documents/
```

2. Run the main script:
```bash
python main.py
```

3. Enter your questions when prompted. Type `/quit` to exit the program.

Example interaction:
```
Loading documents...
Loaded 25 document chunks

Enter your question (or '/quit' to exit): What is machine learning?

Answer: [AI-generated answer will appear here]

Sources:
Source: documents/example.pdf
Page: 1
Content: [Relevant excerpt from the document]
```

## Project Structure

```
langchain-simple-RAG/
├── documents/            # Directory for PDF files
├── main.py              # Main application script
├── README.md            # This file
├── pyproject.toml       # Project configuration
└── uv.lock             # Dependencies lock file
```

## How It Works

1. **Document Loading**: The system loads PDF documents from the `documents` directory using `PyPDFLoader`.

2. **Document Splitting**: Documents are split into smaller chunks using `RecursiveCharacterTextSplitter` for efficient processing.

3. **Embedding**: Document chunks are embedded using Ollama's BGE-M3 model and stored in an in-memory vector store.

4. **Question Answering**: When a user asks a question:
   - The system retrieves relevant document chunks
   - Passes them to the Llama 3.2 model
   - Generates a contextual answer
   - Provides source attribution

## Configuration

The system uses the following default settings:

- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Embedding model: `bge-m3`
- LLM model: `llama3.2-vision:11b`

## Development

To contribute to this project:

1. Install development dependencies:
```bash
uv sync --group lint
```

2. Run linters:
```bash
ruff check .
black .
```

## License

APACHE-2.0 License
See the [LICENSE](LICENSE) file for details.

