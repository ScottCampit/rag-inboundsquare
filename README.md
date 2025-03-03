# RAG Paper Summarizer

A Python application that implements Retrieval Augmented Generation (RAG) to download and summarize academic papers. Currently configured to process the ReAct paper from arXiv.

## Features

- Automatic paper download from arXiv
- PDF processing and text chunking
- Vector store creation using Chroma
- RAG-based summarization using OpenAI's GPT-4 and LangChain

## Prerequisites

- Python 3.x
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-inboundsquare
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the script using:

```bash
python rag.py
```

The script will:
1. Download the ReAct paper if not already present
2. Process the PDF and split it into chunks
3. Create a vector store using Chroma
4. Generate a comprehensive summary using RAG

## Dependencies

- langchain
- openai
- chromadb
- arxiv
- python-dotenv
- requests

## Note

The current implementation is configured to summarize the ReAct paper (arXiv:2210.03629). You can modify the `process_pdf` function to work with other papers or PDF documents.

## License

[Add your license here] 