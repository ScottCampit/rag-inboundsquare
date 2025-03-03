import os
import requests
from dotenv import load_dotenv
import arxiv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def process_pdf(local_path: str = "./react.pdf") -> None:
    """Download and summarize the ReAct paper using RAG."""
    try:
        # Download PDF if needed
        if not os.path.exists(local_path):
            print("Downloading ReAct paper...")
            client = arxiv.Client()
            paper = next(client.results(arxiv.Search(id_list=["2210.03629"])))
            print(f"Downloading: {paper.title}")
            paper.download_pdf(filename=local_path, dirpath=".")
            print(f"PDF saved to {local_path}")

        # Load and split PDF
        documents = PyPDFLoader(local_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(documents)
        
        # Setup RAG chain
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory="db"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that creates concise and accurate "
                      "summaries of documents. Use the following context to create a summary. "
                      "If you don't know the answer, just say that you don't know.\n\n"
                      "Context: {context}\n\n"
                      "Please provide a clear and concise summary of the document."),
        ])
        
        chain = create_retrieval_chain(
            vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            create_stuff_documents_chain(
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                prompt=prompt
            )
        )
        
        # Generate summary
        response = chain.invoke({
            "input": "Please provide a comprehensive summary of this document."
        })
        print("\nDocument Summary:")
        print("-" * 50)
        print(response["answer"])
        
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Process the ReAct paper."""
    load_dotenv()
    process_pdf()


if __name__ == "__main__":
    main()