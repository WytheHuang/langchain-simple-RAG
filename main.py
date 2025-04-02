from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM


def load_documents(directory_path: Path) -> list[Document]:
    """Load PDF documents from a specified directory.

    Args:
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        _type_: _description_
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory_path}")

    documents = []
    pdf_files = [f for f in directory.iterdir() if f.suffix == ".pdf"]

    if not pdf_files:
        raise ValueError("No PDF files found in directory")

    for pdf_file in pdf_files:
        file_path = pdf_file
        documents.extend(PyPDFLoader(file_path).load())

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for processing.

    Args:
        documents (list[Document]): List of documents to be split.

    Returns:
        list[Document]: List of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def main():
    """Main function to load documents, split them, and set up the QA system."""
    print("Loading documents...")

    docs_path = "./documents"
    documents = load_documents(docs_path)
    splits = split_documents(documents)

    print(f"Loaded {len(splits)} document chunks")

    embedder = OllamaEmbeddings(model="bge-m3")
    vectorstore = InMemoryVectorStore.from_documents(splits, embedder)
    retriever = vectorstore.as_retriever()

    llm = OllamaLLM(model="llama3.2-vision:11b")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    while True:
        query = input("\nEnter your question (or '/quit' to exit): ")

        if query.lower() == "/quit":
            break

        result = qa_chain.invoke({"query": query})

        print("\nAnswer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"\nSource: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
